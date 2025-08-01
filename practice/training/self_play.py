#!/usr/bin/env python3
"""
Self-play data generation for AlphaZero-style training

Generates training data by having the current model play against itself
using MCTS. Collects (state, MCTS policy, game outcome) tuples.
"""

import sys
import os
import pickle
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_board import GameBoard
from mcts.mcts_search import create_alphazero_mcts, create_basic_mcts

try:
    import torch
    from core.alphazero_net import create_alphazero_net, AlphaZeroNet
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    TORCH_AVAILABLE = False
    sys.exit(1)

@dataclass
class SelfPlayDataPoint:
    """Data point from self-play games."""
    board_features: np.ndarray  # (10, 17, 2) neural network features
    mcts_policy: np.ndarray     # MCTS visit count policy  
    valid_moves: List[Tuple[int, int, int, int]]  # Valid moves for this position
    game_result: float          # Final game result from this player's perspective
    player: int                 # Player who was to move (0 or 1)
    turn_number: int           # Turn number in the game

def heuristic_value_evaluation(board: GameBoard, player: int) -> float:
    """
    Simple heuristic value evaluation function to replace neural network.
    
    Args:
        board: Current game board state
        player: Player to evaluate for (0 or 1)
        
    Returns:
        Value between -1 and 1 (positive = good for player, negative = bad)
    """
    try:
        # Get valid moves for current player
        valid_moves = board.get_valid_moves()
        
        # If no valid moves, return negative value
        if not valid_moves or (len(valid_moves) == 1 and valid_moves[0] == (-1, -1, -1, -1)):
            return -0.5
        
        # Calculate total area of available rectangles
        total_area = 0
        move_count = 0
        
        for move in valid_moves:
            if move == (-1, -1, -1, -1):  # Skip pass moves
                continue
            r1, c1, r2, c2 = move
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            total_area += area
            move_count += 1
        
        # Simple heuristic: more moves and larger areas = better position
        if move_count == 0:
            return -0.3
        
        # Normalize the value
        avg_area = total_area / move_count if move_count > 0 else 0
        
        # Value based on available moves and average area
        # More moves = better, larger average area = better
        value = (move_count * 0.1 + avg_area * 0.05) / 10.0
        
        # Clamp between -1 and 1
        value = max(-1.0, min(1.0, value))
        
        return value
        
    except Exception as e:
        # Return neutral value on error
        return 0.0

class SelfPlayGenerator:
    """Generates self-play training data using MCTS + neural network."""
    
    def __init__(self, 
                 model: AlphaZeroNet = None,
                 mcts_simulations: int = 400,
                 mcts_time: float = 2.0,
                 temperature: float = 1.0,
                 temperature_threshold: int = 10):
        """
        Initialize self-play generator.
        
        Args:
            model: Combined policy-value network (None for random play)
            mcts_simulations: Number of MCTS simulations per move
            mcts_time: Time limit per move
            temperature: Temperature for move selection (higher = more exploration)
            temperature_threshold: Turn after which to use temperature=0 (deterministic)
        """
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.mcts_time = mcts_time
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        
        # Track current game state for heuristic evaluation
        self._current_board = None
        self._current_player = 0
        
        print(f"SelfPlayGenerator initialized:")
        print(f"  Model: {'Neural' if model else 'Random'}")
        print(f"  MCTS simulations: {mcts_simulations}")
        print(f"  Temperature: {temperature} (→0 after turn {temperature_threshold})")
    
    def _create_mcts_for_model(self, model: AlphaZeroNet):
        """Create MCTS instance that uses the given model."""
        
        def combined_net_function(features, valid_moves):
            """Interface function for MCTS to use the model."""
            try:
                if model is None:
                    # Uniform policy, heuristic value
                    uniform_policy = {move: 1.0/len(valid_moves) for move in valid_moves} if valid_moves else {}
                    # Use heuristic value evaluation instead of neutral 0.0
                    heuristic_value = heuristic_value_evaluation(self._current_board, self._current_player)
                    return uniform_policy, heuristic_value
                
                model.eval()
                with torch.no_grad():
                    # Get policy from neural network, but use heuristic for value
                    move_probs, _ = model.predict_policy_value(features, valid_moves)  # Ignore neural value
                    heuristic_value = heuristic_value_evaluation(self._current_board, self._current_player)
                    return move_probs, heuristic_value
            except Exception as e:
                print(f"Model inference error: {e}")
                uniform_policy = {move: 1.0/len(valid_moves) for move in valid_moves} if valid_moves else {}
                # Use heuristic value even in error case
                try:
                    heuristic_value = heuristic_value_evaluation(self._current_board, self._current_player)
                except:
                    heuristic_value = 0.0
                return uniform_policy, heuristic_value
        
        if model is not None:
            return create_alphazero_mcts(
                combined_net_function=combined_net_function,
                max_simulations=self.mcts_simulations,
                max_time=self.mcts_time,
                exploration_weight=1.0
            )
        else:
            return create_basic_mcts(
                max_simulations=self.mcts_simulations,
                max_time=self.mcts_time
            )
    
    def play_game(self, board_data: List[List[int]], verbose: bool = False) -> List[SelfPlayDataPoint]:
        """
        Play a single self-play game and collect training data.
        
        Args:
            board_data: Initial board configuration
            verbose: Whether to print game progress
            
        Returns:
            List of training data points from the game
        """
        game_board = GameBoard(board_data)
        current_player = 0
        turn_number = 0
        max_turns = 200  # Prevent infinite games
        
        # Create MCTS for current model
        mcts = self._create_mcts_for_model(self.model)
        
        # Store game data
        game_data = []
        
        if verbose:
            print(f"\nStarting self-play game...")
        
        while not game_board.is_game_over() and turn_number < max_turns:
            # Update current state for heuristic evaluation
            self._current_board = game_board
            self._current_player = current_player
            
            # Get valid moves
            valid_moves = game_board.get_valid_moves()
            
            if len(valid_moves) <= 1 and valid_moves[0] == (-1, -1, -1, -1):
                # Only pass move available - game should end soon
                break
            
            # Get current board features
            board_features = game_board.to_neural_features(current_player)  # (10, 17, 2)
            
            # Run MCTS to get move probabilities
            best_move, search_stats = mcts.search(
                initial_state=game_board,
                player=current_player,
                max_time=self.mcts_time
            )
            
            # Get MCTS policy (visit counts normalized)
            visit_counts = search_stats.get('visit_counts', {})
            
            # Convert to policy vector for valid moves
            mcts_policy = np.zeros(len(valid_moves))
            total_visits = 0
            
            for i, move in enumerate(valid_moves):
                visits = visit_counts.get(move, 0)
                mcts_policy[i] = visits
                total_visits += visits
            
            # Normalize based on visit counts
            if total_visits > 0:
                mcts_policy = mcts_policy / total_visits
            else:
                mcts_policy = np.ones(len(valid_moves)) / len(valid_moves)
            
            # Store training data (result will be filled in later)
            data_point = SelfPlayDataPoint(
                board_features=board_features.copy(),
                mcts_policy=mcts_policy.copy(),
                valid_moves=valid_moves.copy(),
                game_result=0.0,  # Will be updated after game ends
                player=current_player,
                turn_number=turn_number
            )
            game_data.append(data_point)
            
            # Select move using temperature
            selected_move = self._select_move_with_temperature(valid_moves, mcts_policy, turn_number)
            
            if verbose:
                move_str = f"({selected_move[0]},{selected_move[1]})-({selected_move[2]},{selected_move[3]})" if selected_move[0] != -1 else "PASS"
                sims = search_stats.get('simulations', 0) if 'search_stats' in locals() else 0
                print(f"Turn {turn_number}: Player {current_player} plays {move_str} ({sims} sims)")
            
            # Make the move
            success = game_board.make_move(*selected_move, current_player)
            if not success:
                print(f"Warning: Move {selected_move} failed!")
                break
            
            # Switch players
            current_player = 1 - current_player
            turn_number += 1
        
        # Game ended - assign results
        winner = game_board.get_winner()
        if verbose:
            scores = game_board.get_score()
            print(f"Game ended after {turn_number} turns. Scores: P0={scores[0]}, P1={scores[1]}, Winner: {winner}")
        
        # Update game results for all data points
        for data_point in game_data:
            if winner == data_point.player:
                data_point.game_result = 1.0  # Win
            elif winner == -1:
                data_point.game_result = 0.0  # Tie
            else:
                data_point.game_result = -1.0  # Loss
        
        return game_data
    
    def _select_move_with_temperature(self, valid_moves: List, policy: np.ndarray, turn_number: int) -> Tuple[int, int, int, int]:
        """Select move using temperature-based sampling."""
        if turn_number >= self.temperature_threshold:
            # Deterministic selection (temperature = 0)
            best_idx = np.argmax(policy)
            return valid_moves[best_idx]
        else:
            # Temperature-based sampling
            if self.temperature == 0:
                best_idx = np.argmax(policy)
                return valid_moves[best_idx]
            
            # Apply temperature
            temp_policy = np.power(policy, 1.0 / self.temperature)
            temp_policy = temp_policy / temp_policy.sum()
            
            # Sample from the distribution
            selected_idx = np.random.choice(len(valid_moves), p=temp_policy)
            return valid_moves[selected_idx]
    
    def generate_self_play_data(self, 
                               board_data: List[List[int]], 
                               num_games: int = 100,
                               verbose: bool = True) -> List[SelfPlayDataPoint]:
        """
        Generate self-play training data.
        
        Args:
            board_data: Initial board configuration
            num_games: Number of games to play
            verbose: Whether to print progress
            
        Returns:
            List of all training data points
        """
        all_data = []
        start_time = time.time()
        
        print(f"Generating {num_games} self-play games...")
        
        for game_num in range(num_games):
            if verbose and game_num % 10 == 0:
                elapsed = time.time() - start_time
                games_per_min = (game_num + 1) / (elapsed / 60) if elapsed > 0 else 0
                print(f"Game {game_num + 1}/{num_games} ({games_per_min:.1f} games/min)")
            
            try:
                game_data = self.play_game(board_data, verbose=(game_num < 3))  # Verbose for first few games
                all_data.extend(game_data)
                
            except Exception as e:
                print(f"Error in game {game_num}: {e}")
                continue
        
        total_time = time.time() - start_time
        avg_data_per_game = len(all_data) / num_games if num_games > 0 else 0
        
        print(f"\nSelf-play generation complete:")
        print(f"  Games played: {num_games}")
        print(f"  Total data points: {len(all_data)}")
        print(f"  Avg data points per game: {avg_data_per_game:.1f}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Time per game: {total_time/num_games:.2f}s")
        
        return all_data

def load_board_config(input_path: str = "practice/testing/input.txt") -> List[List[int]]:
    """Load board configuration from file."""
    full_path = input_path
    if not os.path.exists(full_path):
        # Try relative to this file
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testing", "input.txt")
    
    with open(full_path, 'r') as f:
        lines = f.readlines()
    
    board = []
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.strip()]
            board.append(row)
    
    return board

def main():
    """Main function for self-play data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-play data generation')
    parser.add_argument('--games', type=int, default=50, help='Number of games to play')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--time', type=float, default=1.0, help='Time limit per move')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move selection')
    parser.add_argument('--temp-threshold', type=int, default=10, help='Turn to switch to deterministic')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='self_play_data.pkl', help='Output file')
    parser.add_argument('--random', action='store_true', help='Use random play (no neural network)')
    
    args = parser.parse_args()
    
    # Load board configuration
    try:
        board_data = load_board_config()
        print(f"Loaded board: {len(board_data)}x{len(board_data[0])}")
    except Exception as e:
        print(f"Error loading board: {e}")
        return
    
    # Load model if specified
    model = None
    if not args.random and args.model and os.path.exists(args.model):
        try:
            print(f"Loading model from {args.model}")
            model = create_alphazero_net('cpu')
            checkpoint = torch.load(args.model, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}, using random play")
            model = None
    elif not args.random:
        print("No model specified or found, using random play")
    
    # Create self-play generator
    generator = SelfPlayGenerator(
        model=model,
        mcts_simulations=args.simulations,
        mcts_time=args.time,
        temperature=args.temperature,
        temperature_threshold=args.temp_threshold
    )
    
    # Generate data
    self_play_data = generator.generate_self_play_data(
        board_data=board_data,
        num_games=args.games,
        verbose=True
    )
    
    # Save data
    dataset = {
        'data_points': self_play_data,
        'generation_info': {
            'num_games': args.games,
            'mcts_simulations': args.simulations,
            'temperature': args.temperature,
            'model_used': args.model if model else 'random',
            'generation_time': time.time()
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSelf-play data saved to {args.output}")
    print(f"Dataset contains {len(self_play_data)} training examples")

if __name__ == "__main__":
    main()