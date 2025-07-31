#!/usr/bin/env python3
"""
MCTS-based training data generator for policy network

Generates high-quality training data using Monte Carlo Tree Search self-play.
Data consists of board states and corresponding MCTS visit count distributions.
"""

import sys
import os
import time
import pickle
import random
from typing import List, Dict, Tuple, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/Users/brillight/Documents/CodingProject/NYPC/practice')

from core.game_board import GameBoard
from mcts.mcts_search import create_basic_mcts
# Action encoder not needed for simple approach


class TrainingDataPoint:
    """Single training data point."""
    
    def __init__(self, board_features: np.ndarray, move_probabilities: np.ndarray, 
                 valid_moves: List[Tuple[int, int, int, int]], game_result: int):
        self.board_features = board_features  # (10, 17, 7) neural net input  
        self.move_probabilities = move_probabilities  # MCTS visit count distribution
        self.valid_moves = valid_moves  # List of valid moves for this position
        self.game_result = game_result  # 1 if current player won, -1 if lost, 0 if tie


class MCTSDataGenerator:
    """Generates training data using MCTS self-play."""
    
    def __init__(self, 
                 mcts_simulations: int = 400,
                 mcts_time_limit: float = 2.0,
                 temperature: float = 1.0,
                 temperature_decay_moves: int = 10):
        """
        Initialize data generator.
        
        Args:
            mcts_simulations: Number of MCTS simulations per move
            mcts_time_limit: Time limit for MCTS search (seconds)
            temperature: Temperature for move selection (1.0 = stochastic, 0.0 = deterministic)
            temperature_decay_moves: Moves after which to decay temperature to 0
        """
        self.mcts_simulations = mcts_simulations
        self.mcts_time_limit = mcts_time_limit
        self.temperature = temperature
        self.temperature_decay_moves = temperature_decay_moves
# Action encoder not needed for simple approach
        
        # Performance tracking
        self.games_generated = 0
        self.total_positions = 0
        
    def generate_self_play_game(self, board_data: List[List[int]], 
                              game_id: int = 0) -> List[TrainingDataPoint]:
        """
        Generate a single self-play game using MCTS.
        
        Args:
            board_data: Initial board configuration (10x17 grid)
            game_id: Identifier for this game (for logging)
            
        Returns:
            List of training data points from this game
        """
        print(f"Starting self-play game {game_id}", file=sys.stderr)
        
        # Initialize game board and MCTS
        game_board = GameBoard(board_data)
        mcts_p1 = create_basic_mcts(max_simulations=self.mcts_simulations, 
                                   max_time=self.mcts_time_limit)
        mcts_p2 = create_basic_mcts(max_simulations=self.mcts_simulations, 
                                   max_time=self.mcts_time_limit)
        
        data_points = []
        move_count = 0
        current_player = 0
        
        max_moves = 500  # Safety limit to prevent infinite games
        while not game_board.is_game_over() and move_count < max_moves:
            move_count += 1
            
            # Select MCTS for current player
            mcts = mcts_p1 if current_player == 0 else mcts_p2
            
            # Get valid moves
            valid_moves = game_board.get_valid_moves()
            
            # Calculate temperature for this move
            current_temp = self.temperature if move_count <= self.temperature_decay_moves else 0.0
            
            try:
                # Handle forced pass move
                if len(valid_moves) == 1 and valid_moves[0] == (-1, -1, -1, -1):
                    # Only pass move available - create minimal training data
                    selected_move = (-1, -1, -1, -1)
                    board_features = game_board.to_neural_features()
                    move_probs = np.array([1.0])  # 100% probability for pass
                    
                    data_point = TrainingDataPoint(
                        board_features=board_features,
                        move_probabilities=move_probs,
                        valid_moves=valid_moves.copy(),
                        game_result=0  # Will be filled in at game end
                    )
                    data_points.append((data_point, current_player))
                    
                    print(f"Game {game_id}, Move {move_count}: Player {current_player} -> FORCED PASS", file=sys.stderr)
                    
                else:
                    # Run MCTS search for multiple move options
                    best_move, search_stats = mcts.search(
                        initial_state=game_board, 
                        player=current_player,
                        max_time=self.mcts_time_limit
                    )
                    
                    # Get visit count distribution from MCTS
                    visit_counts = self._extract_visit_counts(mcts, valid_moves)
                    
                    # Convert to probability distribution using temperature
                    move_probs = self._visit_counts_to_probabilities(visit_counts, current_temp)
                    
                    # Store training data point
                    board_features = game_board.to_neural_features()
                    data_point = TrainingDataPoint(
                        board_features=board_features,
                        move_probabilities=move_probs,
                        valid_moves=valid_moves.copy(),
                        game_result=0  # Will be filled in at game end
                    )
                    data_points.append((data_point, current_player))
                    
                    # Select move based on probabilities (not just best MCTS move)
                    if current_temp > 0:
                        selected_move = self._sample_move(valid_moves, move_probs)
                    else:
                        selected_move = best_move
                    
                    print(f"Game {game_id}, Move {move_count}: Player {current_player} -> {selected_move} "
                          f"({search_stats['simulations']} sims)", file=sys.stderr)
                
                # Make the move
                game_board.make_move(*selected_move, current_player)
                
            except Exception as e:
                print(f"MCTS error in game {game_id}: {e}", file=sys.stderr)
                # Make a fallback pass move to avoid infinite loop
                game_board.make_move(-1, -1, -1, -1, current_player)
                break
            
            # Switch players
            current_player = 1 - current_player
        
        # Check if game ended due to move limit
        if move_count >= max_moves:
            print(f"Game {game_id} ended due to move limit ({max_moves} moves)", file=sys.stderr)
        
        # Determine game result and assign to all data points
        winner = game_board.get_winner()
        final_data_points = []
        
        for data_point, player in data_points:
            if winner == -1:  # Tie
                data_point.game_result = 0
            elif winner == player:  # Player won
                data_point.game_result = 1
            else:  # Player lost
                data_point.game_result = -1
            
            final_data_points.append(data_point)
        
        print(f"Game {game_id} finished: {len(final_data_points)} positions, "
              f"winner = {winner}", file=sys.stderr)
        
        self.games_generated += 1
        self.total_positions += len(final_data_points)
        
        return final_data_points
    
    def _extract_visit_counts(self, mcts, valid_moves: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Extract visit counts for valid moves from MCTS tree."""
        visit_counts = np.zeros(len(valid_moves), dtype=float)
        
        try:
            # Access MCTS tree through the last search stats
            if hasattr(mcts, 'last_search_stats') and 'move_probabilities' in mcts.last_search_stats:
                move_probs = mcts.last_search_stats['move_probabilities']
                for i, move in enumerate(valid_moves):
                    visit_counts[i] = move_probs.get(move, 1.0)
            else:
                # Try to access tree directly if available
                tree = getattr(mcts, 'tree', None)
                if tree and hasattr(tree, 'root'):
                    root = tree.root
                    if hasattr(root, 'children'):
                        for i, move in enumerate(valid_moves):
                            if move in root.children:
                                visit_counts[i] = root.children[move].visits
                            else:
                                visit_counts[i] = 1.0  # Unvisited moves get small count
                    else:
                        visit_counts.fill(1.0)
                else:
                    # Fallback: uniform distribution
                    visit_counts.fill(1.0)
                
        except Exception as e:
            print(f"Error extracting visit counts: {e}", file=sys.stderr)
            visit_counts.fill(1.0)
        
        # Ensure no zero counts (for numerical stability)
        visit_counts = np.maximum(visit_counts, 1.0)
        return visit_counts
    
    def _visit_counts_to_probabilities(self, visit_counts: np.ndarray, 
                                     temperature: float) -> np.ndarray:
        """Convert visit counts to probability distribution using temperature."""
        if temperature == 0:
            # Deterministic: select most visited
            probs = np.zeros_like(visit_counts)
            best_idx = np.argmax(visit_counts)
            probs[best_idx] = 1.0
            return probs
        else:
            # Stochastic: use temperature scaling
            log_probs = np.log(visit_counts + 1e-8) / temperature
            log_probs = log_probs - np.max(log_probs)  # Numerical stability
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)
            return probs
    
    def _sample_move(self, valid_moves: List[Tuple[int, int, int, int]], 
                    probabilities: np.ndarray) -> Tuple[int, int, int, int]:
        """Sample a move according to the probability distribution."""
        try:
            selected_idx = np.random.choice(len(valid_moves), p=probabilities)
            return valid_moves[selected_idx]
        except Exception:
            # Fallback to uniform random
            return random.choice(valid_moves)
    
    def generate_dataset(self, 
                        board_configs: List[List[List[int]]], 
                        num_games_per_config: int = 10,
                        output_path: str = "training_data.pkl") -> Dict[str, Any]:
        """
        Generate a complete training dataset using multiple board configurations.
        
        Args:
            board_configs: List of board configurations to use
            num_games_per_config: Number of self-play games per configuration
            output_path: Path to save the dataset
            
        Returns:
            Dataset statistics
        """
        print(f"Generating dataset with {len(board_configs)} configs, "
              f"{num_games_per_config} games each", file=sys.stderr)
        
        all_data_points = []
        start_time = time.time()
        
        # Generate games for each board configuration
        for config_idx, board_config in enumerate(board_configs):
            print(f"Processing board config {config_idx + 1}/{len(board_configs)}", 
                  file=sys.stderr)
            
            # For now, generate games sequentially to avoid multiprocessing issues
            # TODO: Fix multiprocessing with proper pickling support
            for game_idx in range(num_games_per_config):
                try:
                    game_id = config_idx * num_games_per_config + game_idx
                    game_data = self.generate_self_play_game(board_config, game_id)
                    all_data_points.extend(game_data)
                except Exception as e:
                    print(f"Game generation failed: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
        
        # Save dataset
        dataset = {
            'data_points': all_data_points,
            'generation_time': time.time() - start_time,
            'num_games': self.games_generated,
            'num_positions': len(all_data_points),
            'config': {
                'mcts_simulations': self.mcts_simulations,
                'mcts_time_limit': self.mcts_time_limit,
                'temperature': self.temperature,
                'temperature_decay_moves': self.temperature_decay_moves
            }
        }
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                    exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Statistics
        stats = {
            'total_games': self.games_generated,
            'total_positions': len(all_data_points),
            'avg_positions_per_game': len(all_data_points) / max(self.games_generated, 1),
            'generation_time': dataset['generation_time'],
            'positions_per_second': len(all_data_points) / dataset['generation_time']
        }
        
        print(f"Dataset generated: {stats}", file=sys.stderr)
        return stats


def load_board_configurations() -> List[List[List[int]]]:
    """Load board configurations from file."""
    # For now, create some sample configurations
    # In practice, you'd load from actual game configurations
    configs = []
    
    # Load default configuration from practice/testing/input.txt
    try:
        input_path = "/Users/brillight/Documents/CodingProject/NYPC/practice/testing/input.txt"
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        board = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.strip()]
                board.append(row)
        
        if len(board) == 10 and len(board[0]) == 17:
            configs.append(board)
            print(f"Loaded board configuration from {input_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Could not load board config: {e}", file=sys.stderr)
        
        # Fallback: generate random board
        board = [[random.randint(0, 9) for _ in range(17)] for _ in range(10)]
        configs.append(board)
    
    return configs


def main():
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MCTS training data')
    parser.add_argument('--output', type=str, default='training_data.pkl',
                       help='Output file path')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games to generate')
    parser.add_argument('--simulations', type=int, default=400,
                       help='MCTS simulations per move')
    parser.add_argument('--time-limit', type=float, default=2.0,
                       help='MCTS time limit per move (seconds)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MCTSDataGenerator(
        mcts_simulations=args.simulations,
        mcts_time_limit=args.time_limit
    )
    
    # Load board configurations
    board_configs = load_board_configurations()
    
    # Generate dataset
    stats = generator.generate_dataset(
        board_configs=board_configs,
        num_games_per_config=args.games,
        output_path=args.output
    )
    
    print("Data generation complete!")
    print(f"Generated {stats['total_positions']} training positions from {stats['total_games']} games")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()