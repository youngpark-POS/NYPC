#!/usr/bin/env python3
"""
MCTS Search algorithm for NYPC Mushroom Game

This module implements the core Monte Carlo Tree Search algorithm
with neural network integration capabilities.
"""

import math
import time
import random
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any, List
from .mcts_tree import MCTSTree, MCTSNode

class MCTSSearch:
    """
    Monte Carlo Tree Search implementation.
    
    Provides complete MCTS algorithm with:
    - Selection using UCB1
    - Expansion of leaf nodes
    - Simulation (rollout or neural evaluation)
    - Backpropagation of values
    """
    
    def __init__(self,
                 exploration_weight: float = math.sqrt(2),
                 max_simulations: int = 1000,
                 max_time: float = 1.0,
                 evaluation_function: Optional[Callable] = None,
                 simulation_function: Optional[Callable] = None,
                 policy_function: Optional[Callable] = None,
                 use_policy_priors: bool = False):
        """
        Initialize MCTS search.
        
        Args:
            exploration_weight: UCB1 exploration parameter (typically sqrt(2))
            max_simulations: Maximum number of MCTS simulations
            max_time: Maximum search time in seconds
            evaluation_function: Function to evaluate leaf nodes (neural net)
            simulation_function: Function for rollout simulation
            policy_function: Function to get move probabilities from policy network
            use_policy_priors: Whether to use policy network as priors in UCB1
        """
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_time = max_time
        self.evaluation_function = evaluation_function
        self.simulation_function = simulation_function or self._random_simulation
        self.policy_function = policy_function
        self.use_policy_priors = use_policy_priors
        
        # Statistics
        self.simulations_run = 0
        self.total_search_time = 0.0
        self.last_search_stats = {}
    
    def search(self, 
               initial_state: Any, 
               player: int,
               max_simulations: Optional[int] = None,
               max_time: Optional[float] = None) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
        """
        Run MCTS search to find the best move.
        
        Args:
            initial_state: Initial game board state
            player: Current player (0 or 1)
            max_simulations: Override default max simulations
            max_time: Override default max time
            
        Returns:
            Tuple of (best_move, search_statistics)
        """
        start_time = time.time()
        
        # Use provided limits or defaults
        sim_limit = max_simulations or self.max_simulations
        time_limit = max_time or self.max_time
        
        # Initialize tree
        tree = MCTSTree(initial_state, player)
        
        # Run MCTS simulations
        simulations = 0
        while simulations < sim_limit and (time.time() - start_time) < time_limit:
            self._run_simulation(tree)
            simulations += 1
        
        # Get best move
        best_move = self._select_best_move(tree.root)
        
        # Collect statistics
        search_time = time.time() - start_time
        self.total_search_time += search_time
        self.simulations_run += simulations
        
        stats = {
            'simulations': simulations,
            'search_time': search_time,
            'simulations_per_second': simulations / search_time if search_time > 0 else 0,
            'tree_stats': tree.get_statistics(),
            'move_probabilities': tree.root.get_move_probabilities(temperature=1.0),
            'best_move_visits': tree.root.children.get(best_move, MCTSNode(None)).visits if best_move != (-1, -1, -1, -1) else 0
        }
        
        self.last_search_stats = stats
        return best_move, stats
    
    def _run_simulation(self, tree: MCTSTree) -> None:
        """
        Run a single MCTS simulation.
        
        Steps:
        1. Selection: Navigate to leaf using UCB1
        2. Expansion: Add child nodes if not terminal
        3. Simulation: Evaluate position
        4. Backpropagation: Update node values
        """
        # Selection
        leaf = tree.select_leaf(self.exploration_weight)
        
        # Expansion
        if not leaf.is_terminal and not leaf.is_expanded:
            children = leaf.expand()
            tree.nodes_created += len(children)
            
            # Select one child for simulation
            if children:
                leaf = random.choice(children)
        
        # Simulation/Evaluation
        value = self._evaluate_position(leaf)
        
        # Backpropagation
        leaf.backpropagate(value)
    
    def _evaluate_position(self, node: MCTSNode) -> float:
        """
        Evaluate a position using neural network or simulation.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Value from current player's perspective (-1 to 1)
        """
        # If terminal, return exact value
        if node.is_terminal:
            winner = node.state.get_winner()
            if winner == node.player:
                return 1.0
            elif winner == -1:  # Tie
                return 0.0
            else:
                return -1.0
        
        # Use neural network evaluation if available
        if self.evaluation_function is not None:
            try:
                # Try different feature formats for compatibility
                if hasattr(node.state, 'to_neural_features'):
                    features = node.state.to_neural_features(node.player)
                else:
                    features = node.state.to_features()
                
                value = self.evaluation_function(features, node.player)
                return float(value)
            except Exception as e:
                print(f"Neural evaluation failed: {e}, falling back to simulation")
        
        # Fall back to simulation
        return self.simulation_function(node)
    
    def _random_simulation(self, node: MCTSNode) -> float:
        """
        Run random simulation (rollout) from the given node.
        
        Args:
            node: Starting node for simulation
            
        Returns:
            Game result from current player's perspective
        """
        # Clone the state to avoid modifying the original
        sim_state = node.state.clone()
        current_player = node.player
        max_moves = 100  # Prevent infinite games
        
        for _ in range(max_moves):
            if sim_state.is_game_over():
                break
            
            # Get valid moves and select randomly
            valid_moves = sim_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Simple strategy: avoid pass moves unless necessary
            non_pass_moves = [move for move in valid_moves if move != (-1, -1, -1, -1)]
            if non_pass_moves:
                move = random.choice(non_pass_moves)
            else:
                move = (-1, -1, -1, -1)  # Pass
            
            # Make the move
            sim_state.make_move(*move, current_player)
            current_player = 1 - current_player
            
            # Additional safety: break if too many consecutive passes
            if hasattr(sim_state, 'consecutive_passes') and sim_state.consecutive_passes >= 2:
                break
        
        # Determine winner
        winner = sim_state.get_winner()
        if winner == node.player:
            return 1.0
        elif winner == -1:  # Tie
            return 0.0
        else:
            return -1.0
    
    def _select_best_move(self, root: MCTSNode) -> Tuple[int, int, int, int]:
        """
        Select the best move from root node.
        
        Args:
            root: Root node of the search tree
            
        Returns:
            Best move to play
        """
        if not root.children:
            # No children means only pass move is available
            return (-1, -1, -1, -1)
        
        # Select child with most visits (robust choice)
        best_child = max(root.children.values(), key=lambda child: child.visits)
        return best_child.move
    
    def get_principal_variation(self, tree: MCTSTree, depth: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Get the principal variation (most visited path) from the tree.
        
        Args:
            tree: MCTS tree
            depth: Maximum depth to explore
            
        Returns:
            List of moves in the principal variation
        """
        pv = []
        current = tree.root
        
        for _ in range(depth):
            if not current.children:
                break
            
            # Select child with most visits
            best_child = max(current.children.values(), key=lambda child: child.visits)
            pv.append(best_child.move)
            current = best_child
        
        return pv
    
    def set_evaluation_function(self, eval_func: Callable) -> None:
        """Set neural network evaluation function."""
        self.evaluation_function = eval_func
    
    def set_simulation_function(self, sim_func: Callable) -> None:
        """Set custom simulation function."""
        self.simulation_function = sim_func
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get overall search statistics."""
        return {
            'total_simulations': self.simulations_run,
            'total_search_time': self.total_search_time,
            'average_simulations_per_second': (
                self.simulations_run / self.total_search_time 
                if self.total_search_time > 0 else 0
            ),
            'last_search': self.last_search_stats
        }

class AdaptiveMCTSSearch(MCTSSearch):
    """
    Adaptive MCTS that adjusts parameters based on game state.
    
    Features:
    - Dynamic exploration weight based on game phase
    - Adaptive time allocation
    - Progressive simulation limits
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_exploration_weight = self.exploration_weight
        self.base_max_simulations = self.max_simulations
    
    def search(self, initial_state: Any, player: int, **kwargs) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
        """Adaptive search with dynamic parameters."""
        # Analyze game state
        turn_count = initial_state.turn_count
        available_moves = len(initial_state.get_valid_moves())
        
        # Adjust exploration based on game phase
        if turn_count < 10:
            # Early game: more exploration
            self.exploration_weight = self.base_exploration_weight * 1.5
        elif turn_count < 25:
            # Mid game: balanced
            self.exploration_weight = self.base_exploration_weight
        else:
            # Late game: more exploitation
            self.exploration_weight = self.base_exploration_weight * 0.7
        
        # Adjust simulation count based on move count
        if available_moves > 50:
            # Many moves: more simulations needed
            self.max_simulations = int(self.base_max_simulations * 1.5)
        elif available_moves < 10:
            # Few moves: can afford more simulations per move
            self.max_simulations = int(self.base_max_simulations * 2.0)
        else:
            self.max_simulations = self.base_max_simulations
        
        return super().search(initial_state, player, **kwargs)

# Factory functions for common configurations
def create_basic_mcts(max_simulations: int = 1000, 
                     max_time: float = 1.0) -> MCTSSearch:
    """Create basic MCTS with random simulations."""
    return MCTSSearch(
        max_simulations=max_simulations,
        max_time=max_time
    )

def create_neural_mcts(evaluation_function: Callable,
                      max_simulations: int = 500,
                      max_time: float = 1.0) -> MCTSSearch:
    """Create MCTS with neural network evaluation."""
    return MCTSSearch(
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=evaluation_function
    )

def create_policy_guided_mcts(policy_function: Callable,
                             evaluation_function: Optional[Callable] = None,
                             max_simulations: int = 500,
                             max_time: float = 1.0,
                             exploration_weight: float = 1.4) -> MCTSSearch:
    """Create MCTS with policy network guidance (priors)."""
    return MCTSSearch(
        exploration_weight=exploration_weight,
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=evaluation_function,
        policy_function=policy_function,
        use_policy_priors=True
    )

def create_hybrid_mcts(policy_function: Callable,
                      evaluation_function: Optional[Callable] = None,
                      max_simulations: int = 400,
                      max_time: float = 1.5,
                      exploration_weight: float = 1.2) -> MCTSSearch:
    """Create hybrid MCTS: UCB1 + Policy Priors (no PUCT)."""
    return MCTSSearch(
        exploration_weight=exploration_weight,
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=evaluation_function,
        policy_function=policy_function,
        use_policy_priors=True
    )

def create_value_guided_mcts(value_function: Callable,
                            max_simulations: int = 200,
                            max_time: float = 1.0,
                            exploration_weight: float = 1.0) -> MCTSSearch:
    """Create MCTS with value network for leaf evaluation (no rollouts)."""
    return MCTSSearch(
        exploration_weight=exploration_weight,
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=value_function,
        use_policy_priors=False
    )

def create_alphazero_mcts(combined_net_function: Callable,
                         max_simulations: int = 400,
                         max_time: float = 1.5,
                         exploration_weight: float = 1.0) -> MCTSSearch:
    """Create AlphaZero-style MCTS with combined policy-value network."""
    
    def policy_prior_function(features, valid_moves):
        """Extract policy priors from combined network."""
        try:
            move_probs, _ = combined_net_function(features, valid_moves)
            return move_probs
        except Exception as e:
            print(f"Policy extraction failed: {e}")
            return {move: 1.0/len(valid_moves) for move in valid_moves}
    
    def value_eval_function(features, player):
        """Extract value evaluation from combined network."""
        try:
            _, win_prob = combined_net_function(features, [])
            return win_prob
        except Exception as e:
            print(f"Value extraction failed: {e}")
            return 0.0
    
    return MCTSSearch(
        exploration_weight=exploration_weight,
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=value_eval_function,
        policy_function=policy_prior_function,
        use_policy_priors=True
    )

def create_adaptive_mcts(evaluation_function: Optional[Callable] = None,
                        max_simulations: int = 1000,
                        max_time: float = 1.0) -> AdaptiveMCTSSearch:
    """Create adaptive MCTS with dynamic parameters."""
    return AdaptiveMCTSSearch(
        max_simulations=max_simulations,
        max_time=max_time,
        evaluation_function=evaluation_function
    )