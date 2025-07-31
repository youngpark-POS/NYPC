#!/usr/bin/env python3
"""
MCTS Tree implementation for NYPC Mushroom Game

This module provides the tree structure and nodes used in
Monte Carlo Tree Search.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.
    
    Attributes:
        state: Game board state (will be copied)
        parent: Parent node (None for root)
        move: Move that led to this state
        player: Player who made the move to reach this state
        children: Dictionary of child nodes {move: node}
        visits: Number of times this node has been visited
        value_sum: Sum of all values from simulations through this node
        is_terminal: Whether this node represents a terminal game state
        is_expanded: Whether this node has been expanded (children generated)
        policy_priors: Prior probabilities from policy network
    """
    state: Any  # GameBoard object
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int, int, int]] = None
    player: int = 0
    children: Dict[Tuple[int, int, int, int], 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False
    is_expanded: bool = False
    policy_priors: Dict[Tuple[int, int, int, int], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize node after creation."""
        # Create a deep copy of the state to avoid modification issues
        if self.state is not None:
            self.state = self.state.clone()
            self.is_terminal = self.state.is_game_over()
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def unexplored_moves(self) -> List[Tuple[int, int, int, int]]:
        """Get list of moves that haven't been explored yet."""
        if not self.is_expanded:
            return []
        
        all_moves = self.state.get_valid_moves()
        explored_moves = set(self.children.keys())
        return [move for move in all_moves if move not in explored_moves]
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this node have been explored."""
        if self.is_terminal:
            return True
        
        if not self.is_expanded:
            return False
            
        all_moves = self.state.get_valid_moves()
        return len(self.children) == len(all_moves)
    
    def select_child_ucb1(self, exploration_weight: float = math.sqrt(2), use_priors: bool = False) -> 'MCTSNode':
        """
        Select child using UCB1 formula with optional policy priors.
        
        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a)) + prior_bonus
        
        where prior_bonus = policy_weight * P(s,a) if use_priors is True
        
        Args:
            exploration_weight: UCB1 exploration parameter
            use_priors: Whether to use policy network priors
            
        Returns:
            Selected child node
        """
        if not self.children:
            raise ValueError("No children to select from")
        
        best_child = None
        best_value = float('-inf')
        
        for move, child in self.children.items():
            if child.visits == 0:
                # Unvisited children - apply priors if available
                if use_priors and move in self.policy_priors:
                    prior_bonus = self.policy_priors[move] * 2.0  # Scale factor for priors
                    ucb1_value = float('inf') + prior_bonus  # High base value + prior bonus
                else:
                    ucb1_value = float('inf')
                    
                if ucb1_value > best_value:
                    best_value = ucb1_value
                    best_child = child
                    
                # If we found an unvisited node with high prior, select it
                if use_priors and move in self.policy_priors and self.policy_priors[move] > 0.1:
                    return child
                    
            else:
                # Standard UCB1 calculation
                exploitation = child.value
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                
                # Add prior bonus if available
                prior_bonus = 0.0
                if use_priors and move in self.policy_priors:
                    # Decrease prior influence as visits increase
                    prior_weight = 1.0 / (1.0 + child.visits * 0.1)
                    prior_bonus = prior_weight * self.policy_priors[move]
                
                ucb1_value = exploitation + exploration + prior_bonus
                
                if ucb1_value > best_value:
                    best_value = ucb1_value
                    best_child = child
        
        return best_child
    
    def add_child(self, move: Tuple[int, int, int, int], next_player: int) -> 'MCTSNode':
        """
        Add a child node for the given move.
        
        Args:
            move: The move to make
            next_player: The player who will play after this move
            
        Returns:
            The newly created child node
        """
        if move in self.children:
            return self.children[move]
        
        # Create new state by making the move
        new_state = self.state.clone()
        success = new_state.make_move(*move, self.player)
        
        if not success:
            raise ValueError(f"Invalid move: {move}")
        
        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=self,
            move=move,
            player=next_player
        )
        
        self.children[move] = child
        return child
    
    def expand(self, policy_priors: Optional[Dict[Tuple[int, int, int, int], float]] = None) -> List['MCTSNode']:
        """
        Expand this node by adding all possible child nodes.
        
        Args:
            policy_priors: Prior probabilities from policy network
            
        Returns:
            List of newly created child nodes
        """
        if self.is_expanded or self.is_terminal:
            return list(self.children.values())
        
        valid_moves = self.state.get_valid_moves()
        next_player = 1 - self.player  # Alternate players
        
        # Store policy priors for later use
        if policy_priors is not None:
            self.policy_priors = policy_priors.copy()
        
        new_children = []
        for move in valid_moves:
            child = self.add_child(move, next_player)
            new_children.append(child)
        
        self.is_expanded = True
        return new_children
    
    def backpropagate(self, value: float) -> None:
        """
        Backpropagate a value up the tree.
        
        Args:
            value: The value to backpropagate (from current player's perspective)
        """
        self.visits += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Flip value for opponent
            self.parent.backpropagate(-value)
    
    def get_best_child(self, exploration_weight: float = 0.0, use_priors: bool = False) -> Optional['MCTSNode']:
        """
        Get the best child node.
        
        Args:
            exploration_weight: UCB1 exploration weight (0 for pure exploitation)
            use_priors: Whether to use policy priors in selection
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        if exploration_weight == 0.0:
            # Pure exploitation - select child with highest value
            return max(self.children.values(), key=lambda child: child.value)
        else:
            # Use UCB1 with optional priors
            return self.select_child_ucb1(exploration_weight, use_priors)
    
    def get_visit_counts(self) -> Dict[Tuple[int, int, int, int], int]:
        """Get visit counts for all children."""
        return {move: child.visits for move, child in self.children.items()}
    
    def get_move_probabilities(self, temperature: float = 1.0) -> Dict[Tuple[int, int, int, int], float]:
        """
        Get move probabilities based on visit counts.
        
        Args:
            temperature: Temperature parameter (higher = more exploration)
            
        Returns:
            Dictionary of move probabilities
        """
        if not self.children:
            return {}
        
        visit_counts = [child.visits for child in self.children.values()]
        moves = list(self.children.keys())
        
        if temperature == 0.0:
            # Deterministic selection
            best_count = max(visit_counts)
            probs = [1.0 if count == best_count else 0.0 for count in visit_counts]
            # Normalize in case of ties
            total = sum(probs)
            probs = [p / total for p in probs]
        else:
            # Temperature scaling
            scaled_counts = [count ** (1.0 / temperature) for count in visit_counts]
            total = sum(scaled_counts)
            probs = [count / total for count in scaled_counts]
        
        return dict(zip(moves, probs))
    
    def print_tree(self, depth: int = 0, max_depth: int = 3) -> None:
        """Print tree structure for debugging."""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        move_str = f"Move: {self.move}" if self.move else "Root"
        print(f"{indent}{move_str} | Visits: {self.visits} | Value: {self.value:.3f} | Children: {len(self.children)}")
        
        # Print top children
        if depth < max_depth:
            sorted_children = sorted(self.children.values(), key=lambda x: x.visits, reverse=True)
            for child in sorted_children[:3]:  # Top 3 children
                child.print_tree(depth + 1, max_depth)

class MCTSTree:
    """
    MCTS Tree manager.
    
    Provides high-level interface for tree operations:
    - Tree initialization
    - Node selection
    - Tree statistics
    - Memory management
    """
    
    def __init__(self, initial_state: Any, initial_player: int = 0):
        """
        Initialize MCTS tree.
        
        Args:
            initial_state: Initial game board state
            initial_player: Player to move first
        """
        self.root = MCTSNode(state=initial_state, player=initial_player)
        self.nodes_created = 1
        self.max_nodes = 10000  # Memory limit
    
    def select_leaf(self, exploration_weight: float = math.sqrt(2), use_priors: bool = False) -> MCTSNode:
        """
        Select a leaf node using tree policy (UCB1 with optional priors).
        
        Args:
            exploration_weight: UCB1 exploration parameter
            use_priors: Whether to use policy priors in selection
            
        Returns:
            Leaf node for expansion/simulation
        """
        current = self.root
        
        while not current.is_terminal and current.is_fully_expanded():
            current = current.select_child_ucb1(exploration_weight, use_priors)
        
        return current
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        def count_nodes(node: MCTSNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(self.root)
        
        return {
            'total_nodes': total_nodes,
            'root_visits': self.root.visits,
            'root_children': len(self.root.children),
            'tree_depth': self._calculate_max_depth(),
            'nodes_created': self.nodes_created
        }
    
    def _calculate_max_depth(self, node: Optional[MCTSNode] = None) -> int:
        """Calculate maximum depth of the tree."""
        if node is None:
            node = self.root
        
        if not node.children:
            return 0
        
        return 1 + max(self._calculate_max_depth(child) for child in node.children.values())
    
    def reset(self, new_state: Any, new_player: int = 0) -> None:
        """Reset tree with new root state."""
        self.root = MCTSNode(state=new_state, player=new_player)
        self.nodes_created = 1
    
    def advance_root(self, move: Tuple[int, int, int, int]) -> bool:
        """
        Advance root to child corresponding to the given move.
        
        Args:
            move: Move that was played
            
        Returns:
            True if successful, False if move not found
        """
        if move not in self.root.children:
            return False
        
        # Make the child the new root
        new_root = self.root.children[move]
        new_root.parent = None
        self.root = new_root
        
        return True