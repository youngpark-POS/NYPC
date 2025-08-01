#!/usr/bin/env python3
"""
Unified GameBoard class for NYPC Mushroom Game

This module provides a complete implementation of the game board with:
- Game state management
- Move validation and generation
- Progressive expansion optimization
- Neural network feature extraction
- Undo/redo functionality
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class GameBoard:
    """
    Unified game board for NYPC Mushroom Game.
    
    Handles all game logic including:
    - Board state tracking (original values + territory)
    - Move validation according to game rules
    - Efficient move generation with progressive expansion
    - Feature extraction for neural networks
    - Move history and undo functionality
    """
    
    def __init__(self, board_data: List[List[int]]):
        """
        Initialize game board.
        
        Args:
            board_data: 10x17 grid of initial values (0-9)
        """
        self.rows = 10
        self.cols = 17
        self.original_board = np.array(board_data, dtype=int)
        self.board = self.original_board.copy()
        self.territory = np.zeros((self.rows, self.cols), dtype=int)
        self.turn_count = 0
        self.consecutive_passes = 0  # Track consecutive pass moves
        
        # Move history for undo functionality
        self.move_history: List[Tuple[int, int, int, int, int]] = []
        self.board_history: List[np.ndarray] = []
        self.territory_history: List[np.ndarray] = []
    
    def clone(self) -> 'GameBoard':
        """Create a deep copy of the current board state."""
        new_board = GameBoard(self.original_board.tolist())
        new_board.board = self.board.copy()
        new_board.territory = self.territory.copy()
        new_board.turn_count = self.turn_count
        new_board.consecutive_passes = self.consecutive_passes
        
        # Copy history (deep copy)
        new_board.move_history = self.move_history.copy()
        new_board.board_history = [hist.copy() for hist in self.board_history]
        new_board.territory_history = [hist.copy() for hist in self.territory_history]
        
        return new_board
    
    def is_valid_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Check if a move is valid according to game rules.
        
        Rules:
        1. Rectangle must be within bounds
        2. Sum of non-zero values must equal 10
        3. Each edge must have at least one non-zero value
        
        Args:
            r1, c1: Top-left corner of rectangle
            r2, c2: Bottom-right corner of rectangle
            
        Returns:
            True if move is valid, False otherwise
        """
        # Pass move is always valid
        if r1 == -1:
            return True
            
        # Check bounds
        if not (0 <= r1 <= r2 < self.rows and 0 <= c1 <= c2 < self.cols):
            return False
            
        # Check sum and edge requirements
        total = 0
        edges = [False, False, False, False]  # top, bottom, left, right
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] > 0:
                    total += self.board[r][c]
                    # Mark edges that have non-zero values
                    if r == r1: edges[0] = True  # top
                    if r == r2: edges[1] = True  # bottom
                    if c == c1: edges[2] = True  # left
                    if c == c2: edges[3] = True  # right
        
        return total == 10 and all(edges)
    
    def make_move(self, r1: int, c1: int, r2: int, c2: int, player: int) -> bool:
        """
        Make a move on the board.
        
        Args:
            r1, c1: Top-left corner
            r2, c2: Bottom-right corner
            player: Player ID (0 or 1)
            
        Returns:
            True if move was made successfully, False if invalid
        """
        if not self.is_valid_move(r1, c1, r2, c2):
            return False
            
        # Save current state for undo
        self.move_history.append((r1, c1, r2, c2, player))
        self.board_history.append(self.board.copy())
        self.territory_history.append(self.territory.copy())
        
        # Handle pass move
        if r1 == -1:
            self.consecutive_passes += 1
            self.turn_count += 1
            return True
            
        # Make the move
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
                self.territory[r][c] = 1 if player == 0 else -1
        
        # Reset consecutive passes counter when making a real move
        self.consecutive_passes = 0
        self.turn_count += 1
        return True
    
    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if undo was successful, False if no moves to undo
        """
        if not self.move_history:
            return False
            
        # Restore previous state
        self.board = self.board_history.pop()
        self.territory = self.territory_history.pop()
        self.move_history.pop()
        self.turn_count -= 1
        
        return True
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all valid moves using progressive expansion optimization.
        
        Returns:
            List of valid moves as (r1, c1, r2, c2) tuples
        """
        moves = [(-1, -1, -1, -1)]  # Pass move always available
        
        # Progressive expansion: start from each top-left corner and expand
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(r1, self.rows):
                    sum_val = 0
                    r1fit = False
                    r2fit = False
                    c1fit = False
                    
                    for c2 in range(c1, self.cols):
                        col_has_nonzero = False
                        c2fit = False
                        
                        # Add new column to rectangle
                        for r in range(r1, r2 + 1):
                            if self.board[r][c2] > 0:
                                sum_val += self.board[r][c2]
                                col_has_nonzero = True
                                if r == r1: r1fit = True
                                if r == r2: r2fit = True
                                if c2 == c1: c1fit = True
                        
                        if col_has_nonzero: c2fit = True
                        
                        # Early termination if sum exceeds 10
                        if sum_val > 10:
                            break
                            
                        # Check if we have a valid move
                        if sum_val == 10 and r1fit and r2fit and c1fit and c2fit:
                            moves.append((r1, c1, r2, c2))
        
        return moves
    
    def get_score(self) -> Tuple[int, int]:
        """
        Calculate current scores for both players.
        
        Returns:
            Tuple of (player1_score, player2_score)
        """
        player1_score = np.sum(self.territory == 1)
        player2_score = np.sum(self.territory == -1)
        return player1_score, player2_score
    
    def is_game_over(self) -> bool:
        """Check if the game is over (both players passed consecutively or no valid moves)."""
        # Game is over if both players passed consecutively
        if self.consecutive_passes >= 2:
            return True
        
        # Game is also over if only pass move is available
        moves = self.get_valid_moves()
        return len(moves) == 1 and moves[0] == (-1, -1, -1, -1)
    
    def get_winner(self) -> int:
        """
        Determine the winner.
        
        Returns:
            0 if player 1 wins, 1 if player 2 wins, -1 for tie
        """
        p1_score, p2_score = self.get_score()
        if p1_score > p2_score:
            return 0
        elif p2_score > p1_score:
            return 1
        else:
            return -1
    
    def to_features(self) -> np.ndarray:
        """
        Convert board state to neural network feature vector.
        
        Returns:
            683-dimensional feature vector:
            - 680 dimensions: board state (10x17x4)
            - 3 dimensions: meta information
        """
        features = []
        
        # Board state features (10x17x4 = 680 dimensions)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r][c]
                territory = self.territory[r][c]
                
                if cell > 0:
                    # Original cell value (normalized)
                    features.extend([cell / 10.0, 0.0, 0.0, 0.0])
                elif territory == 1:  # Player 1 territory
                    features.extend([0.0, 1.0, 1.0, 0.0])
                elif territory == -1:  # Player 2 territory
                    features.extend([0.0, 1.0, 0.0, 1.0])
                else:
                    # Empty cell
                    features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Meta information (3 dimensions)
        scores = self.get_score()
        max_possible_score = self.rows * self.cols  # 170
        features.append(scores[0] / max_possible_score)  # Player 1 score ratio
        features.append(scores[1] / max_possible_score)  # Player 2 score ratio
        features.append(self.turn_count / 100.0)  # Turn count ratio
        
        return np.array(features, dtype=np.float32)
    
    def to_neural_features(self, current_player: int) -> np.ndarray:
        """
        Convert board state to multi-channel input for neural network.
        
        Args:
            current_player: Player who is about to move (0 or 1)
        
        Returns:
            2-channel spatial features (10x17x2):
            - Channel 0: Normalized mushroom values (0→0.0, 1-9→0.1-0.9)
            - Channel 1: Territory difference (my_territory - opponent_territory)
                        +1.0: My territory
                        -1.0: Opponent territory  
                         0.0: Neutral territory
        """
        features = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        
        # Channel 0: Normalized mushroom values (0→0.0, 1-9→0.1-0.9)
        features[:, :, 0] = np.where(self.board == 0, 0.0, 0.1 + (self.board - 1) * 0.8 / 8)
        
        # Channel 1: Territory difference from current player's perspective
        # Player 0: territory==1 is mine, territory==-1 is opponent's
        # Player 1: territory==-1 is mine, territory==1 is opponent's
        if current_player == 0:
            # Player 0's perspective: +1 for player 1 territory, -1 for player 2 territory
            features[:, :, 1] = self.territory.astype(np.float32)
        else:
            # Player 1's perspective: +1 for myself (-1 territory), -1 for opponent (1 territory)
            features[:, :, 1] = -self.territory.astype(np.float32)
        
        return features
    
    def _compute_influence_map(self) -> np.ndarray:
        """
        Compute territory influence map based on distance to controlled areas.
        
        Returns:
            2D array where values represent influence strength (0.0-1.0)
        """
        influence = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Find all controlled territories
        p1_territories = np.where(self.territory == 1)
        p2_territories = np.where(self.territory == -1)
        
        for r in range(self.rows):
            for c in range(self.cols):
                min_dist_p1 = float('inf')
                min_dist_p2 = float('inf')
                
                # Distance to nearest Player 1 territory
                if len(p1_territories[0]) > 0:
                    for p1_r, p1_c in zip(p1_territories[0], p1_territories[1]):
                        dist = abs(r - p1_r) + abs(c - p1_c)  # Manhattan distance
                        min_dist_p1 = min(min_dist_p1, dist)
                
                # Distance to nearest Player 2 territory
                if len(p2_territories[0]) > 0:
                    for p2_r, p2_c in zip(p2_territories[0], p2_territories[1]):
                        dist = abs(r - p2_r) + abs(c - p2_c)  # Manhattan distance
                        min_dist_p2 = min(min_dist_p2, dist)
                
                # Compute influence (-1.0 to 1.0, positive for P1, negative for P2)
                if min_dist_p1 == float('inf') and min_dist_p2 == float('inf'):
                    influence[r, c] = 0.0  # No influence
                elif min_dist_p1 == float('inf'):
                    influence[r, c] = -1.0 / (1.0 + min_dist_p2)  # P2 influence
                elif min_dist_p2 == float('inf'):
                    influence[r, c] = 1.0 / (1.0 + min_dist_p1)   # P1 influence
                else:
                    # Competing influence
                    p1_influence = 1.0 / (1.0 + min_dist_p1)
                    p2_influence = 1.0 / (1.0 + min_dist_p2)
                    influence[r, c] = p1_influence - p2_influence
        
        # Normalize to [0, 1] range
        if np.max(np.abs(influence)) > 0:
            influence = (influence + 1.0) / 2.0
        
        return influence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert board state to dictionary for serialization."""
        return {
            'original_board': self.original_board.tolist(),
            'board': self.board.tolist(),
            'territory': self.territory.tolist(),
            'turn_count': self.turn_count,
            'move_history': self.move_history,
            'scores': self.get_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameBoard':
        """Create GameBoard from dictionary."""
        board = cls(data['original_board'])
        board.board = np.array(data['board'])
        board.territory = np.array(data['territory'])
        board.turn_count = data['turn_count']
        board.move_history = data.get('move_history', [])
        return board
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = "Original Board:\n"
        for row in self.original_board:
            result += " ".join(str(x) for x in row) + "\n"
        
        result += "\nCurrent Board:\n"
        for row in self.board:
            result += " ".join(str(x) for x in row) + "\n"
        
        result += "\nTerritory:\n"
        for row in self.territory:
            result += " ".join(f"{x:2d}" for x in row) + "\n"
        
        scores = self.get_score()
        result += f"\nScores: P1={scores[0]}, P2={scores[1]}, Turn={self.turn_count}"
        
        return result