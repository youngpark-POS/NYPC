#!/usr/bin/env python3
"""
Game rules and validation utilities for NYPC Mushroom Game

This module defines game constants, validation functions, and utility
methods that are used throughout the system.
"""

from typing import List, Tuple, Optional
import numpy as np

# Game constants
BOARD_ROWS = 10
BOARD_COLS = 17
TARGET_SUM = 10
MAX_CELL_VALUE = 9
PASS_MOVE = (-1, -1, -1, -1)

# Player constants
PLAYER_1 = 0
PLAYER_2 = 1
NO_PLAYER = -1

class GameRules:
    """Static class containing game rule validation and utilities."""
    
    @staticmethod
    def is_valid_coordinates(r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if coordinates form a valid rectangle within board bounds."""
        if r1 == -1:  # Pass move
            return True
            
        return (0 <= r1 <= r2 < BOARD_ROWS and 
                0 <= c1 <= c2 < BOARD_COLS)
    
    @staticmethod
    def is_pass_move(move: Tuple[int, int, int, int]) -> bool:
        """Check if a move is a pass move."""
        return move == PASS_MOVE or move[0] == -1
    
    @staticmethod
    def calculate_rectangle_sum(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculate sum of values in a rectangle."""
        if r1 == -1:  # Pass move
            return 0
            
        total = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if board[r][c] > 0:
                    total += board[r][c]
        return total
    
    @staticmethod
    def check_edge_requirements(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if each edge of the rectangle has at least one non-zero value."""
        if r1 == -1:  # Pass move
            return True
            
        edges = [False, False, False, False]  # top, bottom, left, right
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if board[r][c] > 0:
                    if r == r1: edges[0] = True  # top
                    if r == r2: edges[1] = True  # bottom
                    if c == c1: edges[2] = True  # left
                    if c == c2: edges[3] = True  # right
        
        return all(edges)
    
    @staticmethod
    def validate_move(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Validate a move according to all game rules.
        
        Rules:
        1. Coordinates must be valid
        2. Sum of non-zero values must equal TARGET_SUM
        3. Each edge must have at least one non-zero value
        """
        # Check coordinates
        if not GameRules.is_valid_coordinates(r1, c1, r2, c2):
            return False
            
        # Pass move is always valid
        if GameRules.is_pass_move((r1, c1, r2, c2)):
            return True
            
        # Check sum requirement
        if GameRules.calculate_rectangle_sum(board, r1, c1, r2, c2) != TARGET_SUM:
            return False
            
        # Check edge requirements
        if not GameRules.check_edge_requirements(board, r1, c1, r2, c2):
            return False
            
        return True
    
    @staticmethod
    def get_rectangle_cells(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
        """Get list of all cells in a rectangle."""
        if r1 == -1:  # Pass move
            return []
            
        cells = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cells.append((r, c))
        return cells
    
    @staticmethod
    def move_to_string(move: Tuple[int, int, int, int]) -> str:
        """Convert move to string representation."""
        if GameRules.is_pass_move(move):
            return "PASS"
        return f"({move[0]},{move[1]})-({move[2]},{move[3]})"
    
    @staticmethod
    def string_to_move(move_str: str) -> Tuple[int, int, int, int]:
        """Parse move from string representation."""
        if move_str.upper() == "PASS":
            return PASS_MOVE
            
        # Parse format like "(0,1)-(2,3)"
        parts = move_str.replace("(", "").replace(")", "").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid move string: {move_str}")
            
        start_coords = parts[0].split(",")
        end_coords = parts[1].split(",")
        
        if len(start_coords) != 2 or len(end_coords) != 2:
            raise ValueError(f"Invalid move string: {move_str}")
            
        r1, c1 = int(start_coords[0]), int(start_coords[1])
        r2, c2 = int(end_coords[0]), int(end_coords[1])
        
        return (r1, c1, r2, c2)

class MoveGenerator:
    """Utility class for generating moves efficiently."""
    
    @staticmethod
    def generate_all_moves(board: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Generate all possible moves (naive approach)."""
        moves = [PASS_MOVE]
        
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                for r2 in range(r1, BOARD_ROWS):
                    for c2 in range(c1, BOARD_COLS):
                        if GameRules.validate_move(board, r1, c1, r2, c2):
                            moves.append((r1, c1, r2, c2))
        
        return moves
    
    @staticmethod
    def generate_moves_progressive(board: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Generate moves using progressive expansion (optimized)."""
        moves = [PASS_MOVE]
        
        for r1 in range(BOARD_ROWS):
            for c1 in range(BOARD_COLS):
                for r2 in range(r1, BOARD_ROWS):
                    sum_val = 0
                    r1fit = False
                    r2fit = False
                    c1fit = False
                    
                    for c2 in range(c1, BOARD_COLS):
                        col_has_nonzero = False
                        c2fit = False
                        
                        # Add new column to rectangle
                        for r in range(r1, r2 + 1):
                            if board[r][c2] > 0:
                                sum_val += board[r][c2]
                                col_has_nonzero = True
                                if r == r1: r1fit = True
                                if r == r2: r2fit = True
                                if c2 == c1: c1fit = True
                        
                        if col_has_nonzero: c2fit = True
                        
                        # Early termination if sum exceeds target
                        if sum_val > TARGET_SUM:
                            break
                            
                        # Check if we have a valid move
                        if sum_val == TARGET_SUM and r1fit and r2fit and c1fit and c2fit:
                            moves.append((r1, c1, r2, c2))
        
        return moves

class GameState:
    """Utility class for game state analysis."""
    
    @staticmethod
    def calculate_scores(territory: np.ndarray) -> Tuple[int, int]:
        """Calculate scores from territory array."""
        player1_score = np.sum(territory == 1)
        player2_score = np.sum(territory == -1)
        return player1_score, player2_score
    
    @staticmethod
    def get_winner(territory: np.ndarray) -> int:
        """Determine winner from territory array."""
        p1_score, p2_score = GameState.calculate_scores(territory)
        if p1_score > p2_score:
            return PLAYER_1
        elif p2_score > p1_score:
            return PLAYER_2
        else:
            return NO_PLAYER  # Tie
    
    @staticmethod
    def is_terminal_state(board: np.ndarray) -> bool:
        """Check if board is in terminal state (no valid moves except pass)."""
        moves = MoveGenerator.generate_moves_progressive(board)
        return len(moves) == 1  # Only pass move available
    
    @staticmethod
    def count_available_cells(board: np.ndarray) -> int:
        """Count cells with non-zero values."""
        return np.sum(board > 0)
    
    @staticmethod
    def get_board_density(board: np.ndarray) -> float:
        """Get fraction of non-zero cells."""
        total_cells = BOARD_ROWS * BOARD_COLS
        non_zero_cells = GameState.count_available_cells(board)
        return non_zero_cells / total_cells

# Utility functions for common operations
def encode_move_for_neural_net(move: Tuple[int, int, int, int]) -> int:
    """Encode move as integer for neural network."""
    if GameRules.is_pass_move(move):
        return 0
    r1, c1, r2, c2 = move
    return r1 * 1000 + c1 * 100 + r2 * 10 + c2 + 1

def decode_move_from_neural_net(encoded: int) -> Tuple[int, int, int, int]:
    """Decode integer back to move tuple."""
    if encoded == 0:
        return PASS_MOVE
    
    encoded -= 1  # Adjust for +1 offset
    c2 = encoded % 10
    encoded //= 10
    r2 = encoded % 10
    encoded //= 10
    c1 = encoded % 100
    r1 = encoded // 100
    
    return (r1, c1, r2, c2)

def validate_board_data(board_data: List[List[int]]) -> bool:
    """Validate that board data is correctly formatted."""
    if len(board_data) != BOARD_ROWS:
        return False
        
    for row in board_data:
        if len(row) != BOARD_COLS:
            return False
        for cell in row:
            if not (0 <= cell <= MAX_CELL_VALUE):
                return False
                
    return True