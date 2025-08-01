#!/usr/bin/env python3
"""
Board Generator for NYPC Mushroom Game

Generates diverse board configurations for training data collection.
Ensures boards are playable and have reasonable game length.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import json
import os

# Board dimensions
BOARD_HEIGHT = 10
BOARD_WIDTH = 17

class BoardGenerator:
    """
    Generates diverse board configurations for the mushroom game.
    
    Creates boards with various difficulty levels and patterns to ensure
    training data diversity and prevent overfitting.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize board generator.
        
        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generation parameters
        self.min_nonzero_ratio = 0.3  # Minimum ratio of non-zero cells
        self.max_nonzero_ratio = 0.8  # Maximum ratio of non-zero cells
        self.value_range = (1, 9)     # Range of cell values (excluding 0)
        
        print(f"BoardGenerator initialized with seed: {seed}")
    
    def generate_random_board(self) -> List[List[int]]:
        """
        Generate a completely random board.
        
        Returns:
            10x17 board as list of lists
        """
        board = []
        
        # Decide how many cells should be non-zero
        total_cells = BOARD_HEIGHT * BOARD_WIDTH
        nonzero_ratio = random.uniform(self.min_nonzero_ratio, self.max_nonzero_ratio)
        nonzero_count = int(total_cells * nonzero_ratio)
        
        # Create flat array with desired number of non-zero values
        flat_board = [0] * total_cells
        
        # Fill with random values
        for i in range(nonzero_count):
            flat_board[i] = random.randint(*self.value_range)
        
        # Shuffle to randomize positions
        random.shuffle(flat_board)
        
        # Convert to 2D board
        for r in range(BOARD_HEIGHT):
            row = []
            for c in range(BOARD_WIDTH):
                idx = r * BOARD_WIDTH + c
                row.append(flat_board[idx])
            board.append(row)
        
        return board
    
    def generate_clustered_board(self) -> List[List[int]]:
        """
        Generate a board with clustered patterns.
        
        Returns:
            10x17 board with values clustered in regions
        """
        board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        
        # Create 3-5 clusters
        num_clusters = random.randint(3, 5)
        
        for _ in range(num_clusters):
            # Random cluster center
            center_r = random.randint(1, BOARD_HEIGHT - 2)
            center_c = random.randint(1, BOARD_WIDTH - 2)
            
            # Random cluster size
            cluster_size = random.randint(8, 20)
            
            # Random cluster value tendency
            base_value = random.randint(*self.value_range)
            
            for _ in range(cluster_size):
                # Place values around cluster center
                offset_r = random.randint(-2, 2)
                offset_c = random.randint(-2, 2)
                
                r = max(0, min(BOARD_HEIGHT - 1, center_r + offset_r))
                c = max(0, min(BOARD_WIDTH - 1, center_c + offset_c))
                
                if board[r][c] == 0:  # Don't overwrite existing values
                    # Values similar to base_value with some variation
                    value = max(1, min(9, base_value + random.randint(-2, 2)))
                    board[r][c] = value
        
        return board
    
    def generate_structured_board(self) -> List[List[int]]:
        """
        Generate a board with some structural patterns.
        
        Returns:
            10x17 board with structural patterns
        """
        board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        
        # Add some horizontal lines
        for _ in range(random.randint(1, 3)):
            row = random.randint(0, BOARD_HEIGHT - 1)
            start_col = random.randint(0, BOARD_WIDTH - 5)
            length = random.randint(3, min(8, BOARD_WIDTH - start_col))
            value = random.randint(*self.value_range)
            
            for c in range(start_col, start_col + length):
                if board[row][c] == 0:
                    board[row][c] = value
        
        # Add some vertical lines
        for _ in range(random.randint(1, 3)):
            col = random.randint(0, BOARD_WIDTH - 1)
            start_row = random.randint(0, BOARD_HEIGHT - 3)
            length = random.randint(2, min(5, BOARD_HEIGHT - start_row))
            value = random.randint(*self.value_range)
            
            for r in range(start_row, start_row + length):
                if board[r][c] == 0:
                    board[r][c] = value
        
        # Fill remaining spaces randomly
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if board[r][c] == 0 and random.random() < 0.3:
                    board[r][c] = random.randint(*self.value_range)
        
        return board
    
    def generate_sum_constrained_board(self) -> List[List[int]]:
        """
        Generate a board with many rectangles that sum to 10.
        
        Returns:
            10x17 board optimized for valid moves
        """
        board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        
        # Pre-place some rectangles that sum to 10
        attempts = 20
        placed_rectangles = []
        
        for _ in range(attempts):
            # Random rectangle size
            height = random.randint(1, 3)
            width = random.randint(2, 4) if height == 1 else random.randint(1, 3)
            
            # Random position
            r1 = random.randint(0, BOARD_HEIGHT - height)
            c1 = random.randint(0, BOARD_WIDTH - width)
            r2 = r1 + height - 1
            c2 = c1 + width - 1
            
            # Check for overlap
            overlap = False
            for pr1, pc1, pr2, pc2 in placed_rectangles:
                if not (r2 < pr1 or r1 > pr2 or c2 < pc1 or c1 > pc2):
                    overlap = True
                    break
            
            if overlap:
                continue
            
            # Generate values that sum to 10
            area = height * width
            target_sum = 10
            
            # Create values
            values = []
            remaining_sum = target_sum
            for i in range(area - 1):
                max_val = min(9, remaining_sum - (area - i - 1))
                min_val = max(1, remaining_sum - 9 * (area - i - 1))
                if min_val <= max_val:
                    val = random.randint(min_val, max_val)
                    values.append(val)
                    remaining_sum -= val
            
            if remaining_sum >= 1 and remaining_sum <= 9:
                values.append(remaining_sum)
                
                # Place the rectangle
                idx = 0
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        board[r][c] = values[idx]
                        idx += 1
                
                placed_rectangles.append((r1, c1, r2, c2))
        
        # Fill remaining spaces with random values
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if board[r][c] == 0 and random.random() < 0.2:
                    board[r][c] = random.randint(*self.value_range)
        
        return board
    
    def generate_board(self, board_type: str = "random") -> List[List[int]]:
        """
        Generate a board of specified type.
        
        Args:
            board_type: Type of board to generate
                       ("random", "clustered", "structured", "sum_constrained", "mixed")
                       
        Returns:
            10x17 board as list of lists
        """
        if board_type == "random":
            return self.generate_random_board()
        elif board_type == "clustered":
            return self.generate_clustered_board()
        elif board_type == "structured":
            return self.generate_structured_board()
        elif board_type == "sum_constrained":
            return self.generate_sum_constrained_board()
        elif board_type == "mixed":
            # Randomly choose a type
            types = ["random", "clustered", "structured", "sum_constrained"]
            chosen_type = random.choice(types)
            return self.generate_board(chosen_type)
        else:
            raise ValueError(f"Unknown board type: {board_type}")
    
    def generate_board_set(self, num_boards: int, board_types: List[str] = None) -> List[List[List[int]]]:
        """
        Generate a set of diverse boards.
        
        Args:
            num_boards: Number of boards to generate
            board_types: List of board types to use (None for mixed)
            
        Returns:
            List of boards
        """
        if board_types is None:
            board_types = ["mixed"]
        
        boards = []
        for i in range(num_boards):
            board_type = board_types[i % len(board_types)]
            board = self.generate_board(board_type)
            boards.append(board)
        
        print(f"Generated {num_boards} boards with types: {board_types}")
        return boards
    
    def validate_board(self, board: List[List[int]]) -> Dict[str, bool]:
        """
        Validate a board configuration.
        
        Args:
            board: Board to validate
            
        Returns:
            Dictionary with validation results
        """
        if len(board) != BOARD_HEIGHT or len(board[0]) != BOARD_WIDTH:
            return {"valid": False, "reason": "Invalid dimensions"}
        
        # Check value ranges
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if not (0 <= board[r][c] <= 9):
                    return {"valid": False, "reason": f"Invalid value at ({r},{c}): {board[r][c]}"}
        
        # Check for playability (at least some non-zero values)
        nonzero_count = sum(1 for r in range(BOARD_HEIGHT) for c in range(BOARD_WIDTH) if board[r][c] != 0)
        if nonzero_count < 10:  # Minimum threshold for playability
            return {"valid": False, "reason": "Too few non-zero values"}
        
        return {"valid": True}
    
    def save_boards(self, boards: List[List[List[int]]], filepath: str):
        """Save boards to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(boards, f, indent=2)
        print(f"Saved {len(boards)} boards to {filepath}")
    
    def load_boards(self, filepath: str) -> List[List[List[int]]]:
        """Load boards from JSON file."""
        with open(filepath, 'r') as f:
            boards = json.load(f)
        print(f"Loaded {len(boards)} boards from {filepath}")
        return boards

# Default board configurations
def get_default_board() -> List[List[int]]:
    """Get the default board from practice/testing/input.txt"""
    try:
        input_path = "practice/testing/input.txt"
        if not os.path.exists(input_path):
            # Try relative to this file
            current_dir = os.path.dirname(__file__)
            input_path = os.path.join(os.path.dirname(current_dir), "testing", "input.txt")
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        board = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.strip()]
                board.append(row)
        
        return board
    except Exception as e:
        print(f"Could not load default board: {e}")
        # Return a simple fallback board
        generator = BoardGenerator(seed=42)
        return generator.generate_random_board()

def test_board_generator():
    """Test the board generator."""
    print("Testing BoardGenerator...")
    
    generator = BoardGenerator(seed=42)
    
    # Test different board types
    board_types = ["random", "clustered", "structured", "sum_constrained"]
    
    for board_type in board_types:
        print(f"\nTesting {board_type} board generation...")
        board = generator.generate_board(board_type)
        
        # Validate
        validation = generator.validate_board(board)
        print(f"Validation: {validation}")
        
        # Basic stats
        nonzero_count = sum(1 for r in range(BOARD_HEIGHT) for c in range(BOARD_WIDTH) if board[r][c] != 0)
        total_sum = sum(board[r][c] for r in range(BOARD_HEIGHT) for c in range(BOARD_WIDTH))
        
        print(f"Non-zero cells: {nonzero_count}/{BOARD_HEIGHT * BOARD_WIDTH}")
        print(f"Total sum: {total_sum}")
    
    # Test board set generation
    print(f"\nTesting board set generation...")
    boards = generator.generate_board_set(5, ["mixed"])
    print(f"Generated {len(boards)} boards")
    
    print("BoardGenerator test completed!")

if __name__ == "__main__":
    test_board_generator()