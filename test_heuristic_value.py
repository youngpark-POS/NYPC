#!/usr/bin/env python3
"""
Simple test script for heuristic value evaluation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from practice.core.game_board import GameBoard
from practice.training.self_play import heuristic_value_evaluation, SelfPlayGenerator

def test_heuristic_value():
    """Test the heuristic value evaluation function."""
    print("Testing heuristic value evaluation...")
    
    # Load test board
    try:
        with open('practice/testing/input.txt', 'r') as f:
            lines = f.readlines()
        
        board_data = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.strip()]
                board_data.append(row)
        
        print(f"Loaded board: {len(board_data)}x{len(board_data[0])}")
        
        # Create game board
        game_board = GameBoard(board_data)
        
        # Test heuristic value for both players
        for player in [0, 1]:
            value = heuristic_value_evaluation(game_board, player)
            print(f"Player {player} heuristic value: {value:.4f}")
        
        # Test self-play generator initialization
        print("\nTesting SelfPlayGenerator with heuristic...")
        generator = SelfPlayGenerator(
            model=None,  # No neural network
            mcts_simulations=10,
            mcts_time=0.1,
            temperature=1.0
        )
        
        print("SelfPlayGenerator initialized successfully with heuristic value!")
        
        # Test a single game (shortened)
        print("\nTesting single game...")
        game_data = generator.play_game(board_data, verbose=True)
        print(f"Game completed with {len(game_data)} data points")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_heuristic_value()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")