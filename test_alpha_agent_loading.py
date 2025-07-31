#!/usr/bin/env python3
"""
Test AlphaZero agent model loading
"""

import sys
import os
sys.path.insert(0, 'practice')

try:
    import torch
    from core.value_net import create_combined_net
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

def test_model_loading():
    print("Testing AlphaZero agent model loading...")
    
    # Test different model paths
    model_paths = [
        "experiments/my_first_model/latest_model.pth",
        "experiments/my_first_model/model_iter_000.pth"
    ]
    
    for model_path in model_paths:
        print(f"\nTesting model: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
        
        try:
            # Test loading the model
            combined_net = create_combined_net('cpu')
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print(f"✅ Model loaded successfully")
            print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
            print(f"   Model type: {checkpoint.get('model_type', 'unknown')}")
            print(f"   Iteration: {checkpoint.get('iteration', 'unknown')}")
            
            # Test loading state dict
            combined_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ State dict loaded successfully")
            
            # Test inference
            import numpy as np
            from core.game_board import GameBoard
            from training.self_play import load_board_config
            
            board_data = load_board_config()
            game_board = GameBoard(board_data)
            features = game_board.to_neural_features()
            valid_moves = game_board.get_valid_moves()
            
            move_probs, win_prob = combined_net.predict_policy_value(features, valid_moves)
            print(f"✅ Model inference successful")
            print(f"   Win probability: {win_prob:.3f}")
            print(f"   Top 3 moves: {sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()

def test_alpha_agent_direct():
    """Test alpha agent directly with model path"""
    print(f"\n{'='*60}")
    print("Testing Alpha Agent Direct Usage")
    print(f"{'='*60}")
    
    model_path = "experiments/my_first_model/latest_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Simulate agent initialization 
    try:
        from agents.alpha_agent import AlphaGame
        from training.self_play import load_board_config
        
        board_data = load_board_config()
        
        print("Creating AlphaGame with trained model...")
        alpha_game = AlphaGame(
            board=board_data,
            first=True,  # Player 1
            model_path=model_path,
            mcts_simulations=100,  # Reasonable for testing
            mcts_time=1.0
        )
        
        print("✅ AlphaGame created successfully")
        
        # Test move calculation
        print("Testing move calculation...")
        best_move = alpha_game.calculateMove(myTime=5.0, oppTime=5.0)
        
        move_str = f"({best_move[0]},{best_move[1]})-({best_move[2]},{best_move[3]})" if best_move[0] != -1 else "PASS"
        print(f"✅ AlphaZero selected move: {move_str}")
        
    except Exception as e:
        print(f"❌ Alpha agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
    test_alpha_agent_direct()