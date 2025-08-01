#!/usr/bin/env python3
"""
AlphaZero 훈련 메인 스크립트
셀프플레이 -> 신경망 학습을 반복하는 Expert Iteration 구현
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# 현재 디렉토리를 Python 패스에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_board import GameBoard
from neural_network import AlphaZeroNet
from self_play import SelfPlayGenerator
from training import TrainingManager
from mcts import MCTS

def load_initial_board(input_file: str = "practice/testing/input.txt"):
    """초기 보드 로드"""
    try:
        with open(input_file, 'r') as f:
            board = []
            for line in f:
                row = [int(digit) for digit in line.strip()]
                board.append(row)
        return board
    except Exception as e:
        print(f"Error loading board: {e}")
        # 기본 보드 생성
        return [[1, 2, 3, 4, 5] * 3 + [1, 2] for _ in range(10)]

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--selfplay-games', type=int, default=20, help='Number of self-play games per iteration')
    parser.add_argument('--training-epochs', type=int, default=10, help='Training epochs per iteration')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='practice/models', help='Model save directory')
    parser.add_argument('--input-file', type=str, default='practice/testing/input.txt', help='Game board input file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AlphaZero Training Started")
    print("=" * 60)
    print(f"Configurations:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Self-play games per iteration: {args.selfplay_games}")
    print(f"  Training epochs per iteration: {args.training_epochs}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Save directory: {args.save_dir}")
    print("=" * 60)
    
    # 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 초기 보드 로드
    initial_board = load_initial_board(args.input_file)
    print(f"Loaded board with shape: {len(initial_board)}x{len(initial_board[0])}")
    
    # 모델 생성 (올바른 액션 공간 크기로)
    temp_board = [[1] * 17 for _ in range(10)]
    from game_board import GameBoard
    temp_game = GameBoard(temp_board)
    action_space_size = temp_game.get_action_space_size()
    
    model = AlphaZeroNet(hidden_channels=128, action_space_size=action_space_size)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Action space size: {action_space_size}")
    
    # 훈련 관리자 생성
    trainer = TrainingManager(model, args.save_dir)
    
    # 체크포인트에서 재시작
    start_iteration = 0
    if args.resume:
        if trainer.load_model(args.resume):
            # 파일명에서 iteration 번호 추출 시도
            try:
                start_iteration = int(args.resume.split('_')[-1].split('.')[0])
                print(f"Resuming from iteration {start_iteration}")
            except:
                print("Could not determine starting iteration, starting from 0")
    
    # 훈련 루프
    all_training_data = []
    
    for iteration in range(start_iteration, args.iterations):
        print(f"\n{'='*20} Iteration {iteration + 1}/{args.iterations} {'='*20}")
        
        # 1. 셀프플레이 데이터 생성
        print(f"Generating {args.selfplay_games} self-play games...")
        start_time = time.time()
        
        selfplay_generator = SelfPlayGenerator(
            model, 
            num_simulations=args.simulations,
            temperature=1.0 if iteration < args.iterations // 2 else 0.1  # 후반부에는 temperature 낮춤
        )
        
        game_data_list = selfplay_generator.generate_games(
            initial_board, 
            args.selfplay_games, 
            verbose=args.verbose
        )
        
        selfplay_time = time.time() - start_time
        total_samples = sum(len(data) for data in game_data_list)
        print(f"Self-play completed in {selfplay_time:.1f}s, generated {total_samples} training samples")
        
        # 데이터 누적 (최근 3번의 iteration 데이터만 유지)
        all_training_data.extend(game_data_list)
        if len(all_training_data) > args.selfplay_games * 3:
            all_training_data = all_training_data[-args.selfplay_games * 3:]
        
        # 2. 신경망 훈련
        print(f"Training neural network for {args.training_epochs} epochs...")
        start_time = time.time()
        
        final_stats = trainer.train_from_self_play_data(
            all_training_data,
            epochs=args.training_epochs,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        print(f"Final losses - Total: {final_stats['total_loss']:.4f}, "
              f"Policy: {final_stats['policy_loss']:.4f}, "
              f"Value: {final_stats['value_loss']:.4f}")
        
        # 3. 모델 저장
        model_filename = f"iteration_{iteration + 1}.pth"
        trainer.save_model(model_filename)
        
        # 최신 모델을 latest로 저장
        trainer.save_model("latest_model.pth")
        
        # 4. 간단한 성능 평가
        if len(game_data_list) > 0:
            eval_data = game_data_list[:min(5, len(game_data_list))]  # 5게임 샘플만 평가
            eval_stats = trainer.evaluate_model(eval_data)
            print(f"Evaluation - Accuracy: {eval_stats['accuracy']:.3f}, "
                  f"Value MAE: {eval_stats['value_mae']:.3f}")
        
        print(f"Iteration {iteration + 1} completed!")
    
    # 최종 모델을 바이너리 형태로 저장 (대회 제출용)
    print("\nSaving final model as binary for submission...")
    trainer.save_model_as_binary("data.bin")
    
    # 훈련 통계 출력
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    stats = trainer.get_training_stats()
    if stats:
        print(f"Total epochs: {stats['total_epochs']}")
        print(f"Final total loss: {stats['latest_total_loss']:.4f}")
        print(f"Final policy loss: {stats['latest_policy_loss']:.4f}")
        print(f"Final value loss: {stats['latest_value_loss']:.4f}")
        print(f"Best total loss: {stats['min_total_loss']:.4f}")
    
    print(f"Models saved in: {args.save_dir}")
    print("Training completed successfully!")

def quick_test():
    """빠른 테스트 함수"""
    print("Running quick test...")
    
    # 작은 보드로 테스트
    test_board = [
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
        [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
        [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    ]
    
    # 게임 보드 테스트
    print("Testing GameBoard...")
    game_board = GameBoard(test_board)
    valid_moves = game_board.get_valid_moves()
    print(f"Found {len(valid_moves)} valid moves")
    
    if valid_moves:
        first_move = valid_moves[0]
        print(f"Testing move: {first_move}")
        game_board.make_move(*first_move, 0)
        print("Move executed successfully")
    
    # 신경망 테스트
    print("Testing Neural Network...")
    action_space_size = game_board.get_action_space_size()
    model = AlphaZeroNet(action_space_size=action_space_size)
    state = game_board.get_state_tensor(0)
    print(f"State tensor shape: {state.shape}")
    print(f"Action space size: {action_space_size}")
    
    test_moves = valid_moves[:5] if len(valid_moves) >= 5 else valid_moves
    policy_probs, value = model.predict(state, test_moves, game_board)
    print(f"Policy probs: {policy_probs}")
    print(f"Value: {value}")
    
    # MCTS 테스트
    print("Testing MCTS...")
    mcts = MCTS(model, num_simulations=50)  # 적은 시뮬레이션으로 테스트
    best_move = mcts.get_best_move(game_board, 1, temperature=0.0)
    print(f"MCTS best move: {best_move}")
    
    print("Quick test completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()