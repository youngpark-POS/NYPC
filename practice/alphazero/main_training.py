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

# 프로젝트 루트에서 실행할 수 있도록 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# 프로젝트 루트에서 실행되는 경우를 대비한 경로 추가
if os.path.basename(os.getcwd()) != 'alphazero':
    sys.path.append(os.path.join(os.getcwd(), 'practice', 'alphazero'))

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

def generate_random_board(rows: int = 10, cols: int = 17) -> list[list[int]]:
    """랜덤 게임 보드 생성 (1-9 범위)"""
    import random
    return [[random.randint(1, 9) for _ in range(cols)] for _ in range(rows)]

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--selfplay-games', type=int, default=20, help='Number of self-play games per iteration')
    parser.add_argument('--training-epochs', type=int, default=10, help='Training epochs per iteration')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--time-limit', type=float, default=1.0, help='MCTS time limit in seconds')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='practice/models', help='Model save directory')
    parser.add_argument('--project-name', type=str, default='mushroom_game', help='Project name for model directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--mcts-engine', type=str, default='neural', choices=['neural', 'heuristic'], 
                       help='MCTS engine type: neural (slow, accurate) or heuristic (fast, simple)')
    
    args = parser.parse_args()
    
    # 프로젝트별 디렉토리 생성
    project_save_dir = os.path.join(args.save_dir, args.project_name)
    
    print("=" * 60)
    print("AlphaZero Training Started")
    print("=" * 60)
    print(f"Configurations:")
    print(f"  Project: {args.project_name}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Self-play games per iteration: {args.selfplay_games}")
    print(f"  Training epochs per iteration: {args.training_epochs}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  MCTS time limit: {args.time_limit}s")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Save directory: {project_save_dir}")
    print(f"  Random boards: Always enabled")
    print(f"  MCTS engine: {args.mcts_engine}")
    print("=" * 60)
    
    # 디렉토리 생성
    os.makedirs(project_save_dir, exist_ok=True)
    
    # 랜덤 보드만 사용하므로 기본 보드 생성 (실제로는 사용되지 않음)
    initial_board = generate_random_board()
    print(f"Using random boards for training (10x17)")
    
    # 모델 생성 (올바른 액션 공간 크기로)
    temp_board = [[1] * 17 for _ in range(10)]
    from game_board import GameBoard
    temp_game = GameBoard(temp_board)
    action_space_size = temp_game.get_action_space_size()
    
    model = AlphaZeroNet(hidden_channels=128, action_space_size=action_space_size)
    if args.verbose:
        print(f"Model: {sum(p.numel() for p in model.parameters())} parameters, action space: {action_space_size}")
    
    # GPU/CPU 정보
    device_info = f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    
    # 훈련 관리자 생성
    trainer = TrainingManager(model, project_save_dir)
    
    # 이전 모델 자동 로드
    if args.resume:
        # 특정 파일 로드
        if trainer.load_model(args.resume):
            print(f"Resuming from saved model: {args.resume}")
        else:
            print(f"Could not load model: {args.resume}, starting fresh")
    else:
        # 자동으로 latest_model.pth 찾아서 로드
        if trainer.load_model("latest_model.pth"):
            print("Automatically loaded previous model: latest_model.pth")
        else:
            print("No previous model found, starting fresh")
    
    # 훈련 루프
    
    for iteration in range(args.iterations):
        print(f"\n{'='*20} Iteration {iteration + 1}/{args.iterations} {'='*20}")
        
        
        # 1. 셀프플레이 데이터 생성
        start_time = time.time()
        
        # MCTS 엔진 초기화 (Path Compression 기반)
        print(f"   🚀 Using Path Compression MCTS")
        mcts = MCTS(
            neural_network=model, 
            num_simulations=args.simulations,
            c_puct=1.0,
            time_limit=args.time_limit,
            engine_type=args.mcts_engine
        )
        
        selfplay_generator = SelfPlayGenerator(
            model, 
            num_simulations=args.simulations,
            temperature=1.0 if iteration < args.iterations // 2 else 0.1,  # 후반부에는 temperature 낮춤
            engine_type=args.mcts_engine,
            time_limit=args.time_limit
        )
        
        game_data_list = selfplay_generator.generate_games(
            initial_board, 
            args.selfplay_games, 
            verbose=args.verbose,
            use_random_boards=True
        )
        
        selfplay_time = time.time() - start_time
        total_samples = sum(len(data) for data in game_data_list)
        print(f"Self-play completed in {selfplay_time:.1f}s, generated {total_samples} training samples")
        
        # 표준 AlphaZero: 매 iteration마다 새로운 데이터만 사용
        print(f"Training on fresh data: {len(game_data_list)} games")
        
        # 2. 신경망 훈련
        print(f"Training neural network for {args.training_epochs} epochs...")
        start_time = time.time()
        
        final_stats = trainer.train_from_self_play_data(
            game_data_list,
            epochs=args.training_epochs,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        print(f"Final losses - Total: {final_stats['total_loss']:.4f}, "
              f"Policy: {final_stats['policy_loss']:.4f}, "
              f"Value: {final_stats['value_loss']:.4f}")
        
        
        # 3. 최신 모델만 저장
        trainer.save_model("latest_model.pth")
        
        # 4. 간단한 성능 평가
        if len(game_data_list) > 0:
            eval_data = game_data_list[:min(5, len(game_data_list))]  # 5게임 샘플만 평가
            eval_stats = trainer.evaluate_model(eval_data)
            print(f"Evaluation - Accuracy: {eval_stats['accuracy']:.3f}, "
                  f"Value MAE: {eval_stats['value_mae']:.3f}")
        
        print(f"Iteration {iteration + 1} completed! Model saved as latest_model.pth")
    
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
    
    print(f"Models saved in: {project_save_dir}")
    print("Training completed successfully!")



if __name__ == "__main__":
    main()