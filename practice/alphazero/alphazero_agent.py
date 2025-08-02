import sys
import os
import torch
import numpy as np
from typing import Optional
import pickle

# 현재 디렉토리를 Python 패스에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_board import GameBoard  
from neural_network import AlphaZeroNet
from mcts import MCTS

class AlphaZeroAgent:
    def __init__(self, model_path: Optional[str] = None, num_simulations: int = 400):
        # 임시 게임보드로 액션 공간 크기 계산
        temp_board = [[1] * 17 for _ in range(10)]
        temp_game = GameBoard(temp_board)
        action_space_size = temp_game.get_action_space_size()
        
        self.model = AlphaZeroNet(action_space_size=action_space_size)
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
        self.game_board = None
        self.player_id = None
        self.is_first_player = False
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        try:
            if model_path.endswith('.pth'):
                # PyTorch 모델 로드
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            elif model_path.endswith('.bin'):
                # 바이너리 형태로 저장된 모델 로드
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model.load_state_dict(model_data)
            
            self.model.eval()
            print(f"Model loaded from {model_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load model: {e}", file=sys.stderr)
            # 랜덤 모델로 시작
    
    def save_model_as_binary(self, save_path: str):
        """모델을 바이너리 형태로 저장 (대회 제출용)"""
        model_data = self.model.state_dict()
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def initialize(self, board: list, is_first: bool):
        """게임 초기화"""
        self.game_board = GameBoard(board)
        self.is_first_player = is_first
        self.player_id = 0 if is_first else 1
    
    def calculate_move(self, my_time: int, opp_time: int) -> tuple:
        """
        MCTS를 사용하여 최적의 움직임 계산
        시간에 따른 MCTS 시간 제한 조정
        """
        if self.game_board is None:
            return (-1, -1, -1, -1)
        
        # 시간에 따른 MCTS 시간 제한 조정
        if my_time < 1000:  # 1초 미만
            time_limit = 0.3  # 300ms
        elif my_time < 3000:  # 3초 미만  
            time_limit = 0.5  # 500ms
        elif my_time < 5000:  # 5초 미만
            time_limit = 0.8  # 800ms
        else:
            time_limit = 1.0  # 1초
        
        # 임시로 시간 제한 변경
        original_time_limit = self.mcts.time_limit
        self.mcts.time_limit = time_limit
        
        try:
            # MCTS로 최적 움직임 선택
            best_move = self.mcts.get_best_move(self.game_board, self.player_id, temperature=0.0)
            return best_move
        except Exception as e:
            print(f"MCTS error: {e}", file=sys.stderr)
            # 폴백: 첫 번째 유효한 움직임 선택
            valid_moves = self.game_board.get_valid_moves()
            if valid_moves:
                return valid_moves[0]
            else:
                return (-1, -1, -1, -1)
        finally:
            # 시간 제한 복원
            self.mcts.time_limit = original_time_limit
    
    def update_opponent_move(self, r1: int, c1: int, r2: int, c2: int, time_used: int):
        """상대방의 움직임을 보드에 반영"""
        if self.game_board is not None:
            opponent_id = 1 - self.player_id
            self.game_board.make_move(r1, c1, r2, c2, opponent_id)
    
    def update_my_move(self, r1: int, c1: int, r2: int, c2: int):
        """내 움직임을 보드에 반영"""
        if self.game_board is not None:
            self.game_board.make_move(r1, c1, r2, c2, self.player_id)

# 전역 변수
agent = None
first = False

def main():
    global agent, first
    
    # 명령행 인수에서 모델 경로 확인
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
            
            parts = line.split()
            command = parts[0]
            params = parts[1:] if len(parts) > 1 else []
            
            if command == "READY":
                # 게임 준비
                turn = params[0] if params else "SECOND"
                first = (turn == "FIRST")
                agent = AlphaZeroAgent(model_path)
                print("OK", flush=True)
                
            elif command == "INIT":
                # 보드 초기화
                if agent is None:
                    agent = AlphaZeroAgent(model_path)
                
                # 보드 데이터 파싱
                board_data = []
                for param in params:
                    row = [int(digit) for digit in param]
                    board_data.append(row)
                
                agent.initialize(board_data, first)
                
            elif command == "TIME":
                # 내 차례: 움직임 계산
                my_time = int(params[0]) if len(params) > 0 else 5000
                opp_time = int(params[1]) if len(params) > 1 else 5000
                
                move = agent.calculate_move(my_time, opp_time)
                agent.update_my_move(*move)
                print(f"{move[0]} {move[1]} {move[2]} {move[3]}", flush=True)
                
            elif command == "OPP":  
                # 상대방 움직임 반영
                r1, c1, r2, c2, time_used = map(int, params)
                agent.update_opponent_move(r1, c1, r2, c2, time_used)
                
            elif command == "FINISH":
                # 게임 종료
                break
                
            else:
                print(f"Unknown command: {command}", file=sys.stderr)
                
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            # 에러 발생시 패스
            print("-1 -1 -1 -1", flush=True)

if __name__ == "__main__":
    main()