#!/usr/bin/env python3
"""
대회 제출용 AlphaZero Agent - 단일 파일, 외부 의존성 없음
data.bin 파일에서 신경망 모델을 로드하여 게임을 플레이합니다.
"""

import sys
import os
import pickle
import math

# PyTorch 관련 임포트
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using random strategy", file=sys.stderr)

# ================================
# PyTorch 신경망 모델 정의
# ================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=128, board_height=10, board_width=17):
        super(AlphaZeroNet, self).__init__()
        
        # 액션 공간 크기 계산 (모든 가능한 직사각형 + 패스)
        total_actions = 0
        for r1 in range(board_height):
            for c1 in range(board_width):
                for r2 in range(r1, board_height):
                    for c2 in range(c1, board_width):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:
                            total_actions += 1
        self.action_space_size = total_actions + 1  # +1 for pass
        
        self.board_height = board_height
        self.board_width = board_width
        
        # 백본 네트워크
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)
        
        # 정책 헤드
        self.policy_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(1)
        self.policy_fc = nn.Linear(1 * board_height * board_width, self.action_space_size)
        
        # 가치 헤드
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_height * board_width, hidden_channels)
        self.value_fc2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, x):
        # 백본
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 정책 헤드
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

# ================================
# 게임 로직 클래스 (sample_code.py 기반)
# ================================

class Game:
    def __init__(self, board, first):
        self.board = board            # 게임 보드 (2차원 배열)
        self.first = first            # 선공 여부
        self.passed = False           # 마지막 턴에 패스했는지 여부
        
        # 신경망 모델 관련
        self.model = None
        self.device = torch.device('cpu')  # CPU 사용
        
        # 액션 매핑 테이블 생성
        self.action_to_move = {}
        self.move_to_action = {}
        self._build_action_mapping()
        
        # 모델 로드 시도
        self.load_model()
    
    def _build_action_mapping(self):
        """액션 인덱스와 움직임 간의 매핑 테이블 생성"""
        action_idx = 0
        R, C = 10, 17
        
        for r1 in range(R):
            for c1 in range(C):
                for r2 in range(r1, R):
                    for c2 in range(c1, C):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:
                            move = (r1, c1, r2, c2)
                            self.action_to_move[action_idx] = move
                            self.move_to_action[move] = action_idx
                            action_idx += 1
        
        # 패스 액션
        self.action_to_move[action_idx] = (-1, -1, -1, -1)
        self.move_to_action[(-1, -1, -1, -1)] = action_idx
    
    def load_model(self):
        """data.bin에서 모델 로드"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # 현재 디렉토리에서 data.bin 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'data.bin')
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}", file=sys.stderr)
                return
            
            # 모델 생성
            self.model = AlphaZeroNet()
            
            # data.bin 로드
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model.load_state_dict(model_data)
            
            self.model.eval()
            print("Neural network model loaded successfully", file=sys.stderr)
            
        except Exception as e:
            print(f"Failed to load model: {e}", file=sys.stderr)
            self.model = None
    
    def get_state_tensor(self):
        """현재 보드 상태를 신경망 입력 텐서로 변환"""
        if not TORCH_AVAILABLE:
            return None
        
        state = np.zeros((2, 10, 17), dtype=np.float32)
        player_perspective = 0 if self.first else 1
        
        for i in range(10):
            for j in range(17):
                cell = self.board[i][j]
                if cell > 0:
                    # 버섯 값 정규화 (1-9 -> 0.1-0.9)
                    state[0][i][j] = cell / 10.0
                elif cell < 0:
                    # 점령된 칸 (-1: 플레이어 0, -2: 플레이어 1)
                    if cell == -(player_perspective + 1):
                        state[1][i][j] = 1.0  # 내가 점령한 칸
                    else:
                        state[1][i][j] = -1.0  # 상대가 점령한 칸
        
        return state
    
    def get_valid_moves(self):
        """현재 상태에서 유효한 움직임들을 반환"""
        valid_moves = []
        
        for r1 in range(10):
            for c1 in range(17):
                for r2 in range(r1, 10):
                    for c2 in range(c1, 17):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2 and self.isValid(r1, c1, r2, c2):
                            valid_moves.append((r1, c1, r2, c2))
        
        return valid_moves
    
    def neural_network_predict(self, valid_moves):
        """신경망을 사용한 움직임 예측"""
        if not TORCH_AVAILABLE or self.model is None or not valid_moves:
            return None
        
        try:
            # 상태 텐서 생성
            state_tensor = self.get_state_tensor()
            if state_tensor is None:
                return None
            
            # 텐서 변환
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)
            
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                policy_logits = policy_logits.squeeze(0)
            
            # 유효한 움직임들에 대한 확률 계산
            valid_indices = []
            for move in valid_moves:
                action_idx = self.move_to_action.get(move)
                if action_idx is not None:
                    valid_indices.append(action_idx)
            
            if not valid_indices:
                return None
            
            # 유효한 액션들의 로짓만 추출
            valid_logits = policy_logits[valid_indices]
            
            # 소프트맥스 적용
            probs = F.softmax(valid_logits, dim=0)
            
            # 가장 높은 확률의 움직임 선택
            best_idx = torch.argmax(probs).item()
            return valid_moves[best_idx]
            
        except Exception as e:
            print(f"Neural network prediction error: {e}", file=sys.stderr)
            return None
    
    # 사각형 (r1, c1) ~ (r2, c2)이 유효한지 검사 (합이 10이고, 네 변을 모두 포함)
    def isValid(self, r1, c1, r2, c2):
        sums = 0
        r1fit = c1fit = r2fit = c2fit = False

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] > 0:  # 양수인 경우만 (버섯)
                    sums += self.board[r][c]
                    if r == r1:
                        r1fit = True
                    if r == r2:
                        r2fit = True
                    if c == c1:
                        c1fit = True
                    if c == c2:
                        c2fit = True
        return sums == 10 and r1fit and r2fit and c1fit and c2fit

    # ================================================================
    # ===================== [필수 구현] ===============================
    # 합이 10인 유효한 사각형을 찾아 (r1, c1, r2, c2) 튜플로 반환
    # 없으면 (-1, -1, -1, -1) 반환 (패스 의미)
    # ================================================================
    def calculateMove(self, _myTime, _oppTime):
        # 1. 유효한 움직임들 찾기
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            return (-1, -1, -1, -1)  # 패스
        
        # 2. 신경망 예측 시도
        neural_move = self.neural_network_predict(valid_moves)
        if neural_move is not None:
            return neural_move
        
        # 3. 폴백: 가장 큰 영역 선택
        best_move = valid_moves[0]
        best_area = 0
        
        for move in valid_moves:
            r1, c1, r2, c2 = move
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            if area > best_area:
                best_area = area
                best_move = move
        
        return best_move
    # =================== [필수 구현 끝] =============================

    # 상대방의 수를 받아 보드에 반영
    def updateOpponentAction(self, action, _time):
        self.updateMove(*action, False)

    # 주어진 수를 보드에 반영 (칸을 점령 상태로 변경)
    def updateMove(self, r1, c1, r2, c2, isMyMove):
        if r1 == c1 == r2 == c2 == -1:
            self.passed = True
            return
        
        # 점령 표시: 내 움직임이면 -1 (first) 또는 -2 (second), 상대방은 반대
        player_marker = -1 if self.first else -2
        if not isMyMove:
            player_marker = -2 if self.first else -1
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = player_marker
        
        self.passed = False

# ================================
# main(): 입출력 처리 및 게임 진행
# ================================

def main():
    global game, first
    
    while True:
        try:
            line = input().split()

            if len(line) == 0:
                continue

            command, *param = line

            if command == "READY":
                # 선공 여부 확인
                turn = param[0]
                first = turn == "FIRST"
                print("OK", flush=True)
                continue

            if command == "INIT":
                # 보드 초기화
                board = [list(map(int, row)) for row in param]
                game = Game(board, first)
                continue

            if command == "TIME":
                # 내 턴: 수 계산 및 실행
                myTime, oppTime = map(int, param)
                ret = game.calculateMove(myTime, oppTime)
                game.updateMove(*ret, True)
                print(*ret, flush=True)
                continue

            if command == "OPP":
                # 상대 턴 반영
                r1, c1, r2, c2, time = map(int, param)
                game.updateOpponentAction((r1, c1, r2, c2), time)
                continue

            if command == "FINISH":
                break

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            # 에러 시 패스
            print("-1 -1 -1 -1", flush=True)

if __name__ == "__main__":
    main()