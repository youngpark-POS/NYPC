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
        
        self.device = torch.device('cpu')
        self.board_height = board_height
        self.board_width = board_width
        
        # 백본 네트워크 (입력층 + 4개 잔차블록)
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        
        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)
        self.res_block3 = ResidualBlock(hidden_channels)
        self.res_block4 = ResidualBlock(hidden_channels)
        
        # 정책 헤드 (9채널 YOLO 스타일 출력)
        # 3개 앵커 × (delta_r, delta_c, confidence) = 9채널
        self.policy_conv = nn.Conv2d(hidden_channels, 9, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(9)
        
        # 가치 헤드 (1채널 컨볼루션 + FC)
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_height * board_width, hidden_channels)
        self.value_fc2 = nn.Linear(hidden_channels, 1)
        
        # 모델을 CPU로 설정
        self.to(self.device)
        
    def forward(self, x):
        # 백본
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # 정책 헤드 (YOLO 스타일)
        policy_output = self.policy_bn(self.policy_conv(x))  # (batch, 9, H, W)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_output, value

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
        
        # 모델 로드 시도
        self.load_model()
    
    def load_model(self):
        """data.bin에서 모델 로드 (현재 save_model_as_binary 형식)"""
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
            
            # data.bin 로드 (pickle로 저장된 numpy 배열들)
            with open(model_path, 'rb') as f:
                binary_data = pickle.load(f)
            
            # numpy 배열들을 텐서로 변환하여 state_dict 생성
            state_dict = {}
            for key, numpy_array in binary_data.items():
                state_dict[key] = torch.from_numpy(numpy_array)
            
            self.model.load_state_dict(state_dict)
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
                    # 버섯 값을 정규화 (1-9 -> 0.1-0.9)
                    state[0][i][j] = cell / 10.0
                elif cell == -(player_perspective + 1):
                    # 현재 플레이어가 점령한 칸
                    state[1][i][j] = 1.0
                elif cell == -(2 - player_perspective):
                    # 상대 플레이어가 점령한 칸
                    state[1][i][j] = -1.0
        
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
        """신경망을 사용한 움직임 예측 (현재 프레임워크와 동일한 로직)"""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # 상태 텐서 생성
            state_tensor = self.get_state_tensor()
            if state_tensor is None:
                return None
            
            # 텐서 변환
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)  # (1, 2, 10, 17)
            
            with torch.no_grad():
                policy_output, value = self.model(state_tensor)  # (1, 9, 10, 17), (1, 1)
            
            # 유효한 움직임이 없으면 패스만 가능
            if not valid_moves:
                return (-1, -1, -1, -1)
            
            # valid_moves에 대한 confidence 직접 추출
            confidences = []
            _, _, H, W = policy_output.shape
            
            for move in valid_moves:
                if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
                    r1, c1, r2, c2 = move
                    best_confidence = 0.0
                    
                    # 각 앵커에서 가장 높은 매칭 confidence 찾기
                    for anchor in range(3):
                        delta_r_raw = policy_output[0, anchor*3, r1, c1]
                        delta_c_raw = policy_output[0, anchor*3+1, r1, c1] 
                        conf_raw = policy_output[0, anchor*3+2, r1, c1]
                        
                        # 예측 좌표 계산
                        delta_r = torch.sigmoid(delta_r_raw) * 9
                        delta_c = torch.sigmoid(delta_c_raw) * 16
                        pred_r2 = int(torch.round(torch.clamp(r1 + delta_r, 0, H-1)).item())
                        pred_c2 = int(torch.round(torch.clamp(c1 + delta_c, 0, W-1)).item())
                        
                        # 완벽한 매칭인 경우 confidence 사용
                        if pred_r2 == r2 and pred_c2 == c2:
                            confidence = torch.sigmoid(conf_raw).item()
                            if confidence > best_confidence:
                                best_confidence = confidence
                    
                    confidences.append(best_confidence if best_confidence > 0 else 1e-6)
                else:
                    # 패스는 기본 confidence
                    confidences.append(1e-6)
            
            # Softmax로 확률 분포 계산
            confidence_tensor = torch.tensor(confidences, dtype=torch.float32)
            move_probs = F.softmax(confidence_tensor, dim=0)
            
            # 가장 높은 확률의 움직임 선택
            best_idx = torch.argmax(move_probs).item()
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