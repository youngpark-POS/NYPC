import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# GPU/CPU 자동 감지
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_all_predictions(policy_output: torch.Tensor, training: bool = False, max_delta_r: int = 9, max_delta_c: int = 16) -> List[List[Tuple[int, int, int, int, float]]]:
    """
    모든 예측을 박스 좌표로 변환 (연속값 기반, threshold 없이)
    
    Args:
        policy_output: (batch_size, 9, H, W) 텐서
                      각 앵커: [delta_r, delta_c, confidence]
        training: 학습 모드 여부 (True이면 연속값, False이면 정수)
        max_delta_r: 최대 세로 오프셋
        max_delta_c: 최대 가로 오프셋
        
    Returns:
        List[List[Tuple[r1, c1, r2, c2, confidence]]] 배치별 박스 리스트
    """
    batch_size, _, H, W = policy_output.shape
    all_boxes = []
    
    for b in range(batch_size):
        boxes = []
        for r1 in range(H):
            for c1 in range(W):
                for anchor in range(3):
                    # 각 앵커의 출력 추출 (연속값 처리)
                    delta_r_raw = policy_output[b, anchor*3, r1, c1]
                    delta_c_raw = policy_output[b, anchor*3+1, r1, c1] 
                    conf_raw = policy_output[b, anchor*3+2, r1, c1]
                    
                    # 시그모이드로 0~1 범위로 정규화 후 최대값으로 스케일링
                    delta_r_sigmoid = torch.sigmoid(delta_r_raw)
                    delta_c_sigmoid = torch.sigmoid(delta_c_raw)
                    delta_r_scaled = delta_r_sigmoid * max_delta_r
                    delta_c_scaled = delta_c_sigmoid * max_delta_c
                    
                    # 절대 좌표 계산 (경계 내 제한)
                    r2_continuous = torch.clamp(r1 + delta_r_scaled, 0, H-1)
                    c2_continuous = torch.clamp(c1 + delta_c_scaled, 0, W-1)
                    
                    # 신뢰도 계산
                    conf = torch.sigmoid(conf_raw)
                    
                    if training:
                        # 학습 시: 연속값 유지
                        r2 = r2_continuous
                        c2 = c2_continuous
                        conf_val = conf
                    else:
                        # 추론 시: 정수 변환
                        r2 = int(torch.round(r2_continuous).item())
                        c2 = int(torch.round(c2_continuous).item())
                        conf_val = conf.item()
                    
                    # 유효한 박스인지 확인 (최소 면적만 체크)
                    if training:
                        # 학습 시: 연속값으로 면적 계산
                        area = (r2_continuous - r1 + 1) * (c2_continuous - c1 + 1)
                        if area >= 2:
                            boxes.append((r1, c1, r2, c2, conf_val))
                    else:
                        # 추론 시: 정수값으로 면적 계산
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:
                            boxes.append((r1, c1, r2, c2, conf_val))
        
        all_boxes.append(boxes)
    
    return all_boxes

def find_exact_match(box_list: List[Tuple], target_coords: Tuple[int, int, int, int]):
    """
    정확한 좌표 매칭 찾기
    
    Args:
        box_list: [(r1, c1, r2, c2, conf), ...] 형태의 박스 리스트
        target_coords: (r1, c1, r2, c2) 찾을 좌표
        
    Returns:
        매칭되는 박스 또는 None
    """
    r1, c1, r2, c2 = target_coords
    for box in box_list:
        if len(box) >= 4 and box[0] == r1 and box[1] == c1 and box[2] == r2 and box[3] == c2:
            return box
    return None

def filter_valid_predictions(policy_output: torch.Tensor, valid_moves: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, float]]:
    """
    valid_moves에 대한 confidence 값만 추출 (단순화)
    
    Args:
        policy_output: (1, 9, H, W) 신경망 출력
        valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
        
    Returns:
        [(r1, c1, r2, c2, confidence), ...] valid_moves 순서대로
    """
    batch_size, _, H, W = policy_output.shape
    filtered_boxes = []
    
    # valid_moves에 대해 직접 confidence 추출
    for move in valid_moves:
        if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
            r1, c1, r2, c2 = move
            
            # 해당 좌표에서 가장 높은 confidence를 가진 앵커 찾기
            best_confidence = 0.0
            
            for anchor in range(3):
                # 각 앵커에서 예측된 델타와 confidence 계산
                delta_r_raw = policy_output[0, anchor*3, r1, c1]
                delta_c_raw = policy_output[0, anchor*3+1, r1, c1] 
                conf_raw = policy_output[0, anchor*3+2, r1, c1]
                
                # 연속값으로 예측 좌표 계산
                delta_r_sigmoid = torch.sigmoid(delta_r_raw)
                delta_c_sigmoid = torch.sigmoid(delta_c_raw)
                delta_r_scaled = delta_r_sigmoid * 9  # max_delta_r
                delta_c_scaled = delta_c_sigmoid * 16  # max_delta_c
                
                pred_r2 = torch.clamp(r1 + delta_r_scaled, 0, H-1)
                pred_c2 = torch.clamp(c1 + delta_c_scaled, 0, W-1)
                pred_conf = torch.sigmoid(conf_raw)
                
                # 정수 좌표로 변환하여 정확한 매칭 확인
                pred_r2_int = int(torch.round(pred_r2).item())
                pred_c2_int = int(torch.round(pred_c2).item())
                
                # 정확히 일치하는 경우 confidence 기록
                if pred_r2_int == r2 and pred_c2_int == c2:
                    confidence = pred_conf.item()
                    if confidence > best_confidence:
                        best_confidence = confidence
            
            # 매칭된 confidence 또는 기본값 사용
            final_confidence = best_confidence if best_confidence > 0 else 1e-6
            filtered_boxes.append((r1, c1, r2, c2, final_confidence))
        else:
            # 패스는 기본 confidence
            filtered_boxes.append(move + (1e-6,))
    
    return filtered_boxes

class BoxDetectionLoss(nn.Module):
    """
    YOLO-style 박스 검출 손실
    타겟: 절대 좌표 (r1, c1, r2, c2) -> 상대 오프셋으로 자동 변환
    """
    
    def __init__(self, missing_penalty: float = 1.0, false_positive_penalty: float = 1.0):
        super(BoxDetectionLoss, self).__init__()
        self.missing_penalty = missing_penalty
        self.false_positive_penalty = false_positive_penalty
    
    def forward(self, policy_output: torch.Tensor, policy_targets: List[List[Tuple[int, int, int, int, float]]]) -> torch.Tensor:
        """
        정확한 좌표 매칭 기반 손실 계산
        
        Args:
            policy_output: (batch_size, 9, H, W) 신경망 출력 (delta_r, delta_c, confidence)
            policy_targets: List[List[Tuple[r1, c1, r2, c2, target_probability]]] AlphaZero 정책 타겟
            
        Returns:
            total_loss: 전체 손실값
        """
        batch_size, _, H, W = policy_output.shape
        total_loss = torch.tensor(0.0, device=policy_output.device, requires_grad=True)
        
        for batch_idx in range(min(batch_size, len(policy_targets))):
            true_boxes = policy_targets[batch_idx]  # [(r1,c1,r2,c2,target_prob), ...]
            
            # 모든 위치와 앵커에서 예측 처리
            for r in range(H):
                for c in range(W):
                    for anchor in range(3):
                        # 예측된 박스 계산
                        delta_r_raw = policy_output[batch_idx, anchor*3, r, c]
                        delta_c_raw = policy_output[batch_idx, anchor*3+1, r, c] 
                        conf_raw = policy_output[batch_idx, anchor*3+2, r, c]
                        
                        # 연속값으로 예측 좌표 계산
                        delta_r_sigmoid = torch.sigmoid(delta_r_raw)
                        delta_c_sigmoid = torch.sigmoid(delta_c_raw)
                        delta_r_scaled = delta_r_sigmoid * 9  # max_delta_r
                        delta_c_scaled = delta_c_sigmoid * 16  # max_delta_c
                        
                        pred_r2 = torch.clamp(r + delta_r_scaled, 0, H-1)
                        pred_c2 = torch.clamp(c + delta_c_scaled, 0, W-1)
                        pred_conf = torch.sigmoid(conf_raw)
                        
                        # 정수 좌표로 변환하여 정확한 매칭 찾기
                        pred_r2_int = int(torch.round(pred_r2).item())
                        pred_c2_int = int(torch.round(pred_c2).item())
                        
                        # 정확히 일치하는 타겟 찾기
                        matched_target = None
                        for true_box in true_boxes:
                            if len(true_box) >= 5:
                                tr1, tc1, tr2, tc2, target_prob = true_box
                                
                                # 정수 좌표 직접 비교 (완벽한 일치만)
                                if r == tr1 and c == tc1 and pred_r2_int == tr2 and pred_c2_int == tc2:
                                    matched_target = true_box
                                    break
                        
                        # 매칭 여부에 따라 손실 계산
                        if matched_target is not None:
                            # 매칭됨: 연속값 기반 좌표 손실 + 신뢰도 손실
                            tr1, tc1, tr2, tc2, target_prob = matched_target
                            
                            # 연속값으로 정확한 좌표 회귀 손실
                            coord_loss = (torch.abs(pred_r2 - torch.tensor(tr2, device=policy_output.device, dtype=torch.float32)) + 
                                        torch.abs(pred_c2 - torch.tensor(tc2, device=policy_output.device, dtype=torch.float32)))
                            conf_loss = (pred_conf - target_prob) ** 2
                            
                            total_loss = total_loss + coord_loss + conf_loss
                        else:
                            # 매칭 안됨: False Positive 페널티 (높은 confidence에 비례)
                            fp_penalty = (pred_conf ** 2) * self.false_positive_penalty
                            total_loss = total_loss + fp_penalty
        
        return total_loss / max(1, batch_size * H * W * 3)  # 전체 예측 수로 정규화

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
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
    def __init__(self, input_channels: int = 2, hidden_channels: int = 128, 
                 board_height: int = 10, board_width: int = 17):
        super(AlphaZeroNet, self).__init__()
        
        self.device = device
        self.board_height = board_height
        self.board_width = board_width
        
        # 백본 네트워크 (입력층 + 2개 잔차블록)
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        
        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)
        self.res_block3 = ResidualBlock(hidden_channels)
        self.res_block4 = ResidualBlock(hidden_channels)
        
        # 정책 헤드 (9채널 YOLO 스타일 출력)
        # 3개 앵커 × (delta_r, delta_c, confidence) = 9채널
        # delta_r, delta_c: 현재 위치에서 우측/하측으로 얼마나 이동할지 (상대 오프셋)
        self.policy_conv = nn.Conv2d(hidden_channels, 9, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(9)
        
        # 가치 헤드 (1채널 컨볼루션 + FC)
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_height * board_width, hidden_channels)
        self.value_fc2 = nn.Linear(hidden_channels, 1)
        
        # 모델을 GPU로 이전
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

    def predict(self, state: np.ndarray, valid_moves: list) -> tuple:
        """
        단일 상태에 대한 예측 (YOLO 스타일 박스 검출 - 상대 오프셋 기반)
        Args:
            state: (2, 10, 17) 형태의 numpy 배열
            valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
        Returns:
            policy_probs: 유효한 움직임에 대한 확률 분포 (valid_moves 순서대로)
            value: 상태 가치 (-1 ~ 1)
        네트워크 출력: 9채널 (3앵커 × [delta_r, delta_c, confidence])
        """
        self.eval()
        with torch.no_grad():
            # 입력 텐서 변환 및 GPU로 이전
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2, 10, 17)
            
            # 신경망 예측 (YOLO 스타일) - GPU에서 실행
            policy_output, value = self.forward(state_tensor)  # (1, 9, 10, 17), (1, 1)
            
            # CPU로 결과 이전
            value = value.squeeze().cpu().item()
            
            # 유효한 움직임이 없으면 패스만 가능
            if not valid_moves:
                return [1.0], value
            
            # valid_moves에 대한 confidence 추출 (threshold 없이)
            filtered_boxes = filter_valid_predictions(policy_output, valid_moves)
            
            # confidence 값들로 직접 softmax 적용
            confidences = [box[4] for box in filtered_boxes]
            confidence_tensor = torch.tensor(confidences, dtype=torch.float32)
            move_probs = F.softmax(confidence_tensor, dim=0).tolist()
            
            return move_probs, value

class AlphaZeroTrainer:
    def __init__(self, model: AlphaZeroNet, lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.device = model.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = BoxDetectionLoss()
        self.value_loss_fn = nn.MSELoss()
        
    def train_step(self, states: np.ndarray, policy_targets: List[List[Tuple[int, int, int, int, float]]], 
                   value_targets: np.ndarray) -> dict:
        """
        한 번의 훈련 스텝 수행 (YOLO 스타일)
        Args:
            states: (batch_size, 2, 10, 17)
            policy_targets: List[List[Tuple[r1, c1, r2, c2, probability]]] 박스 리스트
            value_targets: (batch_size,) 게임 결과
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 텐서 변환 및 GPU로 이전
        states_tensor = torch.FloatTensor(states).to(self.device)
        value_targets_tensor = torch.FloatTensor(value_targets).unsqueeze(1).to(self.device)
        
        # 신경망 예측 (GPU에서 실행)
        policy_output, value_pred = self.model(states_tensor)
        
        # 정책 손실 계산 (박스 검출 손실)
        policy_loss = self.loss_fn(policy_output, policy_targets)
        
        # 가치 손실 계산
        value_loss = self.value_loss_fn(value_pred, value_targets_tensor)
        
        # 총 손실
        total_loss = policy_loss + value_loss
        
        # 역전파
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_model_as_binary(self, filepath: str):
        """바이너리 형태로 모델 저장 (대회 제출용)"""
        # 모델의 state_dict를 바이너리로 직렬화
        model_data = self.model.state_dict()
        
        # 텐서들을 numpy 배열로 변환하여 저장
        binary_data = {}
        for key, tensor in model_data.items():
            binary_data[key] = tensor.cpu().numpy()
        
        # pickle을 사용하여 바이너리 파일로 저장
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(binary_data, f)
        
        print(f"Model saved as binary to {filepath}")