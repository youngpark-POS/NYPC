import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# GPU/CPU 자동 감지
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_all_predictions(policy_output: torch.Tensor, validity_threshold: float = 0.5, training: bool = False, max_delta_r: int = 9, max_delta_c: int = 16) -> List[List[Tuple[int, int, int, int, float, float]]]:
    """
    모든 예측을 박스 좌표로 변환 (유효도/선호도 분리)
    
    Args:
        policy_output: (batch_size, 12, H, W) 텐서
                      각 앵커: [delta_r, delta_c, validity, preference]
        validity_threshold: 유효도 임계값
        training: 학습 모드 여부 (True이면 연속값, False이면 정수)
        max_delta_r: 최대 세로 오프셋
        max_delta_c: 최대 가로 오프셋
        
    Returns:
        List[List[Tuple[r1, c1, r2, c2, validity, preference]]] 배치별 박스 리스트
    """
    batch_size, _, H, W = policy_output.shape
    all_boxes = []
    
    for b in range(batch_size):
        boxes = []
        for r1 in range(H):
            for c1 in range(W):
                for anchor in range(3):
                    # 각 앵커의 출력 추출 (유효도/선호도 분리)
                    delta_r_raw = policy_output[b, anchor*4, r1, c1]
                    delta_c_raw = policy_output[b, anchor*4+1, r1, c1] 
                    validity_raw = policy_output[b, anchor*4+2, r1, c1]
                    preference_raw = policy_output[b, anchor*4+3, r1, c1]
                    
                    # 시그모이드로 0~1 범위로 정규화 후 최대값으로 스케일링
                    delta_r_sigmoid = torch.sigmoid(delta_r_raw)
                    delta_c_sigmoid = torch.sigmoid(delta_c_raw)
                    delta_r_scaled = delta_r_sigmoid * max_delta_r
                    delta_c_scaled = delta_c_sigmoid * max_delta_c
                    
                    # 절대 좌표 계산 (경계 내 제한)
                    r2_continuous = torch.clamp(r1 + delta_r_scaled, 0, H-1)
                    c2_continuous = torch.clamp(c1 + delta_c_scaled, 0, W-1)
                    
                    # 유효도와 선호도 계산
                    validity = torch.sigmoid(validity_raw)
                    preference = torch.sigmoid(preference_raw)
                    
                    if training:
                        # 학습 시: 연속값 유지
                        r2 = r2_continuous
                        c2 = c2_continuous
                        validity_val = validity
                        preference_val = preference
                    else:
                        # 추론 시: 정수 변환
                        r2 = int(torch.round(r2_continuous).item())
                        c2 = int(torch.round(c2_continuous).item())
                        validity_val = validity.item()
                        preference_val = preference.item()
                    
                    # 유효한 박스인지 확인 (유효도 임계값 기준)
                    if training:
                        # 학습 시: 연속값으로 면적 계산
                        area = (r2_continuous - r1 + 1) * (c2_continuous - c1 + 1)
                        if validity > validity_threshold and area >= 2:
                            boxes.append((r1, c1, r2, c2, validity_val, preference_val))
                    else:
                        # 추론 시: 정수값으로 면적 계산
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if validity_val > validity_threshold and area >= 2:
                            boxes.append((r1, c1, r2, c2, validity_val, preference_val))
        
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

def filter_valid_predictions(policy_output: torch.Tensor, valid_moves: List[Tuple[int, int, int, int]], 
                            validity_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """
    신경망 출력에서 valid_moves만 필터링하고 선호도 반환
    
    Args:
        policy_output: (1, 12, H, W) 신경망 출력
        valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
        validity_threshold: 유효도 임계값
        
    Returns:
        필터링된 박스 리스트 [(r1, c1, r2, c2, preference), ...]
    """
    # 모든 예측 디코딩 (유효도 필터링 적용)
    all_predictions = decode_all_predictions(policy_output, validity_threshold=validity_threshold, training=False)
    if not all_predictions:
        return []
    
    predicted_boxes = all_predictions[0]  # 첫 번째 배치
    
    # valid_moves와 매칭되는 예측만 선별
    filtered_boxes = []
    for move in valid_moves:
        if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
            r1, c1, r2, c2 = move
            matched_box = find_exact_match(predicted_boxes, (r1, c1, r2, c2))
            if matched_box and len(matched_box) >= 6:  # (r1, c1, r2, c2, validity, preference)
                # 선호도만 사용
                preference = matched_box[5]
                filtered_boxes.append((r1, c1, r2, c2, preference))
            else:
                # 매칭되지 않으면 최소 선호도로 설정
                filtered_boxes.append((r1, c1, r2, c2, 1e-6))
    
    # 정규화
    if filtered_boxes:
        total_conf = sum(box[4] for box in filtered_boxes)
        if total_conf > 0:
            normalized_boxes = []
            for r1, c1, r2, c2, conf in filtered_boxes:
                normalized_conf = conf / total_conf
                normalized_boxes.append((r1, c1, r2, c2, normalized_conf))
            return normalized_boxes
        else:
            # 모든 신뢰도가 0이면 균등 분포
            uniform_prob = 1.0 / len(filtered_boxes)
            return [(r1, c1, r2, c2, uniform_prob) for r1, c1, r2, c2, _ in filtered_boxes]
    
    return []

def validate_box_coordinates(r1: int, c1: int, r2: int, c2: int, board_shape: Tuple[int, int] = (10, 17)) -> bool:
    """
    박스 좌표 유효성 검증
    
    Args:
        r1, c1, r2, c2: 박스 좌표
        board_shape: 보드 크기 (height, width)
        
    Returns:
        유효한 좌표인지 여부
    """
    H, W = board_shape
    
    # 보드 경계 내부 확인
    if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
        return False
    
    # 좌표 순서 확인
    if not (r1 <= r2 and c1 <= c2):
        return False
    
    # 최소 면적 조건 확인 (2칸 이상)
    area = (r2 - r1 + 1) * (c2 - c1 + 1)
    if area < 2:
        return False
    
    return True

class BoxDetectionLoss(nn.Module):
    """
    YOLO 스타일 박스 검출을 위한 손실 함수 (상대 오프셋 기반)
    정확한 좌표 매칭 기반으로 신뢰도 손실 계산
    네트워크 출력: (delta_r, delta_c, confidence)
    타겟: 절대 좌표 (r1, c1, r2, c2) -> 상대 오프셋으로 자동 변환
    """
    
    def __init__(self, missing_penalty: float = 1.0, false_positive_penalty: float = 1.0):
        super(BoxDetectionLoss, self).__init__()
        self.missing_penalty = missing_penalty
        self.false_positive_penalty = false_positive_penalty
    
    def forward(self, policy_output: torch.Tensor, policy_targets: List[List[Tuple[int, int, int, int, float]]]) -> torch.Tensor:
        """
        유효도/선호도 분리 손실 계산
        
        Args:
            policy_output: (batch_size, 12, H, W) 신경망 출력 (delta_r, delta_c, validity, preference)
            policy_targets: List[List[Tuple[r1, c1, r2, c2, target_probability]]] AlphaZero 정책 타겟
            
        Returns:
            total_loss: 전체 손실값 (validity_loss + preference_loss)
        """
        batch_size, _, H, W = policy_output.shape
        validity_loss = torch.tensor(0.0, device=policy_output.device, requires_grad=True)
        preference_loss = torch.tensor(0.0, device=policy_output.device, requires_grad=True)
        coord_loss = torch.tensor(0.0, device=policy_output.device, requires_grad=True)
        
        for batch_idx in range(min(batch_size, len(policy_targets))):
            true_boxes = policy_targets[batch_idx]  # [(r1,c1,r2,c2,target_prob), ...]
            
            # 모든 위치와 앵커에서 예측 처리
            for r in range(H):
                for c in range(W):
                    for anchor in range(3):
                        # 예측된 박스 계산
                        delta_r_raw = policy_output[batch_idx, anchor*4, r, c]
                        delta_c_raw = policy_output[batch_idx, anchor*4+1, r, c] 
                        validity_raw = policy_output[batch_idx, anchor*4+2, r, c]
                        preference_raw = policy_output[batch_idx, anchor*4+3, r, c]
                        
                        # 연속값으로 예측 좌표 계산
                        delta_r_sigmoid = torch.sigmoid(delta_r_raw)
                        delta_c_sigmoid = torch.sigmoid(delta_c_raw)
                        delta_r_scaled = delta_r_sigmoid * 9  # max_delta_r
                        delta_c_scaled = delta_c_sigmoid * 16  # max_delta_c
                        
                        pred_r2 = torch.clamp(r + delta_r_scaled, 0, H-1)
                        pred_c2 = torch.clamp(c + delta_c_scaled, 0, W-1)
                        pred_validity = torch.sigmoid(validity_raw)
                        pred_preference = torch.sigmoid(preference_raw)
                        
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
                            # 매칭됨: 유효도=1.0, 선호도=target_prob로 학습
                            tr1, tc1, tr2, tc2, target_prob = matched_target
                            
                            # 유효도 손실 (이진 분류)
                            validity_target = torch.tensor(1.0, device=policy_output.device)
                            validity_loss = validity_loss + F.binary_cross_entropy(pred_validity, validity_target)
                            
                            # 선호도 손실 (회귀)
                            preference_target = torch.tensor(target_prob, device=policy_output.device, dtype=torch.float32)
                            preference_loss = preference_loss + ((pred_preference - preference_target) ** 2)
                            
                            # 좌표 손실 (연속값 회귀)
                            coord_loss = coord_loss + (torch.abs(pred_r2 - torch.tensor(tr2, device=policy_output.device, dtype=torch.float32)) + 
                                                     torch.abs(pred_c2 - torch.tensor(tc2, device=policy_output.device, dtype=torch.float32)))
                        else:
                            # 매칭 안됨: 유효도=0.0으로 학습 (선호도는 학습 안함)
                            validity_target = torch.tensor(0.0, device=policy_output.device)
                            validity_loss = validity_loss + F.binary_cross_entropy(pred_validity, validity_target)
        
        # 전체 손실 합계
        total_loss = validity_loss + preference_loss + coord_loss
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
        
        # 정책 헤드 (12채널 유효도/선호도 분리)
        # 3개 앵커 × (delta_r, delta_c, validity, preference) = 12채널
        # delta_r, delta_c: 현재 위치에서 우측/하측으로 얼마나 이동할지 (상대 오프셋)
        # validity: 유효한 수인지 판단 (0~1)
        # preference: 전략적 선호도 (0~1)
        self.policy_conv = nn.Conv2d(hidden_channels, 12, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(12)
        
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
        
        # 정책 헤드 (유효도/선호도 분리)
        policy_output = self.policy_bn(self.policy_conv(x))  # (batch, 12, H, W)
        
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
            
            # 유효한 움직임만 필터링하고 정규화
            filtered_boxes = filter_valid_predictions(policy_output, valid_moves, confidence_threshold=0.1)
            
            # valid_moves 순서대로 확률 분포 생성
            move_probs = []
            for move in valid_moves:
                if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
                    # filtered_boxes에서 해당 move의 확률 찾기
                    prob = 0.0
                    for box in filtered_boxes:
                        if len(box) >= 5 and box[0] == move[0] and box[1] == move[1] and box[2] == move[2] and box[3] == move[3]:
                            prob = box[4]
                            break
                    move_probs.append(prob)
                else:
                    # 패스 움직임
                    move_probs.append(0.0)
            
            # 추가 정규화 (필터링 후에도 합이 1이 아닐 수 있음)
            total = sum(move_probs)
            if total > 0:
                move_probs = [p / total for p in move_probs]
            else:
                # 모든 확률이 0이면 균등 분포
                move_probs = [1.0 / len(valid_moves)] * len(valid_moves)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return move_probs, value
    

class AlphaZeroTrainer:
    def __init__(self, model: AlphaZeroNet, lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.device = model.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_loss_fn = nn.MSELoss()
        self.box_detection_loss_fn = BoxDetectionLoss()
        
    def train_step(self, states: np.ndarray, policy_targets: List[List[Tuple[int, int, int, int, float]]], 
                   value_targets: np.ndarray) -> dict:
        """
        AlphaZero 정책 네트워크 훈련 스텝 (상대 오프셋 기반)
        Args:
            states: (batch_size, 2, 10, 17)
            policy_targets: List[List[Tuple[r1, c1, r2, c2, confidence]]] 배치별 정책 타겟
                           내부적으로 상대 오프셋 (delta_r, delta_c)으로 변환됨
            value_targets: (batch_size,) 게임 결과
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 텐서 변환 및 GPU로 이전
        states_tensor = torch.FloatTensor(states).to(self.device)
        value_targets_tensor = torch.FloatTensor(value_targets).unsqueeze(1).to(self.device)
        
        # 신경망 예측 (GPU에서 실행)
        policy_output, value_pred = self.model(states_tensor)  # (batch, 9, H, W), (batch, 1)
        
        # AlphaZero 정책 네트워크 손실 계산
        policy_loss = self.box_detection_loss_fn(policy_output, policy_targets)
        
        # 가치 손실 계산
        value_loss = self.value_loss_fn(value_pred, value_targets_tensor)
        
        # 총 손실
        total_loss = policy_loss + value_loss
        
        # 역전파
        total_loss.backward()
        self.optimizer.step()
        
        # GPU 메모리 사용량 체크 (훈련 후)
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9  # GB 단위
            torch.cuda.empty_cache()  # 메모리 정리
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'gpu_memory_gb': gpu_memory_used
        }
    
    def save_model(self, filepath: str):
        """모델 저장 (가중치만)"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드 (Optimizer 디바이스 이동 포함)"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # 모델 상태 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Optimizer 상태 로드
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 중요: Optimizer state를 GPU로 이동
        if torch.cuda.is_available() and self.device.type == 'cuda':
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        # 검증 정보 수집
        current_params = sum(p.numel() for p in self.model.parameters())
        saved_params = checkpoint.get('model_parameters', 0)
        
        # Optimizer state 검증
        optimizer_step_count = 0
        optimizer_has_state = False
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    optimizer_has_state = True
                    state = self.optimizer.state[param]
                    if 'step' in state:
                        optimizer_step_count = max(optimizer_step_count, state['step'])
        
        return {
            'success': True,
            'parameters_match': current_params == saved_params,
            'current_parameters': current_params,
            'saved_parameters': saved_params,
            'optimizer_has_state': optimizer_has_state,
            'optimizer_step_count': optimizer_step_count,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
    
    def verify_model_functionality(self):
        """YOLO 스타일 모델 기능 검증"""
        try:
            # 테스트 입력 생성 (2, 10, 17)
            test_input = torch.randn(1, 2, 10, 17).to(self.device)
            
            with torch.no_grad():
                policy_output, value = self.model(test_input)
                
            # YOLO 출력 형태 검증
            expected_policy_shape = (1, 9, self.model.board_height, self.model.board_width)
            policy_shape_ok = policy_output.shape == expected_policy_shape
            value_shape_ok = value.shape == (1, 1)
            
            # 값 범위 검증
            value_range_ok = -1.0 <= value.item() <= 1.0
            
            # 박스 디코딩 테스트
            try:
                decoded_boxes = decode_all_predictions(policy_output, confidence_threshold=0.1)
                decode_ok = len(decoded_boxes) == 1  # 배치 크기가 1이므로
                box_count = len(decoded_boxes[0]) if decoded_boxes else 0
            except Exception:
                decode_ok = False
                box_count = 0
            
            return {
                'functional': True,
                'policy_shape_ok': policy_shape_ok,
                'value_shape_ok': value_shape_ok,
                'value_range_ok': value_range_ok,
                'decode_ok': decode_ok,
                'decoded_box_count': box_count,
                'all_checks_passed': policy_shape_ok and value_shape_ok and value_range_ok and decode_ok
            }
            
        except Exception as e:
            return {
                'functional': False,
                'error': str(e),
                'all_checks_passed': False
            }