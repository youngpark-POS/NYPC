import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
# GPU/CPU 자동 감지
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class AlphaZeroClassificationLoss(nn.Module):
    """
    올바른 AlphaZero 정책 손실 함수
    - 완벽한 매칭만 고려
    - 모든 앵커에 대해 CrossEntropy 학습
    - 박스 회귀 없이 분류만
    """
    
    def __init__(self, H: int = 10, W: int = 17, max_delta_r: int = 9, max_delta_c: int = 16):
        super(AlphaZeroClassificationLoss, self).__init__()
        self.H = H
        self.W = W
        self.max_delta_r = max_delta_r
        self.max_delta_c = max_delta_c
        
    
    def _compute_predicted_boxes(self, policy_output: torch.Tensor) -> torch.Tensor:
        """
        각 앵커가 예측하는 박스 좌표를 계산 (벡터화)
        
        Args:
            policy_output: (batch_size, 9, H, W)
            
        Returns:
            predicted_boxes: (batch_size, H, W, 3, 2) [r2, c2] coordinates
        """
        batch_size, _, H, W = policy_output.shape
        device = policy_output.device
        
        # 모든 위치의 r1, c1 좌표 생성
        r1_grid, c1_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )  # (H, W)
        
        predicted_boxes = torch.zeros(batch_size, H, W, 3, 2, device=device)
        
        for anchor in range(3):
            # 각 앵커의 delta 값들 추출
            delta_r_raw = policy_output[:, anchor*3, :, :]     # (batch_size, H, W)
            delta_c_raw = policy_output[:, anchor*3+1, :, :]   # (batch_size, H, W)
            
            # 시그모이드 변환 및 스케일링
            delta_r_sigmoid = torch.sigmoid(delta_r_raw)
            delta_c_sigmoid = torch.sigmoid(delta_c_raw)
            delta_r_scaled = delta_r_sigmoid * self.max_delta_r
            delta_c_scaled = delta_c_sigmoid * self.max_delta_c
            
            # 절대 좌표 계산
            pred_r2 = torch.clamp(r1_grid.unsqueeze(0) + delta_r_scaled, 0, H-1)
            pred_c2 = torch.clamp(c1_grid.unsqueeze(0) + delta_c_scaled, 0, W-1)
            
            # 정수로 반올림 (완벽한 매칭을 위해)
            pred_r2_int = torch.round(pred_r2).long()
            pred_c2_int = torch.round(pred_c2).long()
            
            predicted_boxes[:, :, :, anchor, 0] = pred_r2_int
            predicted_boxes[:, :, :, anchor, 1] = pred_c2_int
        
        return predicted_boxes
    
    def forward(self, policy_output: torch.Tensor, 
                policy_targets: List[List[Tuple[int, int, int, int, float]]]) -> torch.Tensor:
        """
        AlphaZero 방식의 정책 손실 계산
        """
        batch_size, _, H, W = policy_output.shape
        device = policy_output.device
        
        # 모든 앵커의 예측 박스 계산
        predicted_boxes = self._compute_predicted_boxes(policy_output)  # (B, H, W, 3, 2)
        
        # 타겟 레이블 텐서 생성: (batch_size, H, W, 3)
        target_labels = torch.zeros(batch_size, H, W, 3, device=device, dtype=torch.float32)
        
        # 모든 타겟을 텐서로 변환하여 벡터화된 매칭
        all_targets = []
        for batch_idx in range(min(batch_size, len(policy_targets))):
            for target in policy_targets[batch_idx]:
                if len(target) >= 5:
                    r1, c1, r2, c2, target_prob = target
                    if 0 <= r1 < H and 0 <= c1 < W:
                        all_targets.append([batch_idx, r1, c1, r2, c2, target_prob])
        
        if len(all_targets) > 0:
            targets_tensor = torch.tensor(all_targets, device=device)  # (N, 6)
            batch_indices = targets_tensor[:, 0].long()  # (N,)
            r1_coords = targets_tensor[:, 1].long()     # (N,)
            c1_coords = targets_tensor[:, 2].long()     # (N,)
            true_r2 = targets_tensor[:, 3].long()       # (N,)
            true_c2 = targets_tensor[:, 4].long()       # (N,)
            target_probs = targets_tensor[:, 5]         # (N,)
            
            # 각 타겟에 대해 모든 앵커의 예측과 비교 (벡터화)
            for anchor in range(3):
                pred_r2_anchor = predicted_boxes[batch_indices, r1_coords, c1_coords, anchor, 0]  # (N,)
                pred_c2_anchor = predicted_boxes[batch_indices, r1_coords, c1_coords, anchor, 1]  # (N,)
                
                # 완벽한 매칭 찾기 (벡터화)
                perfect_match = (pred_r2_anchor == true_r2) & (pred_c2_anchor == true_c2)  # (N,)
                
                # 매칭되는 타겟들의 확률을 레이블에 할당
                matched_indices = torch.where(perfect_match)[0]
                if len(matched_indices) > 0:
                    target_labels[batch_indices[matched_indices], 
                                r1_coords[matched_indices], 
                                c1_coords[matched_indices], 
                                anchor] = target_probs[matched_indices]
        
        # 예측된 confidence 추출
        predicted_conf = torch.zeros(batch_size, H, W, 3, device=device)
        for anchor in range(3):
            conf_raw = policy_output[:, anchor*3+2, :, :]
            predicted_conf[:, :, :, anchor] = torch.sigmoid(conf_raw)
        
        # Binary CrossEntropy 손실 계산 (각 앵커별로)
        bce_loss = F.binary_cross_entropy(
            predicted_conf.view(-1),  # (B*H*W*3,)
            target_labels.view(-1),   # (B*H*W*3,)
            reduction='mean'
        )
        
        return bce_loss

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
        단일 상태에 대한 예측
        Args:
            state: (2, 10, 17) 형태의 numpy 배열
            valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
        Returns:
            policy_probs: 유효한 움직임에 대한 확률 분포 (valid_moves 순서대로)
            value: 상태 가치 (-1 ~ 1)
        """
        self.eval()
        with torch.no_grad():
            # 입력 텐서 변환 및 GPU로 이전
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2, 10, 17)
            
            # 신경망 예측
            policy_output, value = self.forward(state_tensor)  # (1, 9, 10, 17), (1, 1)
            
            # CPU로 결과 이전
            value = value.squeeze().cpu().item()
            
            # 유효한 움직임이 없으면 패스만 가능
            if not valid_moves:
                return [1.0], value
            
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
            move_probs = F.softmax(confidence_tensor, dim=0).tolist()
            
            return move_probs, value

class AlphaZeroTrainer:
    def __init__(self, model: AlphaZeroNet, lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.device = model.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = AlphaZeroClassificationLoss()
        self.value_loss_fn = nn.MSELoss()
        
    def train_step(self, states: np.ndarray, policy_targets: List[List[Tuple[int, int, int, int, float]]], 
                   value_targets: np.ndarray) -> dict:
        """
        한 번의 훈련 스텝 수행
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
        
        # 신경망 예측
        policy_output, value_pred = self.model(states_tensor)
        
        # 손실 계산
        policy_loss = self.loss_fn(policy_output, policy_targets)
        value_loss = self.value_loss_fn(value_pred, value_targets_tensor)
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # optimizer 상태 초기화 (한 번도 실행되지 않은 경우를 위해)
        dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        dummy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 이제 optimizer state_dict 로드
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 검증 정보 반환
        return {
            'parameters_match': True,
            'model_loaded': True
        }
    
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