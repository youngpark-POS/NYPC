import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# GPU/CPU 자동 감지
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                 board_height: int = 10, board_width: int = 17, action_space_size: int = None):
        super(AlphaZeroNet, self).__init__()
        
        self.device = device  # 글로벌 디바이스 사용
        self.board_height = board_height
        self.board_width = board_width
        
        # 액션 공간 크기를 GameBoard에서 가져오도록 설정
        if action_space_size is None:
            # 고정된 액션 공간 크기 사용 (최소 2칸 박스의 총 개수 + 패스)
            # 10x17 보드에서 최소 2칸 이상의 모든 사각형 조합
            total_actions = 0
            for r1 in range(board_height):
                for c1 in range(board_width):
                    for r2 in range(r1, board_height):
                        for c2 in range(c1, board_width):
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if area >= 2:
                                total_actions += 1
            self.action_space_size = total_actions + 1  # +1 for pass
        else:
            self.action_space_size = action_space_size
        
        # 백본 네트워크 (입력층 + 2개 잔차블록)
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        
        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)
        
        # 정책 헤드 (2채널 컨볼루션 + FC)
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_height * board_width, self.action_space_size)
        
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
    
    def predict(self, state: np.ndarray, valid_moves: list, game_board=None) -> tuple:
        """
        단일 상태에 대한 예측 (인덱스 기반 마스킹)
        Args:
            state: (2, 10, 17) 형태의 numpy 배열
            valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
            game_board: GameBoard 인스턴스 (액션 매핑용)
        Returns:
            policy_probs: 유효한 움직임에 대한 확률 분포 (valid_moves 순서대로)
            value: 상태 가치 (-1 ~ 1)
        """
        self.eval()
        with torch.no_grad():
            # 입력 텐서 변환 및 GPU로 이전
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2, 10, 17)
            
            # 신경망 예측 (8246 전체 액션 공간) - GPU에서 실행
            policy_logits, value = self.forward(state_tensor)
            
            # CPU로 결과 이전
            policy_logits = policy_logits.squeeze(0).cpu()  # (8246,)
            value = value.squeeze().cpu().item()
            
            # 유효한 움직임이 없으면 패스만 가능
            if not valid_moves:
                return [1.0], value
            
            # 유효한 액션들을 8246 공간의 인덱스로 매핑
            valid_indices = []
            for move in valid_moves:
                if game_board is not None:
                    action_idx = game_board.encode_move(*move)
                    if action_idx is not None:
                        valid_indices.append(action_idx)
            
            if not valid_indices:
                # 인덱스 매핑에 실패한 경우 균등 분포 반환
                return [1.0 / len(valid_moves)] * len(valid_moves), value
            
            # 유효한 인덱스에 해당하는 로짓만 추출
            valid_logits = policy_logits[valid_indices]
            
            # 소프트맥스 적용
            policy_probs = F.softmax(valid_logits, dim=0).cpu().numpy().tolist()
            
            # valid_moves 순서에 맞추어 정렬
            # 매핑된 인덱스와 valid_moves가 1:1 대응되도록 보장
            if len(policy_probs) != len(valid_moves):
                # 안전 장치: 길이가 다르면 균등 분포
                policy_probs = [1.0 / len(valid_moves)] * len(valid_moves)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return policy_probs, value
    
    def get_move_probabilities(self, state: np.ndarray, valid_moves: list, game_board=None) -> dict:
        """
        움직임별 확률을 딕셔너리로 반환 (인덱스 기반 마스킹)
        """
        policy_probs, value = self.predict(state, valid_moves, game_board)
        
        move_probs = {}
        for i, move in enumerate(valid_moves):
            if i < len(policy_probs):
                move_probs[move] = policy_probs[i]
            else:
                move_probs[move] = 0.0
        
        return move_probs, value

class AlphaZeroTrainer:
    def __init__(self, model: AlphaZeroNet, lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.device = model.device  # 모델과 같은 디바이스 사용
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
    def train_step(self, states: np.ndarray, policy_targets: np.ndarray, 
                   value_targets: np.ndarray) -> dict:
        """
        한 번의 훈련 스텝 수행 (8246 크기 고정 정책 타겟)
        Args:
            states: (batch_size, 2, 10, 17)
            policy_targets: (batch_size, 8246) 8246 크기 고정 정책 타겟
            value_targets: (batch_size,) 게임 결과
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 텐서 변환 및 GPU로 이전
        states_tensor = torch.FloatTensor(states).to(self.device)
        policy_targets_tensor = torch.FloatTensor(policy_targets).to(self.device)
        value_targets_tensor = torch.FloatTensor(value_targets).unsqueeze(1).to(self.device)
        
        # 신경망 예측 (GPU에서 실행)
        policy_logits, value_pred = self.model(states_tensor)
        
        # 정책 손실 계산 (KL Divergence - CrossEntropy 스타일)
        # 이미 8246 크기로 정규화된 타겟이므로 직접 KL Divergence 적용
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(policy_targets_tensor * log_probs, dim=1)
        
        # 유효한 타겟이 있는 샘플만 고려 (0이 아닌 확률이 있는 경우)
        valid_samples = policy_targets_tensor.sum(dim=1) > 0
        if valid_samples.sum() > 0:
            total_policy_loss = policy_loss[valid_samples].mean()
        else:
            total_policy_loss = torch.tensor(0.0, requires_grad=True)
        
        # 가치 손실 계산
        value_loss = self.value_loss_fn(value_pred, value_targets_tensor)
        
        # 총 손실
        total_loss = total_policy_loss + value_loss
        
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
            'policy_loss': total_policy_loss.item(),
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
        checkpoint = torch.load(filepath, map_location='cpu')
        
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
        """모델 기능 검증 (로드 후 확인용)"""
        try:
            # 테스트 입력 생성 (2, 10, 17)
            test_input = torch.randn(1, 2, 10, 17).to(self.device)
            
            with torch.no_grad():
                policy_logits, value = self.model(test_input)
                
            # 출력 형태 검증
            expected_policy_size = 8246  # 액션 공간 크기
            policy_ok = policy_logits.shape == (1, expected_policy_size)
            value_ok = value.shape == (1, 1)
            
            # 출력 값 범위 검증
            policy_probs = F.softmax(policy_logits, dim=1)
            prob_sum_ok = abs(policy_probs.sum().item() - 1.0) < 1e-5
            value_range_ok = -1.0 <= value.item() <= 1.0
            
            return {
                'functional': True,
                'policy_shape_ok': policy_ok,
                'value_shape_ok': value_ok,
                'probability_sum_ok': prob_sum_ok,
                'value_range_ok': value_range_ok,
                'all_checks_passed': policy_ok and value_ok and prob_sum_ok and value_range_ok
            }
            
        except Exception as e:
            return {
                'functional': False,
                'error': str(e),
                'all_checks_passed': False
            }