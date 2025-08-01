import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        단일 상태에 대한 예측
        Args:
            state: (2, 10, 17) 형태의 numpy 배열
            valid_moves: 유효한 움직임 리스트 [(r1,c1,r2,c2), ...]
            game_board: GameBoard 인스턴스 (액션 인코딩용)
        Returns:
            policy_probs: 유효한 움직임에 대한 확률 분포
            value: 상태 가치 (-1 ~ 1)
        """
        self.eval()
        with torch.no_grad():
            # 입력 텐서 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, 2, 10, 17)
            
            # 신경망 예측
            policy_logits, value = self.forward(state_tensor)
            policy_logits = policy_logits.squeeze(0)  # (action_space_size,)
            value = value.squeeze().item()
            
            # 유효한 움직임에 대해서만 소프트맥스 적용
            if not valid_moves:
                # 유효한 움직임이 없으면 패스만 가능
                return [1.0], value
            
            # GameBoard의 액션 매핑을 사용하여 인덱스 찾기
            valid_indices = []
            valid_logits = []
            
            for move in valid_moves:
                if game_board is not None:
                    # GameBoard의 인코딩 사용
                    idx = game_board.encode_move(*move)
                else:
                    # 패스인 경우 마지막 인덱스
                    if move[0] == -1:
                        idx = self.action_space_size - 1
                    else:
                        # 캐시된 GameBoard 사용
                        if not hasattr(self, '_cached_game_board'):
                            from game_board import GameBoard
                            temp_board = [[1] * self.board_width for _ in range(self.board_height)]
                            self._cached_game_board = GameBoard(temp_board)
                        idx = self._cached_game_board.encode_move(*move)
                
                if idx is not None and 0 <= idx < self.action_space_size:
                    valid_indices.append(idx)
                    valid_logits.append(policy_logits[idx].item())
            
            if not valid_logits:
                # 유효한 인덱스가 없으면 균등 분포
                return [1.0 / len(valid_moves)] * len(valid_moves), value
            
            # 소프트맥스 적용
            valid_logits = torch.FloatTensor(valid_logits)
            policy_probs = F.softmax(valid_logits, dim=0).numpy().tolist()
            
            return policy_probs, value
    
    def get_move_probabilities(self, state: np.ndarray, valid_moves: list, game_board=None) -> dict:
        """
        움직임별 확률을 딕셔너리로 반환
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
    def train_step(self, states: np.ndarray, policy_targets: np.ndarray, 
                   value_targets: np.ndarray, valid_moves_list: list) -> dict:
        """
        한 번의 훈련 스텝 수행
        Args:
            states: (batch_size, 2, 10, 17)
            policy_targets: 각 상태의 유효한 움직임에 대한 MCTS 방문 분포
            value_targets: (batch_size,) 게임 결과
            valid_moves_list: 각 상태의 유효한 움직임 리스트
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 텐서 변환
        states_tensor = torch.FloatTensor(states)
        value_targets_tensor = torch.FloatTensor(value_targets).unsqueeze(1)
        
        # 신경망 예측
        policy_logits, value_pred = self.model(states_tensor)
        
        # 정책 손실 계산 (유효한 움직임에 대해서만)
        policy_losses = []
        for i, (policy_target, valid_moves) in enumerate(zip(policy_targets, valid_moves_list)):
            if len(valid_moves) == 0:
                continue
                
            # 유효한 움직임의 로짓 추출 (전역 GameBoard 사용 또는 직접 계산)
            valid_indices = []
            if hasattr(self, '_cached_game_board'):
                temp_game = self._cached_game_board
            else:
                from game_board import GameBoard
                temp_board = [[1] * self.model.board_width for _ in range(self.model.board_height)]
                temp_game = GameBoard(temp_board)
                self._cached_game_board = temp_game
            
            for move in valid_moves:
                idx = temp_game.encode_move(*move)
                if idx is not None and 0 <= idx < self.model.action_space_size:
                    valid_indices.append(idx)
            
            if not valid_indices:
                continue
                
            valid_logits = policy_logits[i][valid_indices]
            policy_target_tensor = torch.FloatTensor(policy_target[:len(valid_indices)])
            
            # 교차 엔트로피 손실
            if len(policy_target_tensor) == len(valid_logits):
                policy_loss = -torch.sum(policy_target_tensor * F.log_softmax(valid_logits, dim=0))
                policy_losses.append(policy_loss)
        
        if policy_losses:
            total_policy_loss = torch.stack(policy_losses).mean()
        else:
            total_policy_loss = torch.tensor(0.0, requires_grad=True)
        
        # 가치 손실 계산
        value_loss = self.value_loss_fn(value_pred, value_targets_tensor)
        
        # 총 손실
        total_loss = total_policy_loss + value_loss
        
        # 역전파
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': total_policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_checkpoint(self, filepath: str):
        """모델 체크포인트 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])