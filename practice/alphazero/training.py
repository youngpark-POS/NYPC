import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import time
from neural_network import AlphaZeroNet, AlphaZeroTrainer
from self_play import SelfPlayGenerator, SelfPlayData

class AlphaZeroDataset(Dataset):
    """알파제로 훈련 데이터셋 (8246 크기 고정 정책 타겟)"""
    def __init__(self, states: np.ndarray, policy_targets: np.ndarray, value_targets: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.policy_targets = torch.FloatTensor(policy_targets)  # 8246 크기 고정
        self.value_targets = torch.FloatTensor(value_targets)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.policy_targets[idx], self.value_targets[idx])

class TrainingManager:
    """알파제로 훈련 관리 클래스"""
    
    def __init__(self, model: AlphaZeroNet, save_dir: str = "practice/models", max_history_games: int = 10000):
        self.model = model
        self.trainer = AlphaZeroTrainer(model)
        self.save_dir = save_dir
        self.training_history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'epochs': []
        }
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 게임 히스토리 관리 (항상 활성화)
        try:
            from game_history import GameHistoryManager
            history_path = os.path.join(save_dir, "game_history.h5")
            self.history_manager = GameHistoryManager(
                storage_path=history_path,
                max_games=max_history_games
            )
            print(f"Game history: {max_history_games} games max")
        except ImportError as e:
            print(f"Error: h5py package is required for game history storage.")
            print(f"Please install it with: pip install h5py>=3.7.0")
            raise ImportError(f"Missing required dependency: {e}")
        except Exception as e:
            print(f"Warning: Could not initialize game history: {e}")
            print("Falling back to temporary directory...")
            # 임시 디렉토리 사용
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="alphazero_history_")
            temp_path = os.path.join(temp_dir, "game_history.h5")
            self.history_manager = GameHistoryManager(
                storage_path=temp_path,
                max_games=max_history_games
            )
            print(f"Using temporary history storage: {temp_path}")
    
    def train_on_batch(self, states: torch.Tensor, policy_targets: List[List[float]], 
                       value_targets: torch.Tensor, valid_moves_list: List[List[Tuple[int, int, int, int]]]) -> dict:
        """배치 단위 훈련"""
        states_np = states.numpy()
        value_targets_np = value_targets.numpy()
        
        loss_info = self.trainer.train_step(states_np, policy_targets, value_targets_np, valid_moves_list)
        return loss_info
    
    
    def train_epoch(self, dataloader: DataLoader, verbose: bool = True) -> dict:
        """한 에포크 훈련 (8246 크기 고정 정책 타겟)"""
        total_losses = []
        policy_losses = []
        value_losses = []
        
        self.model.train()
        
        for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
            # 8246 크기 고정 타겟으로 훈련
            loss_info = self.trainer.train_step(states.numpy(), policy_targets.numpy(), value_targets.numpy())
            
            total_losses.append(loss_info['total_loss'])
            policy_losses.append(loss_info['policy_loss'])
            value_losses.append(loss_info['value_loss'])
            
            
        
        epoch_stats = {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
        
        return epoch_stats
    
    def train_from_self_play_data(self, game_data_list: List[SelfPlayData], 
                                 epochs: int = 10, batch_size: int = 32, verbose: bool = True, 
                                 history_mix_ratio: float = 0.0) -> dict:
        """셀프플레이 데이터로 훈련 (히스토리 지원)"""
        
        # 1. 새 게임 데이터를 히스토리에 저장 (항상 수행)
        if game_data_list:
            save_stats = self.history_manager.save_games(game_data_list)
            if verbose:
                print(f"Saved {save_stats['saved']} games to history (total: {save_stats['total_games']})")
        
        # 2. 훈련 데이터 준비 (히스토리와 새 게임 혼합)
        training_games = []
        
        if history_mix_ratio > 0.0:
            # 히스토리와 새 게임 혼합
            history_count = int(len(game_data_list) * history_mix_ratio / (1 - history_mix_ratio)) if history_mix_ratio < 1.0 else len(game_data_list) * 2
            history_games = self.history_manager.get_training_batch(history_count, mix_recent=True)
            
            if verbose:
                print(f"Using {len(game_data_list)} new games + {len(history_games)} history games")
            
            training_games = game_data_list + history_games
        else:
            # 새 게임만 사용
            training_games = game_data_list
        
        # 3. 데이터 수집 (4배 증강 적용)
        from self_play import SelfPlayGenerator
        generator = SelfPlayGenerator(self.model)
        states, policy_targets, value_targets = generator.collect_training_data_with_augmentation(training_games, use_augmentation=True)
        
        if len(states) == 0:
            print("No training data available!")
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        # 4배 증강으로 인한 배치 크기 조정
        adjusted_batch_size = max(8, batch_size // 4)  # 최소 8, 기본의 1/4
        
        if verbose:
            print(f"Training on {len(states)} samples for {epochs} epochs")
            # Batch size adjusted for 4x augmentation
        
        # 데이터셋 생성 (8246 크기 고정 policy_targets)
        dataset = AlphaZeroDataset(states, policy_targets, value_targets)
        dataloader = DataLoader(dataset, batch_size=adjusted_batch_size, shuffle=True)
        
        # 훈련 실행
        epoch_stats = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_stats = self.train_epoch(dataloader, verbose=(verbose and epoch % 5 == 0))
            
            # 훈련 기록 저장
            self.training_history['total_loss'].append(epoch_stats['total_loss'])
            self.training_history['policy_loss'].append(epoch_stats['policy_loss'])
            self.training_history['value_loss'].append(epoch_stats['value_loss'])
            self.training_history['epochs'].append(len(self.training_history['epochs']))
            
            # 매 에포크 출력
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}: Total {epoch_stats['total_loss']:.4f}, Policy {epoch_stats['policy_loss']:.4f}, Value {epoch_stats['value_loss']:.4f}")
        
        return epoch_stats
    
    def save_model(self, filename: str = "latest_model.pth"):
        """모델 저장"""
        filepath = os.path.join(self.save_dir, filename)
        self.trainer.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename: str = "latest_model.pth"):
        """모델 로드 (기본 검증만)"""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
            
        try:
            # 모델 로드 및 검증
            verification_info = self.trainer.load_model(filepath)
            
            print(f"Model loaded from {filepath}")
            
            # 중요한 에러만 출력
            if not verification_info['parameters_match']:
                print(f"  WARNING: Parameter mismatch - saved={verification_info['saved_parameters']}, current={verification_info['current_parameters']}")
            
            if not verification_info['optimizer_has_state']:
                print("  WARNING: No optimizer state - training will start fresh")
                
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model_as_binary(self, filename: str = "data.bin"):
        """바이너리 형태로 모델 저장 (대회 제출용)"""
        import pickle
        filepath = os.path.join(self.save_dir, filename)
        model_data = self.model.state_dict()
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Binary model saved to {filepath}")
    
    def evaluate_model(self, test_data: List[SelfPlayData]) -> dict:
        """모델 성능 평가"""
        self.model.eval()
        
        from self_play import SelfPlayGenerator
        generator = SelfPlayGenerator(self.model)
        states, policy_targets, value_targets = generator.collect_training_data(test_data)
        
        if len(states) == 0:
            return {'accuracy': 0.0, 'value_mae': 0.0}
        
        total_samples = len(states)
        correct_predictions = 0
        value_errors = []
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.model.device)
            value_targets_tensor = torch.FloatTensor(value_targets).to(self.model.device)
            
            _, predicted_values = self.model(states_tensor)
            predicted_values = predicted_values.squeeze()
            
            value_errors = torch.abs(predicted_values - value_targets_tensor).cpu().numpy()
            correct_predictions = total_samples // 2  # 임시값
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        value_mae = np.mean(value_errors) if len(value_errors) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'value_mae': value_mae,
            'samples': total_samples
        }
    
    def get_training_stats(self) -> dict:
        """훈련 통계 반환"""
        if not self.training_history['total_loss']:
            return {}
        
        return {
            'latest_total_loss': self.training_history['total_loss'][-1],
            'latest_policy_loss': self.training_history['policy_loss'][-1],
            'latest_value_loss': self.training_history['value_loss'][-1],
            'total_epochs': len(self.training_history['epochs']),
            'avg_total_loss': np.mean(self.training_history['total_loss']),
            'min_total_loss': np.min(self.training_history['total_loss'])
        }
    
    def get_history_stats(self) -> dict:
        """게임 히스토리 통계 반환"""
        stats = self.history_manager.get_storage_stats()
        stats['history_enabled'] = True
        return stats
    
    def save_games_to_history(self, game_data_list: List[SelfPlayData]) -> dict:
        """게임들을 히스토리에 저장"""
        return self.history_manager.save_games(game_data_list)

class LearningRateScheduler:
    """학습률 스케줄러"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float = 0.001):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, iteration: int):
        """반복 횟수에 따라 학습률 조정"""
        if iteration < 100:
            lr = self.initial_lr
        elif iteration < 300:
            lr = self.initial_lr * 0.1
        else:
            lr = self.initial_lr * 0.01
        
        if lr != self.current_lr:
            self.current_lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Learning rate updated to {lr}")
    
    def decay(self, factor: float = 0.9):
        """학습률 감소"""
        self.current_lr *= factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        print(f"Learning rate decayed to {self.current_lr}")

class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False
    
    def __call__(self, loss: float) -> bool:
        """손실이 개선되지 않으면 True 반환"""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped = True
            return True
        
        return False