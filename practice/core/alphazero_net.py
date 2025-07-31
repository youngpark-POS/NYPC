#!/usr/bin/env python3
"""
AlphaZero Neural Network for NYPC Mushroom Game

Unified neural network combining policy and value prediction in AlphaZero style.
Single network with shared ResNet backbone and separate policy/value heads.
"""

import numpy as np
from typing import Tuple, List, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResNetBlock(nn.Module):
    """Basic ResNet block with skip connections."""
    
    def __init__(self, channels: int):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style neural network with unified policy-value prediction.
    
    Architecture:
    - Shared ResNet backbone for spatial feature extraction
    - Policy head for move probability prediction  
    - Value head for position evaluation
    - Automatic device management
    """
    
    def __init__(self, input_channels: int = 7, hidden_channels: int = 64, num_blocks: int = 4):
        super(AlphaZeroNet, self).__init__()
        
        # Input dimensions
        self.input_channels = input_channels
        self.board_height = 10
        self.board_width = 17
        self.hidden_channels = hidden_channels
        
        # Shared CNN backbone
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # Shared ResNet blocks
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_global_pool = nn.AdaptiveAvgPool2d(1)
        self.policy_global_fc = nn.Linear(hidden_channels, 32)
        self.policy_action_scorer = nn.Sequential(
            nn.Linear(32 + 8, 16),  # 32 global + 8 action features
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_global_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    @property
    def device(self):
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def forward(self, board_features: torch.Tensor, valid_moves: List = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value prediction.
        
        Args:
            board_features: (batch_size, 7, 10, 17) board features
            valid_moves: List of valid moves for policy scoring (optional)
            
        Returns:
            Tuple of (policy_scores, value_predictions)
            - policy_scores: (batch_size, num_valid_moves) if valid_moves provided, else None
            - value_predictions: (batch_size, 1)
        """
        batch_size = board_features.shape[0]
        
        # Ensure input is on the same device as model
        board_features = board_features.to(self.device)
        
        # Shared feature extraction
        x = F.relu(self.bn_input(self.conv_input(board_features)))
        
        # Shared ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)
        
        # Policy head
        policy_scores = None
        if valid_moves is not None:
            policy_features = self.policy_global_pool(x).flatten(1)  # (batch_size, hidden_channels)
            policy_features = F.relu(self.policy_global_fc(policy_features))  # (batch_size, 32)
            policy_features = self.dropout(policy_features)
            
            # Score each valid move
            action_scores = []
            for move in valid_moves:
                r1, c1, r2, c2 = move
                
                if r1 == -1:  # Pass move
                    action_feats = torch.zeros(batch_size, 8, device=self.device)
                else:
                    # Rectangle features
                    width = c2 - c1 + 1
                    height = r2 - r1 + 1
                    area = width * height
                    center_r = (r1 + r2) / 2.0 / 10.0
                    center_c = (c1 + c2) / 2.0 / 17.0
                    
                    action_feats = torch.tensor([
                        width / 17.0, height / 10.0, area / 170.0, center_r, center_c,
                        r1 / 10.0, c1 / 17.0, 1.0
                    ], device=self.device).unsqueeze(0).expand(batch_size, -1)
                
                # Combine and score
                combined_feats = torch.cat([policy_features, action_feats], dim=1)
                score = self.policy_action_scorer(combined_feats)
                action_scores.append(score)
            
            if action_scores:
                policy_scores = torch.cat(action_scores, dim=1)  # (batch_size, num_valid_moves)
        
        # Value head
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = self.value_global_pool(value_x).flatten(1)  # (batch_size, 32)
        value_predictions = self.value_head(value_x)  # (batch_size, 1)
        
        return policy_scores, value_predictions
    
    def predict_policy_value(self, board_features: np.ndarray, valid_moves: List) -> Tuple[Dict, float]:
        """
        Predict both policy and value for a single board state.
        
        Args:
            board_features: (10, 17, 7) numpy array
            valid_moves: List of valid moves
            
        Returns:
            Tuple of (move_probabilities, win_probability)
        """
        if not valid_moves:
            return {}, 0.0
        
        self.eval()
        with torch.no_grad():
            # Convert to tensor and ensure correct device
            board_tensor = torch.from_numpy(board_features).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            policy_scores, value_pred = self.forward(board_tensor, valid_moves)
            
            # Convert policy scores to probabilities
            if policy_scores is not None:
                probs = F.softmax(policy_scores, dim=1)[0].cpu().numpy()
                move_probs = dict(zip(valid_moves, probs))
            else:
                move_probs = {}
            
            win_prob = float(value_pred.item())
            
            return move_probs, win_prob
    
    def predict_move(self, board_features: np.ndarray, valid_moves: List) -> Tuple[int, int, int, int]:
        """
        Predict the best move for a single board state.
        
        Args:
            board_features: (10, 17, 7) numpy array
            valid_moves: List of valid moves
            
        Returns:
            Best move as (r1, c1, r2, c2) tuple
        """
        if not valid_moves:
            return (-1, -1, -1, -1)
        
        move_probs, _ = self.predict_policy_value(board_features, valid_moves)
        
        if not move_probs:
            return (-1, -1, -1, -1)
        
        # Select move with highest probability
        best_move = max(move_probs.keys(), key=lambda move: move_probs[move])
        return best_move


def create_alphazero_net(device: str = 'cpu', input_channels: int = 7, 
                        hidden_channels: int = 64, num_blocks: int = 4) -> AlphaZeroNet:
    """
    Create and initialize an AlphaZero neural network.
    
    Args:
        device: Device to place the model on ('cpu' or 'cuda')
        input_channels: Number of input channels (default 7 for board features)
        hidden_channels: Hidden channels in ResNet blocks
        num_blocks: Number of ResNet blocks
        
    Returns:
        Initialized AlphaZeroNet on specified device
    """
    model = AlphaZeroNet(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model.to(device)


def test_alphazero_net():
    """Test the AlphaZero network implementation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return False
    
    print("Testing AlphaZeroNet...")
    
    try:
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_alphazero_net(device=device)
        print(f"✓ Model created successfully on {device}")
        print(f"✓ Model device property: {model.device}")
        
        # Test with dummy data
        batch_size = 2
        board_features = torch.randn(batch_size, 7, 10, 17)
        valid_moves = [(0, 0, 0, 1), (1, 1, 1, 2), (-1, -1, -1, -1)]  # Two rects + pass
        
        # Forward pass
        policy_scores, value_preds = model(board_features, valid_moves)
        print(f"✓ Forward pass successful:")
        print(f"   Policy scores: {policy_scores.shape}")
        print(f"   Value predictions: {value_preds.shape}, range: [{value_preds.min():.3f}, {value_preds.max():.3f}]")
        
        # Test single prediction
        single_features = torch.randn(10, 17, 7).numpy()
        move_probs, win_prob = model.predict_policy_value(single_features, valid_moves)
        print(f"✓ Single prediction successful:")
        print(f"   Win probability: {win_prob:.3f}")
        print(f"   Move probabilities: {[(str(move), f'{prob:.3f}') for move, prob in list(move_probs.items())[:2]]}")
        
        # Test move prediction
        best_move = model.predict_move(single_features, valid_moves)
        print(f"✓ Move prediction: {best_move}")
        
        # Model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        print("✓ All AlphaZeroNet tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_alphazero_net()