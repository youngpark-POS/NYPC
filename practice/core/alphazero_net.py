#!/usr/bin/env python3
"""
AlphaZero Neural Network for NYPC Mushroom Game

Unified neural network combining policy and value prediction in AlphaZero style.
Single network with shared ResNet backbone and separate policy/value heads.
Uses fixed action space with masking for policy prediction.
"""

import numpy as np
from typing import Tuple, List, Dict

try:
    from .action_space import get_action_space, mask_logits
except ImportError:
    # Handle both relative and absolute imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from action_space import get_action_space, mask_logits

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResidualBlock(nn.Module):
    """Residual block with conv-bn-relu-conv-bn structure."""
    
    def __init__(self, channels: int = 64):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out = out + residual
        out = F.relu(out)
        
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style neural network with convolutional architecture.
    
    Architecture:
    - 2-channel input (10x17): mushroom values + territory map
    - Shared ConvNet backbone with residual blocks
    - Policy head for move probability prediction  
    - Value head for position evaluation
    """
    
    def __init__(self, input_channels: int = 2, num_blocks: int = 2, filters: int = 64):
        super(AlphaZeroNet, self).__init__()
        
        # Input dimensions
        self.input_channels = input_channels
        self.board_height = 10
        self.board_width = 17
        self.filters = filters
        
        # Get action space
        self.action_space = get_action_space()
        self.num_actions = self.action_space.num_actions
        
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * self.board_height * self.board_width, self.num_actions)
        
        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(self.board_height * self.board_width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    @property
    def device(self):
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def forward(self, board_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value prediction.
        
        Args:
            board_features: (batch_size, 2, 10, 17) board features
            
        Returns:
            Tuple of (policy_logits, value_predictions)
            - policy_logits: (batch_size, num_actions) raw logits for all actions
            - value_predictions: (batch_size, 1)
        """
        # Ensure input is on the same device as model
        x = board_features.to(self.device)
        
        # Initial convolution
        x = self.initial_conv(x)  # (batch_size, filters, 10, 17)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy_conv_out = self.policy_conv(x)  # (batch_size, 2, 10, 17)
        policy_flat = policy_conv_out.flatten(1)  # (batch_size, 2*10*17)
        policy_logits = self.policy_fc(policy_flat)  # (batch_size, num_actions)
        
        # Value head
        value_conv_out = self.value_conv(x)  # (batch_size, 1, 10, 17)
        value_flat = value_conv_out.flatten(1)  # (batch_size, 1*10*17)
        value_predictions = self.value_fc(value_flat)  # (batch_size, 1)
        
        return policy_logits, value_predictions
    
    def predict_policy_value(self, board_features: np.ndarray, valid_moves: List) -> Tuple[Dict, float]:
        """
        Predict both policy and value for a single board state.
        
        Args:
            board_features: (10, 17, 2) numpy array
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
            
            policy_logits, value_pred = self.forward(board_tensor)
            
            # Apply masking to get valid probabilities
            policy_logits_np = policy_logits[0].cpu().numpy()  # (num_actions,)
            masked_logits = mask_logits(policy_logits_np, valid_moves)
            
            # Convert to probabilities
            masked_logits_tensor = torch.from_numpy(masked_logits).to(self.device)
            probs = F.softmax(masked_logits_tensor, dim=0).cpu().numpy()
            
            # Create move probabilities dictionary
            move_probs = {}
            for move in valid_moves:
                action_idx = self.action_space.action_to_index_map(move)
                if action_idx >= 0:
                    move_probs[move] = float(probs[action_idx])
            
            # Normalize to ensure probabilities sum to 1
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                move_probs = {move: prob/total_prob for move, prob in move_probs.items()}
            
            win_prob = float(value_pred.item())
            
            return move_probs, win_prob
    
    def predict_move(self, board_features: np.ndarray, valid_moves: List) -> Tuple[int, int, int, int]:
        """
        Predict the best move for a single board state.
        
        Args:
            board_features: (10, 17, 2) numpy array
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


def create_alphazero_net(device: str = 'cpu', input_channels: int = 2, 
                        num_blocks: int = 2, filters: int = 64) -> AlphaZeroNet:
    """
    Create and initialize an AlphaZero convolutional neural network.
    
    Args:
        device: Device to place the model on ('cpu' or 'cuda')
        input_channels: Number of input channels (default 2 for board features)
        num_blocks: Number of residual blocks (default 2)
        filters: Number of filters in conv layers (default 64)
        
    Returns:
        Initialized AlphaZeroNet on specified device
    """
    model = AlphaZeroNet(
        input_channels=input_channels,
        num_blocks=num_blocks,
        filters=filters
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
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
        board_features = torch.randn(batch_size, 2, 10, 17)
        valid_moves = [(0, 0, 0, 1), (1, 1, 1, 2), (-1, -1, -1, -1)]  # Two rects + pass
        
        # Forward pass
        policy_logits, value_preds = model(board_features)
        print(f"✓ Forward pass successful:")
        print(f"   Policy logits: {policy_logits.shape}")
        print(f"   Value predictions: {value_preds.shape}, range: [{value_preds.min():.3f}, {value_preds.max():.3f}]")
        
        # Test single prediction
        single_features = torch.randn(10, 17, 2).numpy()
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