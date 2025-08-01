#!/usr/bin/env python3
"""
Action Space Definition for NYPC Mushroom Game

Defines a fixed action space for all possible rectangles on a 10x17 board
with minimum area of 2 cells, plus pass action. This enables masking-based
policy networks instead of dynamic action scoring.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
import pickle
import os

# Board dimensions
BOARD_HEIGHT = 10
BOARD_WIDTH = 17
MIN_AREA = 2

class ActionSpace:
    """
    Fixed action space for the mushroom game.
    
    Defines all possible rectangle actions (r1, c1, r2, c2) where:
    - 0 <= r1 <= r2 < BOARD_HEIGHT
    - 0 <= c1 <= c2 < BOARD_WIDTH  
    - (r2-r1+1) * (c2-c1+1) >= MIN_AREA
    - Plus one pass action (-1, -1, -1, -1)
    """
    
    def __init__(self):
        self.actions = self._generate_all_actions()
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.index_to_action = {idx: action for idx, action in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        
        print(f"ActionSpace initialized:")
        print(f"  Total actions: {self.num_actions}")
        print(f"  Rectangle actions: {self.num_actions - 1}")
        print(f"  Pass action: 1")
    
    def _generate_all_actions(self) -> List[Tuple[int, int, int, int]]:
        """Generate all possible rectangle actions plus pass action."""
        actions = []
        
        # Generate all possible rectangles
        for r1 in range(BOARD_HEIGHT):
            for r2 in range(r1, BOARD_HEIGHT):
                for c1 in range(BOARD_WIDTH):
                    for c2 in range(c1, BOARD_WIDTH):
                        height = r2 - r1 + 1
                        width = c2 - c1 + 1
                        area = height * width
                        
                        if area >= MIN_AREA:
                            actions.append((r1, c1, r2, c2))
        
        # Add pass action at the end
        actions.append((-1, -1, -1, -1))
        
        return actions
    
    def action_to_index_map(self, action: Tuple[int, int, int, int]) -> int:
        """Convert action tuple to index."""
        return self.action_to_index.get(action, -1)
    
    def index_to_action_map(self, index: int) -> Tuple[int, int, int, int]:
        """Convert index to action tuple."""
        return self.index_to_action.get(index, (-1, -1, -1, -1))
    
    def create_action_mask(self, valid_actions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a boolean mask for valid actions.
        
        Args:
            valid_actions: List of valid action tuples
            
        Returns:
            Boolean array where True indicates valid actions
        """
        mask = np.zeros(self.num_actions, dtype=bool)
        
        for action in valid_actions:
            idx = self.action_to_index_map(action)
            if idx >= 0:
                mask[idx] = True
        
        return mask
    
    def mask_logits(self, logits: np.ndarray, valid_actions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Apply masking to logits by setting invalid actions to -inf.
        
        Args:
            logits: Raw policy logits (shape: [num_actions])
            valid_actions: List of valid action tuples
            
        Returns:
            Masked logits ready for softmax
        """
        masked_logits = logits.copy()
        mask = self.create_action_mask(valid_actions)
        
        # Set invalid actions to -inf
        masked_logits[~mask] = float('-inf')
        
        return masked_logits
    
    def valid_actions_to_policy_target(self, valid_actions: List[Tuple[int, int, int, int]], 
                                     action_probs: np.ndarray) -> np.ndarray:
        """
        Convert valid actions and their probabilities to full policy target vector.
        
        Args:
            valid_actions: List of valid action tuples
            action_probs: Probabilities for each valid action (same order)
            
        Returns:
            Full policy target vector of size [num_actions]
        """
        policy_target = np.zeros(self.num_actions)
        
        for action, prob in zip(valid_actions, action_probs):
            idx = self.action_to_index_map(action)
            if idx >= 0:
                policy_target[idx] = prob
        
        return policy_target
    
    def get_action_features(self, action: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Get feature representation for an action.
        
        Args:
            action: Action tuple (r1, c1, r2, c2)
            
        Returns:
            Feature vector for the action
        """
        r1, c1, r2, c2 = action
        
        if r1 == -1:  # Pass action
            return np.zeros(8)
        
        # Rectangle features
        width = c2 - c1 + 1
        height = r2 - r1 + 1
        area = width * height
        center_r = (r1 + r2) / 2.0 / BOARD_HEIGHT
        center_c = (c1 + c2) / 2.0 / BOARD_WIDTH
        
        features = np.array([
            width / BOARD_WIDTH,      # Normalized width
            height / BOARD_HEIGHT,    # Normalized height
            area / (BOARD_WIDTH * BOARD_HEIGHT),  # Normalized area
            center_r,                 # Normalized center row
            center_c,                 # Normalized center column
            r1 / BOARD_HEIGHT,        # Normalized top-left row
            c1 / BOARD_WIDTH,         # Normalized top-left column
            1.0                       # Rectangle indicator (1.0 for rect, 0.0 for pass)
        ])
        
        return features
    
    def get_all_action_features(self) -> np.ndarray:
        """
        Get feature matrix for all actions.
        
        Returns:
            Feature matrix of shape [num_actions, 8]
        """
        features = np.zeros((self.num_actions, 8))
        
        for idx, action in enumerate(self.actions):
            features[idx] = self.get_action_features(action)
        
        return features
    
    def save(self, path: str):
        """Save action space to file."""
        data = {
            'actions': self.actions,
            'action_to_index': self.action_to_index,
            'index_to_action': self.index_to_action,
            'num_actions': self.num_actions
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str):
        """Load action space from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        action_space = cls.__new__(cls)
        action_space.actions = data['actions']
        action_space.action_to_index = data['action_to_index']
        action_space.index_to_action = data['index_to_action']
        action_space.num_actions = data['num_actions']
        
        return action_space

# Global action space instance
_global_action_space = None

def get_action_space() -> ActionSpace:
    """Get the global action space instance."""
    global _global_action_space
    if _global_action_space is None:
        _global_action_space = ActionSpace()
    return _global_action_space

def action_to_index(action: Tuple[int, int, int, int]) -> int:
    """Convert action to index using global action space."""
    return get_action_space().action_to_index_map(action)

def index_to_action(index: int) -> Tuple[int, int, int, int]:
    """Convert index to action using global action space."""
    return get_action_space().index_to_action_map(index)

def create_action_mask(valid_actions: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Create action mask using global action space."""
    return get_action_space().create_action_mask(valid_actions)

def mask_logits(logits: np.ndarray, valid_actions: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Mask logits using global action space."""
    return get_action_space().mask_logits(logits, valid_actions)

def test_action_space():
    """Test the action space implementation."""
    print("Testing ActionSpace...")
    
    action_space = ActionSpace()
    
    # Test basic functionality
    print(f"Total actions: {action_space.num_actions}")
    
    # Test pass action
    pass_action = (-1, -1, -1, -1)
    pass_idx = action_space.action_to_index_map(pass_action)
    print(f"Pass action index: {pass_idx}")
    print(f"Pass action recovered: {action_space.index_to_action_map(pass_idx)}")
    
    # Test rectangle action
    rect_action = (0, 0, 0, 1)  # 1x2 rectangle
    rect_idx = action_space.action_to_index_map(rect_action)
    print(f"Rectangle action {rect_action} -> index {rect_idx}")
    print(f"Index {rect_idx} -> action {action_space.index_to_action_map(rect_idx)}")
    
    # Test masking
    valid_actions = [(0, 0, 0, 1), (1, 1, 1, 2), (-1, -1, -1, -1)]
    mask = action_space.create_action_mask(valid_actions)
    print(f"Valid actions: {len(valid_actions)}")
    print(f"Mask sum (should equal len(valid_actions)): {mask.sum()}")
    
    # Test feature extraction
    features = action_space.get_action_features((0, 0, 0, 1))
    print(f"Features for (0,0,0,1): {features}")
    
    # Test policy target conversion
    action_probs = np.array([0.5, 0.3, 0.2])
    policy_target = action_space.valid_actions_to_policy_target(valid_actions, action_probs)
    print(f"Policy target sum: {policy_target.sum()}")
    print(f"Policy target shape: {policy_target.shape}")
    
    print("ActionSpace test completed!")

if __name__ == "__main__":
    test_action_space()