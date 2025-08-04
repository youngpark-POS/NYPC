#!/usr/bin/env python3
"""
데이터 증강 시스템 - 인덱스-to-인덱스 직접 매핑
GameBoard와 동일한 순서로 액션 공간을 구축하고 미리 계산된 변환 테이블 사용
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import time

class DataAugmentation:
    """인덱스 기반 데이터 증강"""
    
    def __init__(self, board_height: int = 10, board_width: int = 17):
        self.board_height = board_height
        self.board_width = board_width
        
        # 변환 타입들
        self.transform_types = [
            'original',
            'rotate_180', 
            'flip_vertical',
            'flip_horizontal'
        ]
        
        # Build action space and transformation tables silently
        self.standard_actions = []
        self.action_to_index = {}
        self.index_to_action = {}
        
        # GameBoard와 동일한 순서로 액션 공간 구축
        self._build_standard_action_space()
        
        self.transform_map = {}
        self._build_transformation_tables()
    
    def _build_standard_action_space(self):
        """GameBoard와 동일한 순서로 표준 액션 공간 구축"""
        action_idx = 0
        
        # GameBoard 생성자와 동일한 순서
        for r1 in range(self.board_height):
            for c1 in range(self.board_width):
                for r2 in range(r1, self.board_height):
                    for c2 in range(c1, self.board_width):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:  # 최소 2칸 이상
                            action = (r1, c1, r2, c2)
                            self.standard_actions.append(action)
                            self.action_to_index[action] = action_idx
                            self.index_to_action[action_idx] = action
                            action_idx += 1
        
        # 패스 액션 (마지막에 추가)
        pass_action = (-1, -1, -1, -1)
        self.standard_actions.append(pass_action)
        self.action_to_index[pass_action] = action_idx
        self.index_to_action[action_idx] = pass_action
        
        self.action_space_size = len(self.standard_actions)
    
    def _transform_coordinates(self, r: int, c: int, transform_type: str) -> Tuple[int, int]:
        """좌표 변환"""
        if transform_type == 'original':
            return r, c
        elif transform_type == 'rotate_180':
            return self.board_height - 1 - r, self.board_width - 1 - c
        elif transform_type == 'flip_vertical':
            return self.board_height - 1 - r, c
        elif transform_type == 'flip_horizontal':
            return r, self.board_width - 1 - c
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def _transform_action(self, action: Tuple[int, int, int, int], transform_type: str) -> Tuple[int, int, int, int]:
        """액션 변환 (좌표 순서 정규화 포함)"""
        if transform_type == 'original':
            return action
        
        r1, c1, r2, c2 = action
        
        # 패스 액션은 변환하지 않음
        if action == (-1, -1, -1, -1):
            return action
        
        # 각 좌표 변환
        new_r1, new_c1 = self._transform_coordinates(r1, c1, transform_type)
        new_r2, new_c2 = self._transform_coordinates(r2, c2, transform_type)
        
        # 좌표 순서 정규화 (항상 r1 <= r2, c1 <= c2)
        min_r, max_r = min(new_r1, new_r2), max(new_r1, new_r2)
        min_c, max_c = min(new_c1, new_c2), max(new_c1, new_c2)
        
        return (min_r, min_c, max_r, max_c)
    
    def _build_transformation_tables(self):
        """변환 매핑 테이블 구축"""
        for transform_type in self.transform_types:
            self.transform_map[transform_type] = np.full(self.action_space_size, -1, dtype=np.int32)
        
        # 각 액션에 대해 변환된 액션의 인덱스 계산
        for original_idx, action in enumerate(self.standard_actions):
            for transform_type in self.transform_types:
                transformed_action = self._transform_action(action, transform_type)
                
                # 변환된 액션의 인덱스 찾기
                if transformed_action in self.action_to_index:
                    transformed_idx = self.action_to_index[transformed_action]
                    self.transform_map[transform_type][original_idx] = transformed_idx
                else:
                    # 변환된 액션이 유효하지 않은 경우 (거의 발생하지 않음)
                    print(f"Warning: Transformed action {transformed_action} not found for {action}")
                    self.transform_map[transform_type][original_idx] = original_idx  # 원본으로 폴백
    
    def transform_policy_vector(self, policy_vector: np.ndarray, transform_type: str) -> np.ndarray:
        """O(1) 정책 벡터 변환 (인덱스 직접 매핑)"""
        if transform_type == 'original':
            return policy_vector.copy()
        
        if transform_type not in self.transform_map:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        new_policy = np.zeros_like(policy_vector)
        transform_indices = self.transform_map[transform_type]
        
        # 벡터화된 인덱스 매핑 (매우 빠름)
        for original_idx in range(len(policy_vector)):
            if policy_vector[original_idx] > 1e-8:  # 유의미한 확률만
                transformed_idx = transform_indices[original_idx]
                if transformed_idx >= 0:  # 유효한 매핑
                    new_policy[transformed_idx] = policy_vector[original_idx]
        
        # 정규화 (확률 합이 1이 되도록)
        total_prob = np.sum(new_policy)
        if total_prob > 0:
            new_policy = new_policy / total_prob
        
        return new_policy
    
    def augment_training_data(self, states: np.ndarray, policy_targets: np.ndarray, 
                             value_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4배 데이터 증강"""
        augmented_states = []
        augmented_policies = []
        augmented_values = []
        
        # Applying 4x data augmentation...
        start_time = time.time()
        
        for i in range(len(states)):
            state_tensor = states[i]
            policy_target = policy_targets[i]
            value_target = value_targets[i]
            
            # 각 변환 타입에 대해 증강
            for transform_type in self.transform_types:
                # State tensor 변환
                transformed_tensor = self._transform_state_tensor(state_tensor, transform_type)
                
                # Policy target 변환
                transformed_policy = self.transform_policy_vector(policy_target, transform_type)
                
                # 리스트에 추가
                augmented_states.append(transformed_tensor)
                augmented_policies.append(transformed_policy)
                augmented_values.append(value_target)  # Value는 변환과 무관
        
        # NumPy 배열로 변환
        augmented_states = np.stack(augmented_states)
        augmented_policies = np.stack(augmented_policies)
        augmented_values = np.array(augmented_values, dtype=np.float32)
        
        elapsed = time.time() - start_time
        # Data augmentation completed
        
        return augmented_states, augmented_policies, augmented_values
    
    def _transform_state_tensor(self, state_tensor: np.ndarray, transform_type: str) -> np.ndarray:
        """State tensor (2, 10, 17) 변환"""
        if transform_type == 'original':
            return state_tensor.copy()
        
        new_tensor = np.zeros_like(state_tensor)
        
        for r in range(self.board_height):
            for c in range(self.board_width):
                new_r, new_c = self._transform_coordinates(r, c, transform_type)
                new_tensor[:, new_r, new_c] = state_tensor[:, r, c]
        
        return new_tensor
    
    def verify_mapping_consistency(self, game_board) -> bool:
        """GameBoard와 표준 액션 공간의 일치성 검증"""
        print("Verifying mapping consistency with GameBoard...")
        
        # GameBoard의 액션 공간 크기 확인
        gb_action_space_size = game_board.get_action_space_size()
        if gb_action_space_size != self.action_space_size:
            print(f"ERROR: Action space size mismatch - GameBoard:{gb_action_space_size}, Standard:{self.action_space_size}")
            return False
        
        # 몇 개 액션 샘플링해서 일치성 확인
        test_actions = [
            (0, 0, 1, 1),
            (4, 8, 5, 9),
            (8, 14, 9, 16),
            (0, 0, 2, 3),
            (-1, -1, -1, -1)
        ]
        
        mismatches = 0
        for action in test_actions:
            # 표준 인덱스
            if action in self.action_to_index:
                standard_idx = self.action_to_index[action]
                
                # GameBoard 인덱스
                gb_idx = game_board.encode_move(*action)
                
                if gb_idx != standard_idx:
                    print(f"MISMATCH: {action} - Standard:{standard_idx}, GameBoard:{gb_idx}")
                    mismatches += 1
                else:
                    print(f"OK: {action} - Index:{standard_idx}")
            else:
                print(f"WARNING: {action} not in standard action space")
        
        if mismatches == 0:
            print("[OK] Mapping consistency verified!")
            return True
        else:
            print(f"[ERROR] Found {mismatches} mismatches!")
            return False
    
    def get_transform_statistics(self) -> Dict:
        """변환 통계 정보 반환"""
        stats = {
            'action_space_size': self.action_space_size,
            'transform_types': len(self.transform_types),
            'table_memory_mb': (4 * self.action_space_size * 4) / (1024 * 1024),  # 4바이트 * 4변환
        }
        
        # 각 변환별 유효 매핑 개수
        for transform_type in self.transform_types:
            valid_mappings = np.sum(self.transform_map[transform_type] >= 0)
            stats[f'{transform_type}_valid_mappings'] = valid_mappings
        
        return stats

if __name__ == "__main__":
    # 간단한 테스트
    print("Testing DataAugmentation...")
    
    aug = DataAugmentation()
    
    # 통계 출력
    stats = aug.get_transform_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # GameBoard와 일치성 확인
    try:
        from game_board import GameBoard
        test_board = [[1] * 17 for _ in range(10)]
        game_board = GameBoard(test_board)
        aug.verify_mapping_consistency(game_board)
    except ImportError:
        print("GameBoard not available for consistency check")
    
    print("\nDataAugmentation test completed!")