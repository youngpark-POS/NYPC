#!/usr/bin/env python3
"""
게임 히스토리 저장을 위한 압축된 데이터 구조
메모리 효율성을 위해 기존 GameState를 압축된 형태로 변환
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import copy

@dataclass
class CompactGameState:
    """메모리 효율적인 게임 상태 저장 클래스 (~2KB vs 기존 34KB)"""
    
    # 핵심 게임 상태 (압축됨)
    board_state: List[List[int]]  # 10x17 원본 보드 상태 (680 bytes)
    move_coords: Tuple[int, int, int, int]  # 실제 선택된 움직임 (16 bytes)
    
    # Sparse 데이터 (유효한 움직임들만)
    valid_moves: List[Tuple[int, int, int, int]]  # 유효한 움직임 리스트
    move_probabilities: List[float]  # 유효 움직임들의 확률 (valid_moves와 1:1 대응)
    
    # 메타데이터
    player: int  # 현재 플레이어 (0 or 1)
    move_number: int  # 움직임 번호
    mcts_simulations: int  # 실제 수행된 MCTS 시뮬레이션 횟수
    valid_moves_count: int  # 유효한 움직임 개수 (검증용)
    
    def __post_init__(self):
        """데이터 일관성 검증"""
        if len(self.valid_moves) != len(self.move_probabilities):
            raise ValueError(f"Valid moves ({len(self.valid_moves)}) and probabilities ({len(self.move_probabilities)}) count mismatch")
        
        if self.valid_moves_count != len(self.valid_moves):
            raise ValueError(f"Valid moves count mismatch: {self.valid_moves_count} != {len(self.valid_moves)}")
        
        # 확률 합이 1에 가까운지 확인
        prob_sum = sum(self.move_probabilities)
        if abs(prob_sum - 1.0) > 1e-3:
            print(f"Warning: Move probabilities sum to {prob_sum:.6f}, not 1.0")
    
    def get_memory_size(self) -> int:
        """대략적인 메모리 사용량 계산 (bytes)"""
        board_size = 10 * 17 * 4  # 680 bytes (int)
        moves_size = len(self.valid_moves) * 4 * 4  # 4개 좌표 * 4 bytes
        probs_size = len(self.move_probabilities) * 4  # float32
        metadata_size = 6 * 4  # 6개 int 필드
        
        return board_size + moves_size + probs_size + metadata_size

@dataclass
class CompactSelfPlayData:
    """압축된 자기대국 데이터"""
    
    compact_game_states: List[CompactGameState]  # 압축된 게임 상태들
    final_result: Dict[int, float]  # {player_id: reward}
    game_length: int  # 게임 길이
    final_score: Tuple[int, int]  # (player0_score, player1_score)
    winner: int  # 0, 1, or -1 (draw)
    
    # 게임 전체 메타데이터
    total_simulations: int  # 전체 MCTS 시뮬레이션 횟수
    average_simulations: float  # 평균 시뮬레이션 횟수
    
    def __len__(self):
        return len(self.compact_game_states)
    
    def get_memory_size(self) -> int:
        """전체 메모리 사용량 계산"""
        states_size = sum(state.get_memory_size() for state in self.compact_game_states)
        metadata_size = 32  # 기타 메타데이터
        return states_size + metadata_size

class CompactDataConverter:
    """기존 데이터와 압축 데이터 간 변환 클래스"""
    
    @staticmethod
    def from_self_play_data(original_data) -> CompactSelfPlayData:
        """기존 SelfPlayData를 CompactSelfPlayData로 변환"""
        from self_play import SelfPlayData, GameState
        
        if not isinstance(original_data, SelfPlayData):
            raise TypeError("Expected SelfPlayData instance")
        
        compact_states = []
        total_sims = 0
        
        for game_state in original_data.game_states:
            compact_state = CompactDataConverter._compress_game_state(game_state)
            compact_states.append(compact_state)
            total_sims += compact_state.mcts_simulations
        
        avg_sims = total_sims / len(compact_states) if compact_states else 0
        
        return CompactSelfPlayData(
            compact_game_states=compact_states,
            final_result=original_data.final_result.copy(),
            game_length=original_data.game_length,
            final_score=original_data.final_score,
            winner=original_data.winner,
            total_simulations=total_sims,
            average_simulations=avg_sims
        )
    
    @staticmethod
    def to_self_play_data(compact_data: CompactSelfPlayData):
        """CompactSelfPlayData를 기존 SelfPlayData로 변환"""
        from self_play import SelfPlayData, GameState
        
        game_states = []
        
        for compact_state in compact_data.compact_game_states:
            original_state = CompactDataConverter._decompress_game_state(compact_state)
            game_states.append(original_state)
        
        return SelfPlayData(
            game_states=game_states,
            final_result=compact_data.final_result.copy(),
            game_length=compact_data.game_length,
            final_score=compact_data.final_score,
            winner=compact_data.winner
        )
    
    @staticmethod
    def _compress_game_state(game_state) -> CompactGameState:
        """GameState를 CompactGameState로 압축"""
        from game_board import GameBoard
        
        # 보드 상태 추출 (state_tensor에서 원본 보드 복원)
        state_tensor = game_state.state_tensor
        board_state = CompactDataConverter._extract_board_from_tensor(state_tensor)
        
        # 정책 벡터에서 유효한 움직임과 확률 추출
        policy_target = game_state.policy_target
        valid_moves, move_probs = CompactDataConverter._extract_moves_from_policy(policy_target)
        
        # 실제 선택된 움직임 추정 (가장 높은 확률의 움직임)
        if valid_moves and move_probs:
            best_move_idx = np.argmax(move_probs)
            selected_move = valid_moves[best_move_idx]
        else:
            selected_move = (-1, -1, -1, -1)  # 패스
        
        return CompactGameState(
            board_state=board_state,
            move_coords=selected_move,
            valid_moves=valid_moves,
            move_probabilities=move_probs,
            player=game_state.player,
            move_number=game_state.move_number,
            mcts_simulations=game_state.mcts_simulations,
            valid_moves_count=len(valid_moves)
        )
    
    @staticmethod
    def _decompress_game_state(compact_state: CompactGameState):
        """CompactGameState를 GameState로 복원"""
        from self_play import GameState
        from game_board import GameBoard
        
        # 보드 상태로부터 state_tensor 생성
        state_tensor = CompactDataConverter._create_tensor_from_board(
            compact_state.board_state, compact_state.player
        )
        
        # sparse한 정책을 전체 8246 크기 벡터로 확장
        policy_vector = CompactDataConverter._expand_policy_vector(
            compact_state.valid_moves, compact_state.move_probabilities
        )
        
        return GameState(
            state_tensor=state_tensor,
            policy_target=policy_vector,
            player=compact_state.player,
            move_number=compact_state.move_number,
            mcts_simulations=compact_state.mcts_simulations,
            valid_moves_count=compact_state.valid_moves_count
        )
    
    @staticmethod
    def _extract_board_from_tensor(state_tensor: np.ndarray) -> List[List[int]]:
        """state_tensor (2, 10, 17)에서 원본 보드 상태 추출"""
        board = [[0 for _ in range(17)] for _ in range(10)]
        
        # 채널 0: 버섯 정보 (정규화된 값 복원)
        mushroom_channel = state_tensor[0]
        
        # 채널 1: 플레이어 점령 정보
        player_channel = state_tensor[1]
        
        for i in range(10):
            for j in range(17):
                if mushroom_channel[i][j] > 0:
                    # 정규화된 버섯 값을 원래 값으로 복원 (0.1-0.9 → 1-9)
                    board[i][j] = int(mushroom_channel[i][j] * 10)
                elif player_channel[i][j] == 1.0:
                    # 현재 플레이어가 점령한 칸
                    board[i][j] = -1  # 임시로 -1 (실제로는 플레이어 정보 필요)
                elif player_channel[i][j] == -1.0:
                    # 상대 플레이어가 점령한 칸
                    board[i][j] = -2  # 임시로 -2
        
        return board
    
    @staticmethod
    def _extract_moves_from_policy(policy_vector: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """정책 벡터 (8246,)에서 유효한 움직임과 확률 추출"""
        from game_board import GameBoard
        
        # 임시 보드 생성 (액션 디코딩용)
        temp_board = [[1] * 17 for _ in range(10)]
        temp_game = GameBoard(temp_board)
        
        valid_moves = []
        move_probs = []
        
        # 0이 아닌 확률을 가진 액션들만 추출
        for action_idx, prob in enumerate(policy_vector):
            if prob > 1e-6:  # 임계값 이상의 확률만
                move = temp_game.decode_action(action_idx)
                if move is not None:
                    valid_moves.append(move)
                    move_probs.append(float(prob))
        
        return valid_moves, move_probs
    
    @staticmethod
    def _create_tensor_from_board(board_state: List[List[int]], perspective_player: int) -> np.ndarray:
        """보드 상태로부터 state_tensor (2, 10, 17) 생성"""
        state = np.zeros((2, 10, 17), dtype=np.float32)
        
        for i in range(10):
            for j in range(17):
                cell = board_state[i][j]
                if cell > 0:
                    # 버섯 값을 정규화 (1-9 → 0.1-0.9)
                    state[0][i][j] = cell / 10.0
                elif cell == -(perspective_player + 1):
                    # 현재 플레이어가 점령한 칸
                    state[1][i][j] = 1.0
                elif cell == -(2 - perspective_player):
                    # 상대 플레이어가 점령한 칸
                    state[1][i][j] = -1.0
        
        return state
    
    @staticmethod
    def _expand_policy_vector(valid_moves: List[Tuple[int, int, int, int]], 
                            move_probs: List[float]) -> np.ndarray:
        """sparse한 움직임들을 전체 8246 크기 정책 벡터로 확장"""
        from game_board import GameBoard
        
        # 임시 보드 생성 (액션 인코딩용)
        temp_board = [[1] * 17 for _ in range(10)]
        temp_game = GameBoard(temp_board)
        action_space_size = temp_game.get_action_space_size()
        
        policy_vector = np.zeros(action_space_size, dtype=np.float32)
        
        for move, prob in zip(valid_moves, move_probs):
            action_idx = temp_game.encode_move(*move)
            if action_idx is not None:
                policy_vector[action_idx] = prob
        
        # 정규화 (혹시나 하는 안전장치)
        total_prob = np.sum(policy_vector)
        if total_prob > 0:
            policy_vector = policy_vector / total_prob
        
        return policy_vector

def calculate_compression_ratio(original_data, compact_data: CompactSelfPlayData) -> dict:
    """압축 비율 계산"""
    from self_play import SelfPlayData
    
    if not isinstance(original_data, SelfPlayData):
        raise TypeError("Expected SelfPlayData for comparison")
    
    # 원본 데이터 크기 추정
    original_size = 0
    for state in original_data.game_states:
        # state_tensor: (2, 10, 17) * 4 bytes
        state_tensor_size = 2 * 10 * 17 * 4
        # policy_target: (8246,) * 4 bytes  
        policy_size = 8246 * 4
        metadata_size = 20  # 기타 메타데이터
        original_size += state_tensor_size + policy_size + metadata_size
    
    # 압축된 데이터 크기
    compact_size = compact_data.get_memory_size()
    
    compression_ratio = original_size / compact_size if compact_size > 0 else 0
    space_saved = original_size - compact_size
    space_saved_percent = (space_saved / original_size * 100) if original_size > 0 else 0
    
    return {
        'original_size_bytes': original_size,
        'compact_size_bytes': compact_size,
        'compression_ratio': compression_ratio,
        'space_saved_bytes': space_saved,
        'space_saved_percent': space_saved_percent
    }

if __name__ == "__main__":
    # 간단한 테스트
    print("CompactData 모듈 테스트")
    print("=" * 50)
    
    # 더미 데이터로 테스트
    test_board = [[1, 2, 3] * 6 for _ in range(10)]  # 10x18이지만 테스트용
    test_moves = [(0, 0, 1, 1), (2, 2, 3, 3)]
    test_probs = [0.7, 0.3]
    
    compact_state = CompactGameState(
        board_state=test_board,
        move_coords=(0, 0, 1, 1),
        valid_moves=test_moves,
        move_probabilities=test_probs,
        player=0,
        move_number=1,
        mcts_simulations=400,
        valid_moves_count=len(test_moves)
    )
    
    print(f"CompactGameState 메모리 사용량: {compact_state.get_memory_size()} bytes")
    print(f"Valid moves: {compact_state.valid_moves}")
    print(f"Move probabilities: {compact_state.move_probabilities}")
    print("테스트 완료!")