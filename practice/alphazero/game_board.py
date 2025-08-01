import numpy as np
from typing import List, Tuple, Optional
import copy

class GameBoard:
    def __init__(self, initial_board: List[List[int]]):
        self.R = 10
        self.C = 17
        self.original_board = np.array(initial_board, dtype=np.int32)
        self.board = self.original_board.copy()
        self.move_history = []
        
        # 액션 공간 매핑 테이블 생성
        self.action_to_move = {}  # action_idx -> (r1, c1, r2, c2)
        self.move_to_action = {}  # (r1, c1, r2, c2) -> action_idx
        self._build_action_mapping()
    
    def _build_action_mapping(self):
        """모든 유효한 사각형 조합에 대한 액션 매핑 테이블 구축"""
        action_idx = 0
        
        # 최소 2칸 이상의 모든 유효한 사각형 조합 생성
        for r1 in range(self.R):
            for c1 in range(self.C):
                for r2 in range(r1, self.R):
                    for c2 in range(c1, self.C):
                        # 최소 2칸 조건: (r2-r1+1) * (c2-c1+1) >= 2
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:
                            move = (r1, c1, r2, c2)
                            self.action_to_move[action_idx] = move
                            self.move_to_action[move] = action_idx
                            action_idx += 1
        
        # 패스 액션 추가
        pass_move = (-1, -1, -1, -1)
        self.action_to_move[action_idx] = pass_move
        self.move_to_action[pass_move] = action_idx
        self.action_space_size = action_idx + 1
        
        # print(f"Action space built: {self.action_space_size} total actions ({self.action_space_size-1} moves + 1 pass)")  # 로그 출력 비활성화
        
    def get_state_tensor(self, player: int) -> np.ndarray:
        """
        2채널 입력 데이터 생성
        채널 0: 버섯 숫자 정규화 (0~9 -> 0~0.9)
        채널 1: 영역 표시 (내진영: 1, 상대진영: -1, 빈곳: 0)
        """
        state = np.zeros((2, self.R, self.C), dtype=np.float32)
        
        # 채널 0: 정규화된 버섯 숫자
        for r in range(self.R):
            for c in range(self.C):
                if self.board[r][c] > 0:
                    state[0][r][c] = self.board[r][c] / 10.0
        
        # 채널 1: 영역 표시
        for r in range(self.R):
            for c in range(self.C):
                if self.board[r][c] == -(player + 1):
                    state[1][r][c] = 1.0  # 내 진영
                elif self.board[r][c] < 0:
                    state[1][r][c] = -1.0  # 상대 진영
        
        return state
    
    def is_valid_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """사각형이 유효한지 검사 (합이 10이고, 네 변을 모두 포함)"""
        if not (0 <= r1 <= r2 < self.R and 0 <= c1 <= c2 < self.C):
            return False
        
        total_sum = 0
        r1_fit = c1_fit = r2_fit = c2_fit = False
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] > 0:
                    total_sum += self.board[r][c]
                    if r == r1:
                        r1_fit = True
                    if r == r2:
                        r2_fit = True
                    if c == c1:
                        c1_fit = True
                    if c == c2:
                        c2_fit = True
        
        return total_sum == 10 and r1_fit and r2_fit and c1_fit and c2_fit
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """모든 유효한 움직임을 반환"""
        valid_moves = []
        
        for r1 in range(self.R):
            for c1 in range(self.C):
                for r2 in range(r1, self.R):
                    for c2 in range(c1, self.C):
                        if self.is_valid_move(r1, c1, r2, c2):
                            valid_moves.append((r1, c1, r2, c2))
        
        return valid_moves
    
    def make_move(self, r1: int, c1: int, r2: int, c2: int, player: int):
        """움직임을 실행하고 보드를 업데이트"""
        if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
            # 패스
            self.move_history.append((r1, c1, r2, c2, player))
            return
        
        if not self.is_valid_move(r1, c1, r2, c2):
            raise ValueError(f"Invalid move: ({r1}, {c1}, {r2}, {c2})")
        
        # 사각형 영역을 플레이어 영역으로 마킹
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = -(player + 1)
        
        self.move_history.append((r1, c1, r2, c2, player))
    
    def undo_move(self):
        """마지막 움직임을 되돌림"""
        if not self.move_history:
            return
        
        r1, c1, r2, c2, player = self.move_history.pop()
        
        if r1 == -1:  # 패스였다면 되돌릴 것이 없음
            return
        
        # 원래 보드 상태로 복원
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = self.original_board[r][c]
        
        # 이전 움직임들을 다시 적용
        temp_history = self.move_history.copy()
        self.board = self.original_board.copy()
        self.move_history = []
        
        for move in temp_history:
            self.make_move(*move)
    
    def is_game_over(self) -> bool:
        """게임이 끝났는지 확인 (연속 패스 또는 유효한 움직임이 없음)"""
        if len(self.move_history) >= 2:
            last_two = self.move_history[-2:]
            if all(move[0] == -1 for move in last_two):
                return True
        
        return len(self.get_valid_moves()) == 0
    
    def get_score(self, player: int) -> int:
        """플레이어의 점수 (차지한 영역 수) 반환"""
        score = 0
        for r in range(self.R):
            for c in range(self.C):
                if self.board[r][c] == -(player + 1):
                    score += 1
        return score
    
    def get_winner(self) -> Optional[int]:
        """승자 반환 (0 또는 1, 무승부시 None)"""
        if not self.is_game_over():
            return None
        
        score_0 = self.get_score(0)
        score_1 = self.get_score(1)
        
        if score_0 > score_1:
            return 0
        elif score_1 > score_0:
            return 1
        else:
            return None  # 무승부
    
    def copy(self) -> 'GameBoard':
        """게임 보드의 복사본 생성"""
        new_board = GameBoard(self.original_board.tolist())
        new_board.board = self.board.copy()
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def encode_move(self, r1: int, c1: int, r2: int, c2: int) -> Optional[int]:
        """움직임을 액션 인덱스로 인코딩"""
        move = (r1, c1, r2, c2)
        return self.move_to_action.get(move, None)
    
    def decode_move(self, action_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """액션 인덱스를 움직임으로 디코딩"""
        return self.action_to_move.get(action_idx, None)
    
    def get_action_space_size(self) -> int:
        """전체 액션 공간 크기 반환 (패스 포함)"""
        return self.action_space_size
    
    def get_all_possible_moves(self) -> List[Tuple[int, int, int, int]]:
        """모든 가능한 움직임 반환 (패스 제외)"""
        moves = []
        for action_idx in range(self.action_space_size - 1):  # 패스 제외
            move = self.decode_move(action_idx)
            if move:
                moves.append(move)
        return moves

    def __str__(self) -> str:
        """보드 상태를 문자열로 출력"""
        result = []
        for row in self.board:
            result.append(' '.join(f'{cell:2d}' for cell in row))
        return '\n'.join(result)