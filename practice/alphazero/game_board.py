import numpy as np
from typing import List, Tuple, Optional
import copy

class GameBoard:
    """
    버섯 게임 보드 클래스
    - 10x17 보드에서 합이 10인 직사각형을 선택하는 게임
    - 플레이어 0과 1이 번갈아가며 플레이
    """
    
    def __init__(self, initial_board: List[List[int]]):
        self.R = 10  # 행 수
        self.C = 17  # 열 수
        self.board = copy.deepcopy(initial_board)
        self.current_player = 0  # 0: 첫 번째 플레이어, 1: 두 번째 플레이어
        self.pass_count = 0  # 연속 패스 횟수
        self.game_over = False
        self.winner = None
        
        # 액션 공간 매핑 테이블 생성
        self.action_to_move = {}
        self.move_to_action = {}
        self._build_action_mapping()
        
        # 유효 움직임 캐시 (성능 최적화)
        self._valid_moves_cache = {}
    
    def _build_action_mapping(self):
        """액션 인덱스와 움직임 간의 매핑 테이블 생성"""
        action_idx = 0
        
        # 모든 가능한 직사각형에 대해 매핑 생성
        for r1 in range(self.R):
            for c1 in range(self.C):
                for r2 in range(r1, self.R):
                    for c2 in range(c1, self.C):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area >= 2:  # 최소 2칸 이상
                            move = (r1, c1, r2, c2)
                            self.action_to_move[action_idx] = move
                            self.move_to_action[move] = action_idx
                            action_idx += 1
        
        # 패스 액션 추가
        self.pass_action_idx = action_idx
        self.action_to_move[action_idx] = (-1, -1, -1, -1)
        self.move_to_action[(-1, -1, -1, -1)] = action_idx
    
    def get_action_space_size(self) -> int:
        """액션 공간 크기 반환"""
        return len(self.action_to_move)
    
    def _get_board_hash(self) -> tuple:
        """보드 상태의 해시 키 생성 (캐싱용)"""
        return tuple(tuple(row) for row in self.board)
    
    
    def encode_move(self, r1: int, c1: int, r2: int, c2: int) -> Optional[int]:
        """움직임을 액션 인덱스로 변환"""
        move = (r1, c1, r2, c2)
        return self.move_to_action.get(move)
    
    def decode_action(self, action_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """액션 인덱스를 움직임으로 변환"""
        return self.action_to_move.get(action_idx)
    
    def copy(self):
        """게임 보드 복사본 생성"""
        new_board = GameBoard([[0] * self.C for _ in range(self.R)])
        new_board.board = copy.deepcopy(self.board)
        new_board.current_player = self.current_player
        new_board.pass_count = self.pass_count
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.action_to_move = self.action_to_move.copy()
        new_board.move_to_action = self.move_to_action.copy()
        
        # 캐시 정보는 공유 (성능 최적화)
        new_board._valid_moves_cache = self._valid_moves_cache
        
        return new_board
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """현재 상태에서 유효한 움직임 반환 (캐싱 적용)"""
        if self.game_over:
            return []
        
        # 캐시 체크
        board_hash = self._get_board_hash()
        if board_hash in self._valid_moves_cache:
            return self._valid_moves_cache[board_hash].copy()
        
        # 캐시 미스 - 실제 계산 수행 (조기 종료 최적화 적용)
        valid_moves = []
        
        for r1 in range(self.R):
            for c1 in range(self.C):
                skip_larger_r2 = False
                for r2 in range(r1, self.R):
                    if skip_larger_r2:
                        break
                    for c2 in range(c1, self.C):
                        # 면적 체크
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area < 2:
                            continue
                        
                        # 합계 계산
                        total_sum = self._get_box_sum(r1, c1, r2, c2)
                        
                        if total_sum >= 10:
                            if total_sum == 10 and self._check_edges(r1, c1, r2, c2):
                                valid_moves.append((r1, c1, r2, c2))
                            # 같은 r2에서 더 큰 c2들은 건너뛰기
                            break
                        
                        # 세로 한 줄(c1==c2)에서 합>=10이면 더 큰 r2들도 건너뛰기
                        if c1 == c2 and total_sum >= 10:
                            skip_larger_r2 = True
        
        # 캐시에 저장
        self._valid_moves_cache[board_hash] = valid_moves.copy()
        
        return valid_moves
    
    def _get_box_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """박스 내부 점수 합계 계산 (양수만)"""
        total_sum = 0
        for i in range(r1, r2 + 1):
            for j in range(c1, c2 + 1):
                if self.board[i][j] > 0:
                    total_sum += self.board[i][j]
        return total_sum
    
    def _check_edges(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """네 변에 각각 최소 하나 이상의 버섯이 있는지 확인"""
        top, down, left, right = False, False, False, False
        
        # 상단과 하단 변
        for j in range(c1, c2 + 1):
            if self.board[r1][j] > 0:
                top = True
            if self.board[r2][j] > 0:
                down = True
        
        # 좌측과 우측 변
        for i in range(r1, r2 + 1):
            if self.board[i][c1] > 0:
                left = True
            if self.board[i][c2] > 0:
                right = True
        
        return top and down and left and right
    
    def _is_valid_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """움직임이 유효한지 검사"""
        # 범위 체크
        if not (0 <= r1 <= r2 < self.R and 0 <= c1 <= c2 < self.C):
            return False
        
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if area < 2:  # 최소 2칸 이상
            return False
        
        # 합이 10인지 확인
        total_sum = self._get_box_sum(r1, c1, r2, c2)
        if total_sum != 10:
            return False
        
        # 네 변에 각각 최소 하나 이상의 버섯이 있는지 확인
        return self._check_edges(r1, c1, r2, c2)
    
    def make_move(self, r1: int, c1: int, r2: int, c2: int, player: int) -> bool:
        """움직임 실행"""
        if self.game_over:
            return False
        
        # 패스인 경우
        if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
            self.pass_count += 1
            if self.pass_count >= 2:
                self._end_game()
            else:
                self.current_player = 1 - self.current_player
            return True
        
        # 유효한 움직임인지 확인
        if not self._is_valid_move(r1, c1, r2, c2):
            return False
        
        # 영역 점령
        for i in range(r1, r2 + 1):
            for j in range(c1, c2 + 1):
                self.board[i][j] = -(player + 1)  # -1은 플레이어 0, -2는 플레이어 1
        
        self.pass_count = 0  # 패스 카운트 초기화
        self.current_player = 1 - self.current_player
        
        # 더 이상 유효한 움직임이 없으면 게임 종료
        if not self.get_valid_moves():
            self._end_game()
        
        return True
    
    def _end_game(self):
        """게임 종료 처리"""
        self.game_over = True
        
        # 점수 계산
        score = [0, 0]
        for row in self.board:
            for cell in row:
                if cell == -1:
                    score[0] += 1
                elif cell == -2:
                    score[1] += 1
        
        # 승자 결정
        if score[0] > score[1]:
            self.winner = 0
        elif score[1] > score[0]:
            self.winner = 1
        else:
            self.winner = -1  # 무승부
    
    def is_terminal(self) -> bool:
        """게임이 끝났는지 확인"""
        return self.game_over
    
    def get_winner(self) -> Optional[int]:
        """승자 반환 (0, 1, -1(무승부), None(게임 진행 중))"""
        if self.game_over:
            return self.winner
        return None
    
    def get_reward(self, player: int) -> float:
        """플레이어의 보상 반환"""
        if not self.game_over:
            return 0.0
        
        if self.winner == player:
            return 1.0
        elif self.winner == -1:  # 무승부
            return 0.0
        else:
            return -1.0
    
    def get_state_tensor(self, perspective_player: int) -> np.ndarray:
        """신경망 입력용 상태 텐서 생성 (2, 10, 17)"""
        state = np.zeros((2, self.R, self.C), dtype=np.float32)
        
        for i in range(self.R):
            for j in range(self.C):
                cell = self.board[i][j]
                if cell > 0:
                    # 버섯 값을 정규화 (1-5 -> 0.2-1.0)
                    state[0][i][j] = cell / 5.0
                elif cell == -(perspective_player + 1):
                    # 현재 플레이어가 점령한 칸
                    state[1][i][j] = 1.0
                elif cell == -(2 - perspective_player):
                    # 상대 플레이어가 점령한 칸
                    state[1][i][j] = -1.0
        
        return state
    
    def get_score(self) -> Tuple[int, int]:
        """현재 점수 반환 (플레이어 0 점수, 플레이어 1 점수)"""
        score = [0, 0]
        for row in self.board:
            for cell in row:
                if cell == -1:
                    score[0] += 1
                elif cell == -2:
                    score[1] += 1
        return tuple(score)
    
    def display(self) -> str:
        """보드 상태를 문자열로 표현"""
        result = []
        result.append(f"Current Player: {self.current_player}")
        result.append(f"Pass Count: {self.pass_count}")
        result.append(f"Game Over: {self.game_over}")
        if self.game_over:
            result.append(f"Winner: {self.winner}")
        
        score = self.get_score()
        result.append(f"Score - Player 0: {score[0]}, Player 1: {score[1]}")
        result.append("")
        
        # 보드 출력
        for i, row in enumerate(self.board):
            row_str = ""
            for j, cell in enumerate(row):
                if cell > 0:
                    row_str += f"{cell:2d} "
                elif cell == -1:
                    row_str += "P0 "
                elif cell == -2:
                    row_str += "P1 "
                else:
                    row_str += " . "
            result.append(f"{i:2d}: {row_str}")
        
        return "\n".join(result)
    
    def __str__(self) -> str:
        return self.display()