#!/usr/bin/env python3
"""
AB Pruning 로그를 신경망 훈련 데이터로 변환하는 컨버터
"""

import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from self_play import GameState, SelfPlayData


class SimpleGameBoard:
    """GameBoard를 대체하는 순수 Python 구현"""
    
    def __init__(self, board: List[List[int]]):
        self.board = [row[:] for row in board]  # 복사본 생성
        self.territory = [[0 for _ in range(17)] for _ in range(10)]
        self.BOARD_ROW = 10
        self.BOARD_COLUMN = 17
    
    def get_state_tensor(self, player: int) -> np.ndarray:
        """상태 텐서 생성 (기존 GameBoard와 동일한 형식)"""
        state = np.zeros((3, self.BOARD_ROW, self.BOARD_COLUMN), dtype=np.float32)
        
        # 버섯 보드 (정규화: 1-9 -> 0.1-0.9)
        for i in range(self.BOARD_ROW):
            for j in range(self.BOARD_COLUMN):
                if self.board[i][j] > 0:
                    state[0][i][j] = self.board[i][j] / 10.0
        
        # 영토 보드
        for i in range(self.BOARD_ROW):
            for j in range(self.BOARD_COLUMN):
                if self.territory[i][j] == 1:  # FIRST 플레이어 영토
                    state[1][i][j] = 1.0
                elif self.territory[i][j] == -1:  # SECOND 플레이어 영토
                    state[2][i][j] = 1.0
        
        return state
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """유효한 움직임들 반환"""
        valid_moves = []
        
        for r1 in range(self.BOARD_ROW):
            for r2 in range(r1, self.BOARD_ROW):
                for c1 in range(self.BOARD_COLUMN):
                    for c2 in range(c1, self.BOARD_COLUMN):
                        if self._is_valid_move(r1, c1, r2, c2):
                            valid_moves.append((r1, c1, r2, c2))
        
        return valid_moves
    
    def _is_valid_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """움직임이 유효한지 검사"""
        sums = 0
        r1fit, c1fit, r2fit, c2fit = False, False, False, False
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] != 0:
                    sums += self.board[r][c]
                    if sums > 10:
                        return False
                    if r == r1:
                        r1fit = True
                    if r == r2:
                        r2fit = True
                    if c == c1:
                        c1fit = True
                    if c == c2:
                        c2fit = True
        
        return sums == 10 and r1fit and r2fit and c1fit and c2fit
    
    def make_move(self, r1: int, c1: int, r2: int, c2: int, player: int):
        """움직임 실행"""
        if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
            return  # 패스
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
                self.territory[r][c] = 1 if player == 0 else -1


class ABLogConverter:
    """AB Pruning 로그를 SelfPlayData로 변환"""
    
    def __init__(self, ab_move_ratio: float = 0.85, noise_ratio: float = 0.15):
        """
        Args:
            ab_move_ratio: AB가 선택한 움직임의 확률 (기본 85%)
            noise_ratio: 다른 유효 움직임들에 분배할 확률 (기본 15%)
        """
        self.ab_move_ratio = ab_move_ratio
        self.noise_ratio = noise_ratio
    
    def parse_log_file(self, log_path: str) -> Optional[SelfPlayData]:
        """로그 파일을 파싱하여 SelfPlayData 생성"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 로그 파싱
            initial_board, game_moves, final_scores = self._parse_log_lines(lines)
            
            if not initial_board or not game_moves:
                return None
            
            # 게임 재생성 및 데이터 변환
            return self._convert_to_selfplay_data(initial_board, game_moves, final_scores)
            
        except Exception as e:
            print(f"Error parsing log file {log_path}: {e}")
            return None
    
    def convert_log_to_selfplay_data(self, game_log: List[str], initial_board: List[List[int]]) -> Optional[SelfPlayData]:
        """게임 로그를 직접 SelfPlayData로 변환"""
        try:
            # 게임 로그에서 정보 추출
            game_moves = []
            final_scores = None
            
            for line in game_log:
                line = line.strip()
                
                if line.startswith('FIRST ') or line.startswith('SECOND '):
                    parts = line.split()
                    if len(parts) >= 6:
                        player_name = parts[0]
                        player = 0 if player_name == 'FIRST' else 1
                        r1, c1, r2, c2, time_used = map(int, parts[1:6])
                        game_moves.append((player, r1, c1, r2, c2, time_used))
                
                elif line.startswith('SCOREFIRST '):
                    first_score = int(line.split()[1])
                    if final_scores is None:
                        final_scores = [first_score, 0]
                    else:
                        final_scores[0] = first_score
                
                elif line.startswith('SCORESECOND '):
                    second_score = int(line.split()[1])
                    if final_scores is None:
                        final_scores = [0, second_score]
                    else:
                        final_scores[1] = second_score
            
            if not game_moves:
                return None
            
            # 최종 점수가 없으면 기본값 설정
            if final_scores is None:
                final_scores = (0, 0)
            else:
                final_scores = tuple(final_scores)
            
            # SelfPlayData 생성
            return self._convert_to_selfplay_data(initial_board, game_moves, final_scores)
            
        except Exception as e:
            print(f"Error converting log to selfplay data: {e}")
            return None
    
    def _parse_log_lines(self, lines: List[str]) -> Tuple[Optional[List[List[int]]], List[Tuple], Optional[Tuple[int, int]]]:
        """로그 라인들을 파싱"""
        initial_board = None
        game_moves = []
        final_scores = None
        
        for line in lines:
            line = line.strip()
            
            # INIT 라인에서 초기 보드 추출
            if line.startswith('INIT'):
                board_str = line.split()[1]
                initial_board = self._parse_board_string(board_str)
            
            # 플레이어 움직임 추출
            elif line.startswith('FIRST') or line.startswith('SECOND'):
                parts = line.split()
                player = 0 if parts[0] == 'FIRST' else 1
                r1, c1, r2, c2 = map(int, parts[1:5])
                time_used = int(parts[5]) if len(parts) > 5 else 0
                game_moves.append((player, r1, c1, r2, c2, time_used))
            
            # 최종 점수 추출
            elif line.startswith('SCOREFIRST'):
                first_score = int(line.split()[1])
                final_scores = (first_score, None)
            elif line.startswith('SCORESECOND'):
                second_score = int(line.split()[1])
                if final_scores:
                    final_scores = (final_scores[0], second_score)
                else:
                    final_scores = (None, second_score)
        
        return initial_board, game_moves, final_scores
    
    def _parse_board_string(self, board_str: str) -> List[List[int]]:
        """보드 문자열을 2D 리스트로 변환"""
        # "61682978525943196 34445163876398296 ..." 형식
        rows = board_str.split()
        board = []
        
        for row_str in rows:
            row = [int(c) for c in row_str]
            board.append(row)
        
        return board
    
    def _convert_to_selfplay_data(self, initial_board: List[List[int]], 
                                 game_moves: List[Tuple], 
                                 final_scores: Tuple[int, int]) -> SelfPlayData:
        """게임 재생성하며 SelfPlayData 생성"""
        
        # 게임 보드 초기화
        game_board = SimpleGameBoard(initial_board)
        game_states = []
        
        # 각 움직임마다 상태 저장
        for move_idx, (player, r1, c1, r2, c2, time_used) in enumerate(game_moves):
            # 현재 상태 텐서 생성
            state_tensor = game_board.get_state_tensor(player)
            
            # Policy target 생성 (AB 선택 + 노이즈 분배)
            if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
                # 패스 - 패스 움직임에 100% 확률 부여
                policy_target = [(-1, -1, -1, -1, 1.0)]
            else:
                # 현재 유효한 움직임들 가져오기
                valid_moves = game_board.get_valid_moves()
                policy_target = []
                
                if len(valid_moves) > 1:
                    # AB 선택 움직임 찾기
                    ab_move_found = False
                    other_moves = []
                    
                    for move in valid_moves:
                        if len(move) >= 4 and move[0] == r1 and move[1] == c1 and move[2] == r2 and move[3] == c2:
                            ab_move_found = True
                        else:
                            other_moves.append(move)
                    
                    if ab_move_found and other_moves:
                        # AB 움직임 + 노이즈 분배
                        noise_prob_per_move = self.noise_ratio / len(other_moves)
                        
                        # AB 선택 움직임
                        policy_target.append((r1, c1, r2, c2, self.ab_move_ratio))
                        
                        # 다른 움직임들에 노이즈 분배
                        for move in other_moves:
                            policy_target.append((*move, noise_prob_per_move))
                    else:
                        # AB 움직임을 찾을 수 없는 경우, 모든 움직임에 균등 분배
                        equal_prob = 1.0 / len(valid_moves)
                        for move in valid_moves:
                            policy_target.append((*move, equal_prob))
                else:
                    # 유효한 움직임이 하나뿐이면 100% 확률
                    policy_target = [(r1, c1, r2, c2, 1.0)]
            
            # GameState 생성
            game_state = GameState(
                state_tensor=state_tensor,
                policy_target=policy_target,
                player=player,
                move_number=move_idx,
                mcts_simulations=0,  # AB pruning이므로 MCTS 시뮬레이션 없음
                valid_moves_count=len(game_board.get_valid_moves())
            )
            
            game_states.append(game_state)
            
            # 움직임 실행
            if not (r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1):
                game_board.make_move(r1, c1, r2, c2, player)
        
        # 최종 결과 계산
        if final_scores and final_scores[0] is not None and final_scores[1] is not None:
            score_0, score_1 = final_scores
            final_score = (score_0, score_1)
            
            # 승자 결정
            if score_0 > score_1:
                winner = 0
                final_result = {0: 1.0, 1: -1.0}
            elif score_1 > score_0:
                winner = 1  
                final_result = {0: -1.0, 1: 1.0}
            else:
                winner = -1
                final_result = {0: 0.0, 1: 0.0}
        else:
            # 점수 정보가 없으면 게임보드에서 계산
            final_score = game_board.get_score()
            if final_score[0] > final_score[1]:
                winner = 0
                final_result = {0: 1.0, 1: -1.0}
            elif final_score[1] > final_score[0]:
                winner = 1
                final_result = {0: -1.0, 1: 1.0}
            else:
                winner = -1
                final_result = {0: 0.0, 1: 0.0}
        
        # 각 게임 상태에 최종 결과 반영 (백프로파게이션)
        for i, game_state in enumerate(game_states):
            # 플레이어 관점에서 결과 조정
            player = game_state.player
            # 간단한 백프로파게이션: 게임 길이에 따른 할인 없이 최종 결과만 사용
            game_state.value_target = final_result[player]
        
        return SelfPlayData(
            game_states=game_states,
            final_result=final_result,
            game_length=len(game_moves),
            final_score=final_score,
            winner=winner
        )


def test_converter():
    """테스트 함수"""
    converter = ABLogConverter()
    
    # 테스트 로그 파일 경로
    log_path = "practice/testing/log.txt"
    
    data = converter.parse_log_file(log_path)
    if data:
        print(f"성공적으로 변환됨:")
        print(f"  게임 길이: {data.game_length}")
        print(f"  최종 점수: {data.final_score}")
        print(f"  승자: {data.winner}")
        print(f"  게임 상태 수: {len(data.game_states)}")
        
        if data.game_states:
            first_state = data.game_states[0]
            print(f"  첫 번째 상태 - 플레이어: {first_state.player}")
            print(f"  첫 번째 상태 - 정책 타겟: {first_state.policy_target}")
    else:
        print("변환 실패")


if __name__ == "__main__":
    test_converter()