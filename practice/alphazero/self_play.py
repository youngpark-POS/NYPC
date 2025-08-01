import numpy as np
import torch
from typing import List, Tuple, Dict
import random
import pickle
from game_board import GameBoard
from neural_network import AlphaZeroNet
from mcts import MCTS

class SelfPlayData:
    """셀프플레이 데이터를 저장하는 클래스"""
    def __init__(self):
        self.states = []           # 게임 상태들 (2, 10, 17)
        self.policy_targets = []   # MCTS 방문 분포
        self.value_targets = []    # 게임 결과 (승부)
        self.valid_moves_list = [] # 각 상태의 유효한 움직임
    
    def add_sample(self, state: np.ndarray, policy_target: List[float], 
                   valid_moves: List[Tuple[int, int, int, int]]):
        """샘플 추가 (게임 결과는 나중에 업데이트)"""
        self.states.append(state.copy())
        self.policy_targets.append(policy_target.copy())
        self.valid_moves_list.append(valid_moves.copy())
        self.value_targets.append(0.0)  # 임시값, 나중에 업데이트
    
    def update_values(self, game_result: int):
        """게임 결과로 모든 value_target 업데이트"""
        for i in range(len(self.value_targets)):
            # 플레이어 교대를 고려하여 결과 할당
            player = i % 2
            if game_result == player:
                self.value_targets[i] = 1.0   # 승리
            elif game_result == 1 - player:
                self.value_targets[i] = -1.0  # 패배
            else:
                self.value_targets[i] = 0.0   # 무승부
    
    def get_training_data(self) -> Tuple[np.ndarray, List[List[float]], np.ndarray, List[List[Tuple[int, int, int, int]]]]:
        """학습용 데이터 반환"""
        return (np.array(self.states), 
                self.policy_targets, 
                np.array(self.value_targets), 
                self.valid_moves_list)
    
    def __len__(self):
        return len(self.states)

class SelfPlayGenerator:
    def __init__(self, model: AlphaZeroNet, num_simulations: int = 400, temperature: float = 1.0):
        self.model = model
        self.mcts = MCTS(model, num_simulations=num_simulations)
        self.temperature = temperature
    
    def play_game(self, initial_board: List[List[int]], verbose: bool = False) -> SelfPlayData:
        """
        한 게임의 셀프플레이 수행
        Returns:
            SelfPlayData: 수집된 학습 데이터
        """
        game_board = GameBoard(initial_board)
        game_data = SelfPlayData()
        current_player = 0
        move_count = 0
        
        if verbose:
            print(f"Starting self-play game")
            print(f"Initial board:\n{game_board}")
        
        while not game_board.is_game_over() and move_count < 200:  # 무한루프 방지
            # 현재 상태 저장
            state = game_board.get_state_tensor(current_player)
            
            # MCTS로 액션 확률 계산
            action_probs = self.mcts.get_action_probabilities(
                game_board, current_player, self.temperature
            )
            
            if not action_probs:
                # 유효한 움직임이 없으면 패스
                move = (-1, -1, -1, -1)
                valid_moves = [move]
                policy_target = [1.0]
            else:
                valid_moves = list(action_probs.keys())
                policy_target = list(action_probs.values())
                
                # 확률적으로 움직임 선택
                if self.temperature > 0:
                    move_idx = np.random.choice(len(valid_moves), p=policy_target)
                    move = valid_moves[move_idx]
                else:
                    # 가장 높은 확률의 움직임 선택
                    move_idx = np.argmax(policy_target)
                    move = valid_moves[move_idx]
            
            # 데이터 저장
            game_data.add_sample(state, policy_target, valid_moves)
            
            # 움직임 실행
            game_board.make_move(*move, current_player)
            
            if verbose:
                print(f"Move {move_count}: Player {current_player} -> {move}")
                print(f"Board state:\n{game_board}")
            
            # 플레이어 교대
            current_player = 1 - current_player
            move_count += 1
        
        # 게임 결과 계산
        winner = game_board.get_winner()
        if winner is not None:
            game_result = winner
        else:
            # 무승부 또는 점수로 승부 결정
            score_0 = game_board.get_score(0)
            score_1 = game_board.get_score(1)
            if score_0 > score_1:
                game_result = 0
            elif score_1 > score_0:
                game_result = 1
            else:
                game_result = -1  # 무승부
        
        # 게임 결과로 value target 업데이트
        if game_result != -1:  # 무승부가 아닌 경우만
            game_data.update_values(game_result)
        
        if verbose:
            print(f"Game finished. Winner: {game_result}, Moves: {move_count}")
            print(f"Final scores - Player 0: {game_board.get_score(0)}, Player 1: {game_board.get_score(1)}")
        
        return game_data
    
    def generate_games(self, initial_board: List[List[int]], num_games: int, verbose: bool = False) -> List[SelfPlayData]:
        """
        여러 게임의 셀프플레이 데이터 생성
        """
        all_game_data = []
        
        for game_idx in range(num_games):
            if verbose or game_idx % 10 == 0:
                print(f"Playing game {game_idx + 1}/{num_games}")
            
            game_data = self.play_game(initial_board, verbose=(verbose and game_idx < 3))
            all_game_data.append(game_data)
        
        return all_game_data
    
    def collect_training_data(self, game_data_list: List[SelfPlayData]) -> Tuple[np.ndarray, List[List[float]], np.ndarray, List[List[Tuple[int, int, int, int]]]]:
        """
        여러 게임의 데이터를 하나로 통합
        """
        all_states = []
        all_policy_targets = []
        all_value_targets = []
        all_valid_moves = []
        
        for game_data in game_data_list:
            states, policy_targets, value_targets, valid_moves_list = game_data.get_training_data()
            
            all_states.extend(states)
            all_policy_targets.extend(policy_targets)
            all_value_targets.extend(value_targets)
            all_valid_moves.extend(valid_moves_list)
        
        return (np.array(all_states), all_policy_targets, 
                np.array(all_value_targets), all_valid_moves)

class DataAugmentation:
    """게임 데이터 증강을 위한 클래스"""
    
    @staticmethod
    def flip_horizontal(state: np.ndarray, move: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """수평 뒤집기"""
        if len(state.shape) == 3:  # (2, 10, 17)
            new_state = np.flip(state, axis=2).copy()
        else:
            return state, move
        
        if move[0] == -1:  # 패스
            return new_state, move
        
        # 움직임 좌표 변환
        r1, c1, r2, c2 = move
        C = state.shape[2]
        new_c1 = C - 1 - c2
        new_c2 = C - 1 - c1
        new_move = (r1, new_c1, r2, new_c2)
        
        return new_state, new_move
    
    @staticmethod
    def flip_vertical(state: np.ndarray, move: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """수직 뒤집기"""
        if len(state.shape) == 3:  # (2, 10, 17)
            new_state = np.flip(state, axis=1).copy()
        else:
            return state, move
        
        if move[0] == -1:  # 패스
            return new_state, move
        
        # 움직임 좌표 변환
        r1, c1, r2, c2 = move
        R = state.shape[1]
        new_r1 = R - 1 - r2
        new_r2 = R - 1 - r1
        new_move = (new_r1, c1, new_r2, c2)
        
        return new_state, new_move
    
    @staticmethod
    def augment_data(states: np.ndarray, moves: List[Tuple[int, int, int, int]], 
                     policy_targets: List[List[float]], value_targets: np.ndarray,
                     valid_moves_list: List[List[Tuple[int, int, int, int]]]) -> Tuple:
        """데이터 증강 적용"""
        augmented_states = []
        augmented_policy_targets = []
        augmented_value_targets = []
        augmented_valid_moves = []
        
        for i in range(len(states)):
            state = states[i]
            policy_target = policy_targets[i]
            value_target = value_targets[i]
            valid_moves = valid_moves_list[i]
            
            # 원본 데이터
            augmented_states.append(state)
            augmented_policy_targets.append(policy_target)
            augmented_value_targets.append(value_target)
            augmented_valid_moves.append(valid_moves)
            
            # 수평 뒤집기 (50% 확률)
            if random.random() < 0.5:
                aug_state = np.flip(state, axis=2).copy()
                aug_valid_moves = []
                
                # 유효한 움직임들 변환
                for move in valid_moves:
                    if move[0] == -1:
                        aug_valid_moves.append(move)
                    else:
                        r1, c1, r2, c2 = move
                        C = state.shape[2]
                        new_c1 = C - 1 - c2
                        new_c2 = C - 1 - c1
                        aug_valid_moves.append((r1, new_c1, r2, new_c2))
                
                augmented_states.append(aug_state)
                augmented_policy_targets.append(policy_target)  # 정책은 순서만 바뀜
                augmented_value_targets.append(value_target)
                augmented_valid_moves.append(aug_valid_moves)
        
        return (np.array(augmented_states), augmented_policy_targets,
                np.array(augmented_value_targets), augmented_valid_moves)