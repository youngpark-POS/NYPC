import math
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from game_board import GameBoard

def heuristic_evaluate_board(game_board: GameBoard, player: int) -> float:
    """간단한 휴리스틱 보드 평가"""
    if game_board.is_terminal():
        winner = game_board.get_winner()
        if winner == player:
            return 1.0
        elif winner == -1:
            return 0.0
        else:
            return -1.0
    
    # 간단한 휴리스틱: 점수 차이 기반
    scores = game_board.get_score()
    my_score = scores[player]
    opp_score = scores[1 - player]
    
    # 정규화된 점수 차이
    total_score = my_score + opp_score + 1  # +1로 0 나누기 방지
    score_diff = (my_score - opp_score) / total_score
    
    # -1 ~ 1 범위로 제한
    return max(-1.0, min(1.0, score_diff))

class MCTSNode:
    """MCTS 트리의 노드"""
    
    def __init__(self, state: GameBoard, parent: 'MCTSNode' = None, 
                 action: Tuple[int, int, int, int] = None, prior_prob: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action  # 부모에서 이 노드로 오는 액션
        self.prior_prob = prior_prob  # 신경망이 예측한 사전 확률
        
        self.children: Dict[Tuple[int, int, int, int], 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_fully_expanded(self) -> bool:
        """모든 가능한 액션이 확장되었는지 확인"""
        if not self.is_expanded:
            return False
        valid_moves = self.state.get_valid_moves()
        if not valid_moves:  # 유효한 움직임이 없으면 패스만 가능
            return (-1, -1, -1, -1) in self.children
        return len(self.children) >= len(valid_moves)
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """UCB 점수 계산"""
        if self.visit_count == 0:
            return float('inf')
        
        # Q-value (평균 가치)
        q_value = self.value_sum / self.visit_count
        
        # UCB 보너스
        if self.parent is None:
            return q_value
        
        exploration_bonus = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q_value + exploration_bonus
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """UCB 점수가 가장 높은 자식 노드 선택"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: List[float], valid_moves: List[Tuple[int, int, int, int]]):
        """노드 확장"""
        self.is_expanded = True
        
        # valid_moves에는 이미 패스가 포함되어 있어야 함
        # 각 움직임에 대해 자식 노드 생성
        for i, move in enumerate(valid_moves):
            if i < len(policy_probs):
                prior_prob = policy_probs[i]
            else:
                prior_prob = 1.0 / len(valid_moves)  # 균등 분포
            
            new_state = self.state.copy()
            if new_state.make_move(*move, self.state.current_player):
                self.children[move] = MCTSNode(new_state, self, move, prior_prob)
    
    def backup(self, value: float):
        """백프로파게이션"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 상대방의 관점에서는 가치를 뒤집음
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> Dict[Tuple[int, int, int, int], float]:
        """방문 횟수 기반 액션 확률 분포 반환"""
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]
        
        if temperature == 0:
            # 탐욕적 선택: 최대 방문 횟수 액션만 1.0
            best_action_idx = np.argmax(visit_counts)
            probs = [0.0] * len(actions)
            probs[best_action_idx] = 1.0
        else:
            # 온도 조절된 확률 분포
            if temperature == float('inf'):
                # 균등 분포
                probs = [1.0 / len(actions)] * len(actions)
            else:
                # 표준 AlphaZero 방식: visit_count^(1/T)
                visit_counts = np.array(visit_counts, dtype=np.float64)
                
                # 방문 횟수 0 처리 (최소값 보장)
                visit_counts = np.maximum(visit_counts, 1e-8)
                
                # 온도 적용
                scaled_counts = visit_counts ** (1.0 / temperature)
                
                # 정규화
                total = np.sum(scaled_counts)
                if total > 0:
                    probs = scaled_counts / total
                else:
                    # 안전장치: 균등 분포
                    probs = np.ones(len(actions)) / len(actions)
                
                # 정규화 검증 및 보정
                prob_sum = np.sum(probs)
                if abs(prob_sum - 1.0) > 1e-6:
                    probs = probs / prob_sum
                
                probs = probs.tolist()
        
        return dict(zip(actions, probs))

class MCTS:
    """Monte Carlo Tree Search 알고리즘"""
    
    def __init__(self, neural_network, num_simulations: int = 800, 
                 c_puct: float = 1.0, time_limit: float = None, engine_type: str = 'neural'):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.time_limit = time_limit  # 시간 제한 (초)
        self.engine_type = engine_type  # 'neural' or 'heuristic'
        
    def search(self, root_state: GameBoard, perspective_player: int) -> Tuple[MCTSNode, int]:
        """MCTS 검색 실행"""
        root = MCTSNode(root_state.copy())
        start_time = time.time()
        actual_simulations = 0
        
        # 디버깅: 첫 번째 시뮬레이션 시간 측정
        first_sim_time = None
        
        for simulation in range(self.num_simulations):
            sim_start = time.time()
            
            # 시간 제한 체크
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            # 1. Selection: 리프 노드까지 내려가기
            node = self._select(root)
            
            # 2. Expansion and Evaluation
            if node.state.is_terminal():
                # 터미널 노드면 실제 게임 결과 사용
                value = node.state.get_reward(perspective_player)
            else:
                # 신경망으로 평가하고 확장
                value = self._expand_and_evaluate(node, perspective_player)
            
            # 3. Backup: 결과를 루트까지 전파
            node.backup(value)
            actual_simulations += 1
            
            # 첫 번째 시뮬레이션 시간 기록
            if first_sim_time is None:
                first_sim_time = time.time() - sim_start
        
        total_time = time.time() - start_time
        
        
        return root, actual_simulations
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Selection 단계: UCB를 사용하여 리프까지 이동"""
        current = root
        
        while current.is_fully_expanded() and not current.state.is_terminal():
            current = current.select_child(self.c_puct)
        
        return current
    
    def _expand_and_evaluate(self, node: MCTSNode, perspective_player: int) -> float:
        """Expansion과 Evaluation 단계"""
        state = node.state
        
        if self.engine_type == 'heuristic':
            # 휴리스틱 모드: 간단한 평가
            value = heuristic_evaluate_board(state, perspective_player)
            valid_moves = state.get_valid_moves()
            
            # 가장 큰 박스 또는 점수 이득이 큰 것 선택
            if not valid_moves:
                policy_probs = []
            else:
                move_scores = []
                
                for move in valid_moves:
                    if move == (-1, -1, -1, -1):  # 패스
                        move_scores.append(0.1)  # 낮은 점수
                    else:
                        r1, c1, r2, c2 = move
                        # 박스 면적 계산
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        
                        # 박스 내부 점수 합계 계산
                        box_sum = 0
                        for i in range(r1, r2 + 1):
                            for j in range(c1, c2 + 1):
                                if state.board[i][j] > 0:
                                    box_sum += state.board[i][j]
                        
                        # 휴리스틱: 박스 면적 - 박스 내부 점수 합계
                        heuristic_score = area - box_sum
                        move_scores.append(heuristic_score)
                
                # 정규화 (소프트맥스 스타일)
                if move_scores:
                    # 수치 안정성을 위해 최대값 빼기
                    max_score = max(move_scores)
                    exp_scores = [np.exp(score - max_score) for score in move_scores]
                    total = sum(exp_scores)
                    policy_probs = [exp_score / total for exp_score in exp_scores]
                else:
                    policy_probs = [1.0 / len(valid_moves)] * len(valid_moves)
        else:
            # 신경망 평가 (느림)
            valid_moves = state.get_valid_moves()
            try:
                state_tensor = state.get_state_tensor(perspective_player)
                policy_probs, value = self.neural_network.predict(
                    state_tensor, valid_moves, state
                )
            except Exception as e:
                # 신경망 오류 시 기본값 사용
                if valid_moves:
                    policy_probs = [1.0 / len(valid_moves)] * len(valid_moves)
                else:
                    policy_probs = [1.0]
                value = 0.0
        
        # 노드 확장
        if not node.is_expanded:
            node.expand(policy_probs, valid_moves)
        
        return value
    
    def get_best_move(self, state: GameBoard, perspective_player: int, 
                     temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], int]:
        """최적 움직임 반환"""
        root, actual_simulations = self.search(state, perspective_player)
        
        if not root.children:
            # 자식이 없으면 패스
            return (-1, -1, -1, -1), actual_simulations
        
        action_probs = root.get_action_probs(temperature)
        
        if not action_probs:
            return (-1, -1, -1, -1), actual_simulations
        
        if temperature == 0:
            # 탐욕적 선택
            best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # 확률적 선택
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            best_action = np.random.choice(len(actions), p=probs)
            best_action = actions[best_action]
        
        return best_action, actual_simulations
    
    def get_action_probabilities(self, state: GameBoard, perspective_player: int, 
                               temperature: float = 1.0) -> Tuple[Dict[Tuple[int, int, int, int], float], int]:
        """액션 확률 분포 반환 (훈련 데이터용)"""
        root, actual_simulations = self.search(state, perspective_player)
        self._last_root = root  # 디버깅용 루트 노드 저장
        return root.get_action_probs(temperature), actual_simulations
    
    def get_policy_vector(self, state: GameBoard, perspective_player: int, 
                         temperature: float = 1.0) -> Tuple[np.ndarray, int]:
        """전체 액션 공간에 대한 정책 벡터 반환"""
        action_probs, actual_simulations = self.get_action_probabilities(state, perspective_player, temperature)
        action_space_size = state.get_action_space_size()
        
        policy_vector = np.zeros(action_space_size, dtype=np.float32)
        
        for move, prob in action_probs.items():
            action_idx = state.encode_move(*move)
            if action_idx is not None:
                policy_vector[action_idx] = prob
        
        # 정규화
        if np.sum(policy_vector) > 0:
            policy_vector = policy_vector / np.sum(policy_vector)
        else:
            # 모든 확률이 0이면 균등 분포
            policy_vector.fill(1.0 / action_space_size)
        
        return policy_vector, actual_simulations
    
    def get_move_and_probs(self, state: GameBoard, perspective_player: int, 
                          temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], Dict, np.ndarray, int]:
        """
        최적화된 메서드: 한 번의 MCTS 검색으로 모든 정보 반환
        Returns:
            best_move: 선택된 움직임
            action_probs: 액션 확률 분포
            policy_vector: 8246차원 정책 벡터
            actual_simulations: 실제 시뮬레이션 횟수
        """
        # 한 번만 검색 실행
        root, actual_simulations = self.search(state, perspective_player)
        self._last_root = root  # 디버깅용
        
        # 1. 액션 확률 분포 계산
        action_probs = root.get_action_probs(temperature)
        
        # 2. 최적 움직임 선택
        if not action_probs:
            best_move = (-1, -1, -1, -1)
        elif temperature == 0:
            # 탐욕적 선택
            best_move = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # 확률적 선택
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            best_action_idx = np.random.choice(len(actions), p=probs)
            best_move = actions[best_action_idx]
        
        # 3. 정책 벡터 생성
        action_space_size = state.get_action_space_size()
        policy_vector = np.zeros(action_space_size, dtype=np.float32)
        
        for move, prob in action_probs.items():
            action_idx = state.encode_move(*move)
            if action_idx is not None:
                policy_vector[action_idx] = prob
        
        # 정규화
        if np.sum(policy_vector) > 0:
            policy_vector = policy_vector / np.sum(policy_vector)
        else:
            policy_vector.fill(1.0 / action_space_size)
        
        return best_move, action_probs, policy_vector, actual_simulations

class MCTSPlayer:
    """MCTS를 사용하는 플레이어"""
    
    def __init__(self, neural_network, num_simulations: int = 800, 
                 temperature: float = 1.0, c_puct: float = 1.0, time_limit: float = None):
        self.mcts = MCTS(neural_network, num_simulations, c_puct, time_limit)
        self.temperature = temperature
        self.player_id = None
    
    def set_player_id(self, player_id: int):
        """플레이어 ID 설정"""
        self.player_id = player_id
    
    def get_move(self, state: GameBoard) -> Tuple[int, int, int, int]:
        """움직임 결정"""
        if self.player_id is None:
            raise ValueError("Player ID not set")
        
        move, _ = self.mcts.get_best_move(state, self.player_id, self.temperature)
        return move
    
    def get_training_data(self, state: GameBoard) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 생성 (상태, 정책)"""
        if self.player_id is None:
            raise ValueError("Player ID not set")
        
        state_tensor = state.get_state_tensor(self.player_id)
        policy_vector, _ = self.mcts.get_policy_vector(state, self.player_id, self.temperature)
        
        return state_tensor, policy_vector