import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from game_board import GameBoard
from neural_network import AlphaZeroNet

class MCTSNode:
    def __init__(self, game_board: GameBoard, player: int, parent: Optional['MCTSNode'] = None, 
                 parent_action: Optional[Tuple[int, int, int, int]] = None, prior_prob: float = 0.0):
        self.game_board = game_board.copy()
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob
        
        # MCTS 통계
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple[int, int, int, int], MCTSNode] = {}
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return not self.is_expanded
    
    def get_value(self) -> float:
        """노드의 평균 가치 반환"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """UCB1 점수 계산"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.get_value()
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """UCB1 점수가 가장 높은 자식 노드 선택"""
        best_child = None
        best_score = float('-inf')
        
        for child in self.children.values():
            score = child.get_ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, model: AlphaZeroNet):
        """노드 확장 (신경망으로 자식 노드들의 prior 확률 계산)"""
        if self.is_expanded:
            return
        
        # 게임 종료 체크
        if self.game_board.is_game_over():
            self.is_expanded = True
            return
        
        # 유효한 움직임 가져오기
        valid_moves = self.game_board.get_valid_moves()
        if not valid_moves:
            valid_moves = [(-1, -1, -1, -1)]  # 패스
        
        # 신경망으로 정책과 가치 예측
        state = self.game_board.get_state_tensor(self.player)
        move_probs, _ = model.get_move_probabilities(state, valid_moves, self.game_board)
        
        # 자식 노드 생성
        for move in valid_moves:
            if move in move_probs:
                prior_prob = move_probs[move]
                
                # 새로운 게임 상태 생성
                new_board = self.game_board.copy()
                new_board.make_move(*move, self.player)
                next_player = 1 - self.player
                
                # 자식 노드 생성
                child_node = MCTSNode(new_board, next_player, self, move, prior_prob)
                self.children[move] = child_node
        
        self.is_expanded = True
    
    def backup(self, value: float):
        """값을 역전파"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 상대방 관점에서 값 반전
            self.parent.backup(-value)
    
    def get_visit_distribution(self, temperature: float = 1.0) -> List[float]:
        """방문 횟수 기반 확률 분포 반환"""
        if not self.children:
            return []
        
        moves = list(self.children.keys())
        visits = [child.visit_count for child in self.children.values()]
        
        if temperature == 0:
            # 최대 방문 횟수인 행동만 선택
            max_visits = max(visits)
            probs = [1.0 if v == max_visits else 0.0 for v in visits]
            probs = [p / sum(probs) for p in probs]
        else:
            # Temperature 적용한 소프트맥스
            visits = np.array(visits, dtype=np.float64)
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
            probs = probs.tolist()
        
        return probs

class MCTS:
    def __init__(self, model: AlphaZeroNet, c_puct: float = 1.0, num_simulations: int = 800):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def search(self, game_board: GameBoard, player: int, temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], List[float], List[Tuple[int, int, int, int]]]:
        """
        MCTS 탐색 수행
        Returns:
            best_move: 선택된 최적 움직임
            visit_probs: 방문 횟수 기반 확률 분포 
            valid_moves: 유효한 움직임 리스트
        """
        root = MCTSNode(game_board, player)
        
        # 시뮬레이션 수행
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            
            # Selection: 리프 노드까지 내려가기
            while not node.is_leaf() and not node.game_board.is_game_over():
                node = node.select_child(self.c_puct)
                path.append(node)
            
            # Expansion & Evaluation
            if not node.game_board.is_game_over():
                node.expand(self.model)
                
                # 자식이 있으면 하나 선택
                if node.children:
                    node = node.select_child(self.c_puct)
                    path.append(node)
            
            # 현재 노드 평가
            if node.game_board.is_game_over():
                # 게임 종료시 실제 결과 사용
                winner = node.game_board.get_winner()
                if winner == player:
                    value = 1.0
                elif winner == 1 - player:
                    value = -1.0
                else:
                    value = 0.0  # 무승부
            else:
                # 신경망으로 가치 평가
                state = node.game_board.get_state_tensor(node.player)
                valid_moves = node.game_board.get_valid_moves()
                if not valid_moves:
                    valid_moves = [(-1, -1, -1, -1)]
                _, value = self.model.predict(state, valid_moves, node.game_board)
                
                # 현재 플레이어 관점으로 값 조정
                if node.player != player:
                    value = -value
            
            # Backup: 경로를 따라 값 역전파
            for node in reversed(path):
                node.backup(value)
                value = -value  # 플레이어 전환시 값 반전
        
        # 최적 움직임 선택
        if not root.children:
            return (-1, -1, -1, -1), [], [(-1, -1, -1, -1)]
        
        valid_moves = list(root.children.keys())
        visit_probs = root.get_visit_distribution(temperature)
        
        # 확률 분포에 따라 움직임 선택
        if temperature == 0 or len(visit_probs) == 1:
            # 탐욕적 선택
            best_idx = np.argmax([child.visit_count for child in root.children.values()])
            best_move = valid_moves[best_idx]
        else:
            # 확률적 선택
            best_idx = np.random.choice(len(valid_moves), p=visit_probs)
            best_move = valid_moves[best_idx]
        
        return best_move, visit_probs, valid_moves
    
    def get_action_probabilities(self, game_board: GameBoard, player: int, temperature: float = 1.0) -> Dict[Tuple[int, int, int, int], float]:
        """행동별 확률을 딕셔너리로 반환"""
        _, visit_probs, valid_moves = self.search(game_board, player, temperature)
        
        action_probs = {}
        for i, move in enumerate(valid_moves):
            action_probs[move] = visit_probs[i] if i < len(visit_probs) else 0.0
        
        return action_probs

    def get_best_move(self, game_board: GameBoard, player: int, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """최적 움직임만 반환"""
        best_move, _, _ = self.search(game_board, player, temperature)
        return best_move