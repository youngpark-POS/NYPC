import math
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from game_board import GameBoard
import queue

class NeuralNetworkBatchProcessor:
    """신경망 배치 처리를 위한 클래스"""
    
    def __init__(self, neural_network, batch_size: int = 32, timeout: float = 0.01):
        self.neural_network = neural_network
        self.batch_size = batch_size
        self.timeout = timeout
        
        # 요청 큐와 결과 딕셔너리
        self.request_queue = queue.Queue()
        self.results = {}
        self.request_id_counter = 0
        self.lock = threading.Lock()
        
        # 배치 처리 스레드
        self.processing = False
        self.processor_thread = None
    
    def start_processing(self):
        """배치 처리 스레드 시작"""
        if not self.processing:
            self.processing = True
            self.processor_thread = threading.Thread(target=self._batch_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
    
    def stop_processing(self):
        """배치 처리 스레드 중지"""
        self.processing = False
        if self.processor_thread:
            self.processor_thread.join()
    
    def predict_batch(self, state_tensor, valid_moves, game_board) -> Tuple[List[float], float]:
        """신경망 예측 요청 (배치 처리됨)"""
        if not self.processing:
            self.start_processing()
        
        # 고유 요청 ID 생성
        with self.lock:
            request_id = self.request_id_counter
            self.request_id_counter += 1
        
        # 요청을 큐에 추가
        request = {
            'id': request_id,
            'state_tensor': state_tensor,
            'valid_moves': valid_moves,
            'game_board': game_board,
            'event': threading.Event()
        }
        
        self.request_queue.put(request)
        
        # 결과 대기
        request['event'].wait()
        
        # 결과 반환 및 정리
        with self.lock:
            result = self.results.pop(request_id)
        
        return result
    
    def _batch_processor(self):
        """배치 처리 메인 루프"""
        batch = []
        
        while self.processing:
            try:
                # 타임아웃으로 요청 수집
                request = self.request_queue.get(timeout=self.timeout)
                batch.append(request)
                
                # 배치가 가득 찼거나 큐가 비었으면 처리
                if len(batch) >= self.batch_size or self.request_queue.empty():
                    self._process_batch(batch)
                    batch = []
                    
            except queue.Empty:
                # 타임아웃 발생시 현재 배치 처리
                if batch:
                    self._process_batch(batch)
                    batch = []
    
    def _process_batch(self, batch):
        """실제 배치 처리"""
        if not batch:
            return
        
        # 개별 처리 (배치 처리 기능 미구현)
        for request in batch:
            try:
                policy_probs, value = self.neural_network.predict(
                    request['state_tensor'],
                    request['valid_moves'],
                    request['game_board']
                )
                
                with self.lock:
                    self.results[request['id']] = (policy_probs, value)
                
            except Exception as e:
                with self.lock:
                    self.results[request['id']] = ([], 0.0)
            
            # 결과 준비 완료 신호
            request['event'].set()

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
    """Path Compression 기반 MCTS 노드 - action sequence만 저장"""
    
    def __init__(self, action_sequence: List[Tuple[int, int, int, int]] = None, prior_prob: float = 0.0):
        # 핵심: 루트부터 이 노드까지의 모든 action들
        self.action_sequence = action_sequence or []
        self.prior_prob = prior_prob
        
        # 자식 노드들
        self.children: Dict[Tuple[int, int, int, int], 'MCTSNode'] = {}
        
        # MCTS 통계
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
        # 성능 최적화용 캐시
        self._cached_state = None
        self._cache_valid = False
        self._depth = len(self.action_sequence)  # 미리 계산해둠
    
    def get_game_state(self, root_state: GameBoard) -> GameBoard:
        """Action sequence를 순차 적용하여 현재 상태 재구성"""
        if self._cache_valid and self._cached_state is not None:
            return self._cached_state
        
        # 루트 상태에서 시작 (1회만 복사)
        state = root_state.copy()
        
        # Action sequence 순차 적용
        for action in self.action_sequence:
            r1, c1, r2, c2 = action
            if not state.make_move(r1, c1, r2, c2, state.current_player):
                # 잘못된 action이면 에러 (디버깅용)
                print(f"⚠️ Invalid action in sequence: {action}")
                break
        
        # 캐시 저장
        self._cached_state = state
        self._cache_valid = True
        return state
    
    def invalidate_cache(self):
        """캐시 무효화"""
        self._cache_valid = False
        self._cached_state = None
        
        # 자식들의 캐시도 무효화 (필요시)
        for child in self.children.values():
            child.invalidate_cache()
    
    def is_fully_expanded(self, root_state: GameBoard) -> bool:
        """모든 가능한 액션이 확장되었는지 확인"""
        if not self.is_expanded:
            return False
        
        current_state = self.get_game_state(root_state)
        valid_moves = current_state.get_valid_moves()
        
        if not valid_moves:  # 유효한 움직임이 없으면 패스만 가능
            return (-1, -1, -1, -1) in self.children
        
        return len(self.children) >= len(valid_moves)
    
    def get_ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """UCB 점수 계산"""
        if self.visit_count == 0:
            return float('inf')
        
        # Q-value (평균 가치)
        q_value = self.value_sum / self.visit_count
        
        # UCB 보너스
        if parent_visit_count == 0:
            return q_value
        
        exploration_bonus = c_puct * self.prior_prob * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return q_value + exploration_bonus
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """UCB 점수가 가장 높은 자식 노드 선택"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: List[float], valid_moves: List[Tuple[int, int, int, int]]):
        """노드 확장 - action sequence 확장만"""
        self.is_expanded = True
        
        # 각 움직임에 대해 새로운 action sequence 생성
        for i, move in enumerate(valid_moves):
            if i < len(policy_probs):
                prior_prob = policy_probs[i]
            else:
                prior_prob = 1.0 / len(valid_moves)
            
            # 새로운 action sequence = 현재 sequence + 새 action
            child_sequence = self.action_sequence + [move]
            
            # 자식 노드 생성 (메모리 최소화!)
            child = MCTSNode(
                action_sequence=child_sequence,
                prior_prob=prior_prob
            )
            
            self.children[move] = child
    
    def backup(self, value: float):
        """백프로파게이션 - 부모로 재귀 호출 없이"""
        # 현재 노드 업데이트
        self.visit_count += 1
        self.value_sum += value
    
    def get_action_probs(self, temperature: float = 1.0) -> Dict[Tuple[int, int, int, int], float]:
        """방문 횟수 기반 액션 확률 분포 반환"""
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]
        
        if temperature == 0:
            # 탐욕적 선택
            best_action_idx = np.argmax(visit_counts)
            probs = [0.0] * len(actions)
            probs[best_action_idx] = 1.0
        else:
            # 온도 조절된 확률 분포
            if temperature == float('inf'):
                probs = [1.0 / len(actions)] * len(actions)
            else:
                visit_counts = np.array(visit_counts, dtype=np.float64)
                visit_counts = np.maximum(visit_counts, 1e-8)
                
                scaled_counts = visit_counts ** (1.0 / temperature)
                total = np.sum(scaled_counts)
                
                if total > 0:
                    probs = scaled_counts / total
                else:
                    probs = np.ones(len(actions)) / len(actions)
                
                prob_sum = np.sum(probs)
                if abs(prob_sum - 1.0) > 1e-6:
                    probs = probs / prob_sum
                
                probs = probs.tolist()
        
        return dict(zip(actions, probs))

class MCTS:
    """Path Compression 기반 Ultra-Efficient MCTS"""
    
    def __init__(self, neural_network=None, num_simulations: int = 800, 
                 c_puct: float = 1.0, time_limit: float = None, engine_type: str = 'neural',
                 num_threads: int = 4, batch_size: int = 32):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.engine_type = engine_type
        self.num_threads = num_threads
        
        # 배치 처리기 (neural 엔진에서만 사용)
        self.batch_processor = None
        if self.engine_type == 'neural' and self.neural_network:
            self.batch_processor = NeuralNetworkBatchProcessor(
                self.neural_network, batch_size=batch_size
            )
    
    def search(self, root_state: GameBoard, perspective_player: int) -> Tuple[MCTSNode, int]:
        """Path 기반 MCTS 검색 - 멀티스레드 + 배치 처리"""
        # 루트 노드 생성 (action sequence 없음)
        root = MCTSNode()
        
        # 배치 처리기 시작 (neural 엔진인 경우)
        if self.batch_processor:
            self.batch_processor.start_processing()
        
        start_time = time.time()
        
        # 멀티스레드로 시뮬레이션 실행
        if self.num_threads > 1:
            actual_simulations = self._search_multithreaded(root, root_state, perspective_player, start_time)
        else:
            actual_simulations = self._search_single_threaded(root, root_state, perspective_player, start_time)
        
        # 배치 처리기 정리
        if self.batch_processor:
            self.batch_processor.stop_processing()
        
        return root, actual_simulations
    
    def _search_single_threaded(self, root: MCTSNode, root_state: GameBoard, perspective_player: int, start_time: float) -> int:
        """단일 스레드 MCTS 검색 (기존 로직)"""
        actual_simulations = 0
        
        for simulation in range(self.num_simulations):
            # 시간 제한 체크
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            actual_simulations += self._single_simulation(root, root_state, perspective_player)
        
        return actual_simulations
    
    def _search_multithreaded(self, root: MCTSNode, root_state: GameBoard, perspective_player: int, start_time: float) -> int:
        """멀티스레드 MCTS 검색"""
        simulations_per_thread = self.num_simulations // self.num_threads
        remaining_simulations = self.num_simulations % self.num_threads
        
        actual_simulations = 0
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 각 스레드에 시뮬레이션 할당
            futures = []
            for i in range(self.num_threads):
                thread_simulations = simulations_per_thread
                if i < remaining_simulations:  # 나머지 시뮬레이션 분배
                    thread_simulations += 1
                
                future = executor.submit(
                    self._thread_worker, 
                    root, root_state.copy(), perspective_player, 
                    thread_simulations, start_time
                )
                futures.append(future)
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    thread_simulations = future.result()
                    actual_simulations += thread_simulations
                except Exception as e:
                    print(f"Thread execution error: {e}")
        
        return actual_simulations
    
    def _thread_worker(self, root: MCTSNode, root_state: GameBoard, perspective_player: int, 
                      num_simulations: int, start_time: float) -> int:
        """개별 스레드에서 실행되는 워커"""
        actual_simulations = 0
        
        for simulation in range(num_simulations):
            # 시간 제한 체크
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            actual_simulations += self._single_simulation(root, root_state, perspective_player)
        
        return actual_simulations
    
    def _single_simulation(self, root: MCTSNode, root_state: GameBoard, perspective_player: int) -> int:
        """단일 시뮬레이션 실행"""
        # 1. Selection - 리프 노드까지 이동
        path_to_leaf = []  # 방문한 노드들의 경로
        current = root
        
        while current.is_fully_expanded(root_state) and not current.get_game_state(root_state).is_terminal():
            current = current.select_child(self.c_puct)
            path_to_leaf.append(current)
        
        # 2. Expansion and Evaluation
        current_state = current.get_game_state(root_state)
        
        if current_state.is_terminal():
            value = current_state.get_reward(perspective_player)
        else:
            value = self._expand_and_evaluate(current, perspective_player, root_state)
        
        # 3. Backup - 경로상의 모든 노드 업데이트
        current.backup(value)
        for i, node in enumerate(reversed(path_to_leaf[:-1])):  # 역순으로
            # 상대방 관점에서는 값 반전
            value = -value
            node.backup(value)
        
        return 1
    
    def _expand_and_evaluate(self, node: MCTSNode, perspective_player: int, root_state: GameBoard) -> float:
        """노드 확장 및 평가"""
        # 현재 상태 가져오기 (필요시 재구성)
        current_state = node.get_game_state(root_state)
        
        if self.engine_type == 'heuristic':
            # 휴리스틱 평가
            value = heuristic_evaluate_board(current_state, perspective_player)
            valid_moves = current_state.get_valid_moves()
            
            # 휴리스틱 정책 계산
            if not valid_moves:
                policy_probs = []
            else:
                move_scores = []
                for move in valid_moves:
                    if move == (-1, -1, -1, -1):
                        move_scores.append(0.1)
                    else:
                        r1, c1, r2, c2 = move
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        
                        box_sum = 0
                        for i in range(r1, r2 + 1):
                            for j in range(c1, c2 + 1):
                                if current_state.board[i][j] > 0:
                                    box_sum += current_state.board[i][j]
                        
                        heuristic_score = area - box_sum
                        move_scores.append(heuristic_score)
                
                if move_scores:
                    max_score = max(move_scores)
                    exp_scores = [np.exp(score - max_score) for score in move_scores]
                    total = sum(exp_scores)
                    policy_probs = [exp_score / total for exp_score in exp_scores]
                else:
                    policy_probs = [1.0 / len(valid_moves)] * len(valid_moves)
        else:
            # 신경망 평가 (배치 처리 사용)
            valid_moves = current_state.get_valid_moves()
            
            try:
                state_tensor = current_state.get_state_tensor(perspective_player)
                
                # 배치 처리기가 있으면 사용, 없으면 직접 호출
                if self.batch_processor:
                    policy_probs, value = self.batch_processor.predict_batch(
                        state_tensor, valid_moves, current_state
                    )
                else:
                    policy_probs, value = self.neural_network.predict(
                        state_tensor, valid_moves, current_state
                    )
            except Exception:
                if valid_moves:
                    policy_probs = [1.0 / len(valid_moves)] * len(valid_moves)
                else:
                    policy_probs = [1.0]
                value = 0.0
        
        # 노드 확장 (action sequence만 확장!)
        if not node.is_expanded and valid_moves:
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
        return root.get_action_probs(temperature), actual_simulations
    
    
    def get_move_and_box_data(self, state: GameBoard, perspective_player: int, 
                             temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], Dict, List[Tuple[int, int, int, int, float]], int]:
        """
        최적화된 메서드: 한 번의 MCTS 검색으로 모든 정보 반환 (YOLO 스타일)
        Returns:
            best_move: 선택된 움직임
            action_probs: 액션 확률 분포
            box_targets: YOLO 박스 훈련 데이터
            actual_simulations: 실제 시뮬레이션 횟수
        """
        # 한 번만 검색 실행
        root, actual_simulations = self.search(state, perspective_player)
        
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
        
        # 3. AlphaZero 정책 훈련 데이터 생성
        policy_target = []
        for move, prob in action_probs.items():
            if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
                r1, c1, r2, c2 = move
                policy_target.append((r1, c1, r2, c2, prob))
        
        return best_move, action_probs, policy_target, actual_simulations

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
    
    def get_training_data(self, state: GameBoard) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """훈련 데이터 생성 (상태, AlphaZero 정책 타겟)"""
        if self.player_id is None:
            raise ValueError("Player ID not set")
        
        state_tensor = state.get_state_tensor(self.player_id)
        action_probs, _ = self.mcts.get_action_probabilities(state, self.player_id, self.temperature)
        
        # 액션 확률을 정책 타겟으로 변환
        policy_target = []
        for move, prob in action_probs.items():
            if len(move) >= 4 and move != (-1, -1, -1, -1):  # 패스 제외
                r1, c1, r2, c2 = move
                policy_target.append((r1, c1, r2, c2, prob))
        
        return state_tensor, policy_target