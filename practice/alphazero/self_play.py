import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from game_board import GameBoard
from mcts import MCTS

@dataclass
class GameState:
    """게임 상태 정보"""
    state_tensor: np.ndarray  # (2, 10, 17)
    policy_target: np.ndarray  # (action_space_size,)
    player: int
    move_number: int
    mcts_simulations: int  # 실제 수행된 MCTS 시뮬레이션 횟수
    valid_moves_count: int  # 유효한 움직임 개수

@dataclass 
class SelfPlayData:
    """한 게임의 자기대국 데이터"""
    game_states: List[GameState]
    final_result: Dict[int, float]  # {player_id: reward}
    game_length: int
    final_score: Tuple[int, int]  # (player0_score, player1_score)
    winner: int  # 0, 1, or -1 (draw)
    
    def __len__(self):
        return len(self.game_states)

class SelfPlayGenerator:
    """자기대국 데이터 생성기"""
    
    def __init__(self, neural_network, num_simulations: int = 800, 
                 temperature: float = 1.0, c_puct: float = 1.0, engine_type: str = 'neural', 
                 time_limit: float = None, num_threads: int = 4, batch_size: int = 32):
        self.neural_network = neural_network
        self.mcts = MCTS(
            neural_network, num_simulations, c_puct, 
            time_limit=time_limit, engine_type=engine_type,
            num_threads=num_threads, batch_size=batch_size
        )
        self.temperature = temperature
        self.num_simulations = num_simulations
        self.engine_type = engine_type
        self.num_threads = num_threads
        self.batch_size = batch_size
        
    def play_game(self, initial_board: List[List[int]], verbose: bool = False) -> SelfPlayData:
        """한 게임 자기대국 실행"""
        game_board = GameBoard(initial_board)
        game_states = []
        move_count = 0
        max_moves = 200  # 무한 루프 방지
        
        if verbose:
            print("Starting self-play game...")
            print(f"Initial board shape: {len(initial_board)}x{len(initial_board[0])}")
        
        total_simulations = 0
        
        while not game_board.is_terminal() and move_count < max_moves:
            current_player = game_board.current_player
            valid_moves = game_board.get_valid_moves()
            
            # 패스 옵션을 항상 추가 (게임 규칙상 언제든 패스 가능)
            if (-1, -1, -1, -1) not in valid_moves:
                valid_moves.append((-1, -1, -1, -1))
            
            # 최적화된 MCTS 호출 (한 번에 모든 정보 획득)
            try:
                best_move, action_probs, policy_vector, actual_simulations = self.mcts.get_move_and_probs(
                    game_board, current_player, self.temperature
                )
                total_simulations += actual_simulations
                
                # 상태 텐서 생성
                state_tensor = game_board.get_state_tensor(current_player)
                
                # 게임 상태 저장
                game_state = GameState(
                    state_tensor=state_tensor.copy(),
                    policy_target=policy_vector.copy(),
                    player=current_player,
                    move_number=move_count,
                    mcts_simulations=actual_simulations,
                    valid_moves_count=len(valid_moves)
                )
                game_states.append(game_state)
                
                
                if verbose and move_count % 10 == 0:
                    print(f"Move {move_count}: Player {current_player} -> {best_move} "
                          f"(valid:{len(valid_moves)}, sim:{actual_simulations})")
                
                # 움직임 실행
                success = game_board.make_move(*best_move, current_player)
                if not success:
                    if verbose:
                        print(f"Invalid move attempted: {best_move}")
                    break
                
                move_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error during move {move_count}: {e}")
                break
        
        # 게임 결과 계산
        final_result = {}
        if game_board.is_terminal():
            winner = game_board.get_winner()
            if winner == -1 or winner is None:  # 무승부 또는 승자 결정 불가
                final_result = {0: 0.0, 1: 0.0}
            else:
                final_result = {winner: 1.0, 1-winner: -1.0}
        else:
            # 시간 초과나 오류로 끝난 경우 점수로 판정
            score = game_board.get_score()
            if score[0] > score[1]:
                final_result = {0: 1.0, 1: -1.0}
            elif score[1] > score[0]:
                final_result = {0: -1.0, 1: 1.0}
            else:
                final_result = {0: 0.0, 1: 0.0}
        
        final_score = game_board.get_score()
        winner = game_board.get_winner() if game_board.get_winner() is not None else -1
        winner_text = "Draw" if winner == -1 else f"Player {winner}"
        
        avg_simulations = total_simulations / move_count if move_count > 0 else 0
        
        if verbose:
            print(f"Game finished after {move_count} moves")
            print(f"Total MCTS simulations: {total_simulations} (avg: {avg_simulations:.0f})")
            print(f"Final result: {final_result}")
            print(f"Final score: P0={final_score[0]}, P1={final_score[1]} ({winner_text})")
            
        
        return SelfPlayData(
            game_states=game_states,
            final_result=final_result,
            game_length=move_count,
            final_score=final_score,
            winner=winner
        )
    
    def generate_games(self, initial_board: List[List[int]], num_games: int, 
                      verbose: bool = False, use_random_boards: bool = False) -> List[SelfPlayData]:
        """여러 게임의 자기대국 데이터 생성"""
        games_data = []
        
        print(f"Generating {num_games} self-play games...")
        start_time = time.time()
        
        for game_idx in range(num_games):
            show_progress = verbose or (game_idx + 1) % max(1, num_games // 10) == 0
            
            try:
                # 랜덤 보드 사용 여부에 따라 보드 선택
                if use_random_boards:
                    import random
                    game_board = [[random.randint(1, 9) for _ in range(17)] for _ in range(10)]
                else:
                    game_board = initial_board
                    
                game_data = self.play_game(game_board, verbose=verbose and game_idx < 2)
                if len(game_data.game_states) > 0:  # 유효한 게임만 추가
                    games_data.append(game_data)
                    
                    # 게임 요약을 한 줄로 통합하여 출력
                    winner_text = "Draw" if game_data.winner == -1 else f"P{game_data.winner}"
                    score_text = f"P0={game_data.final_score[0]} P1={game_data.final_score[1]}"
                    
                    # 평균 시뮬레이션 계산
                    avg_sim = sum(state.mcts_simulations for state in game_data.game_states) / len(game_data.game_states) if game_data.game_states else 0
                    
                    # ETA 계산
                    if show_progress:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (game_idx + 1) if game_idx > 0 else 0
                        eta = avg_time * (num_games - game_idx - 1)
                        progress_text = f" | Avg:{avg_time:.1f}s ETA:{eta:.0f}s"
                    else:
                        progress_text = ""
                    
                    print(f"Game {game_idx + 1}: {game_data.game_length}moves {avg_sim:.0f}sim {score_text} {winner_text}{progress_text}")
                else:
                    print(f"Game {game_idx + 1}: No training data produced")
            except Exception as e:
                print(f"Game {game_idx + 1}: Error - {e}")
                continue
        
        total_time = time.time() - start_time
        valid_games = len(games_data)
        total_states = sum(len(game.game_states) for game in games_data)
        
        # 게임 결과 통계
        if valid_games > 0:
            p0_wins = sum(1 for game in games_data if game.winner == 0)
            p1_wins = sum(1 for game in games_data if game.winner == 1)
            draws = sum(1 for game in games_data if game.winner == -1)
            
            print(f"Generated {valid_games}/{num_games} valid games in {total_time:.1f}s")
            print(f"Results: P0={p0_wins} P1={p1_wins} Draw={draws}")
            print(f"Total training states: {total_states}")
            print(f"Average game length: {total_states/valid_games:.1f}")
        else:
            print("No valid games generated!")
        
        return games_data
    
    def collect_training_data(self, games_data: List[SelfPlayData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """게임 데이터를 훈련용 형태로 변환"""
        if not games_data:
            return np.array([]), np.array([]), np.array([])
        
        states = []
        policy_targets = []
        value_targets = []
        
        for game_data in games_data:
            final_result = game_data.final_result
            
            for game_state in game_data.game_states:
                player = game_state.player
                
                # 상태와 정책 타겟 추가
                states.append(game_state.state_tensor)
                policy_targets.append(game_state.policy_target)
                
                # 가치 타겟 (게임 결과)
                if player in final_result:
                    value_target = final_result[player]
                else:
                    value_target = 0.0
                
                value_targets.append(value_target)
        
        if not states:
            return np.array([]), np.array([]), np.array([])
        
        # numpy 배열로 변환
        states = np.stack(states)  # (N, 2, 10, 17)
        policy_targets = np.stack(policy_targets)  # (N, action_space_size)
        value_targets = np.array(value_targets, dtype=np.float32)  # (N,)
        
        print(f"Collected training data shapes:")
        print(f"  States: {states.shape}")
        print(f"  Policy targets: {policy_targets.shape}")
        print(f"  Value targets: {value_targets.shape}")
        
        return states, policy_targets, value_targets


