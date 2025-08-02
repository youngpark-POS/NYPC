import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from game_board import GameBoard
from mcts import MCTS, MCTSPlayer

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
                 temperature: float = 1.0, c_puct: float = 1.0):
        self.neural_network = neural_network
        self.mcts = MCTS(neural_network, num_simulations, c_puct)
        self.temperature = temperature
        self.num_simulations = num_simulations
        
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
            if winner == -1:  # 무승부
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
        
        board_type = "random boards" if use_random_boards else "fixed board"
        print(f"Generating {num_games} self-play games with {board_type}...")
        start_time = time.time()
        
        for game_idx in range(num_games):
            show_progress = verbose or (game_idx + 1) % max(1, num_games // 10) == 0
            
            try:
                # 랜덤 보드 사용 여부에 따라 보드 선택
                if use_random_boards:
                    import random
                    game_board = [[random.randint(1, 5) for _ in range(17)] for _ in range(10)]
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

class CompetitivePlayer:
    """경쟁용 플레이어 (더 강한 탐색)"""
    
    def __init__(self, neural_network, num_simulations: int = 1600, 
                 temperature: float = 0.1):
        self.player = MCTSPlayer(neural_network, num_simulations, temperature)
        self.num_simulations = num_simulations
        
    def set_player_id(self, player_id: int):
        self.player.set_player_id(player_id)
        
    def get_move(self, game_board: GameBoard) -> Tuple[int, int, int, int]:
        return self.player.get_move(game_board)

def evaluate_models(model1, model2, initial_board: List[List[int]], 
                   num_games: int = 10, verbose: bool = False) -> Dict[str, float]:
    """두 모델 간의 대전 평가"""
    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}
    
    for game_idx in range(num_games):
        # 플레이어 순서 랜덤하게 결정
        if game_idx % 2 == 0:
            player1_model, player2_model = model1, model2
            is_flipped = False
        else:
            player1_model, player2_model = model2, model1  
            is_flipped = True
        
        # 플레이어 생성
        player1 = CompetitivePlayer(player1_model, num_simulations=800)
        player2 = CompetitivePlayer(player2_model, num_simulations=800)
        player1.set_player_id(0)
        player2.set_player_id(1)
        
        # 게임 실행
        game_board = GameBoard(initial_board)
        players = [player1, player2]
        move_count = 0
        
        while not game_board.is_terminal() and move_count < 200:
            current_player = game_board.current_player
            player = players[current_player]
            
            try:
                move = player.get_move(game_board)
                game_board.make_move(*move, current_player)
                move_count += 1
            except Exception as e:
                if verbose:
                    print(f"Error in game {game_idx + 1}: {e}")
                break
        
        # 결과 집계
        winner = game_board.get_winner()
        if winner == -1:
            results['draws'] += 1
        elif (winner == 0 and not is_flipped) or (winner == 1 and is_flipped):
            results['model1_wins'] += 1
        else:
            results['model2_wins'] += 1
        
        if verbose:
            print(f"Game {game_idx + 1}: Winner = {winner}, Moves = {move_count}")
    
    # 승률 계산
    total_games = sum(results.values())
    win_rate = results['model1_wins'] / total_games if total_games > 0 else 0.0
    
    results['win_rate'] = win_rate
    results['total_games'] = total_games
    
    return results

def debug_self_play(neural_network, initial_board: List[List[int]]):
    """0-0 무승부 원인 규명을 위한 디버깅 자기대국"""
    print("=" * 60)
    print("DEBUG SELF-PLAY: 0-0 무승부 원인 규명")
    print("=" * 60)
    
    generator = SelfPlayGenerator(neural_network, num_simulations=100, temperature=1.0)
    generator.mcts.time_limit = 5.0  # 5초 타임아웃 설정
    game_board = GameBoard(initial_board)
    
    print(f"Initial board: {len(initial_board)}x{len(initial_board[0])}")
    initial_valid_moves = game_board.get_valid_moves()
    print(f"Initial valid moves: {len(initial_valid_moves)}")
    if len(initial_valid_moves) <= 5:
        print(f"  First few moves: {initial_valid_moves[:5]}")
    
    print("\nStarting debug game...")
    print(game_board.display())
    print()
    
    move_count = 0
    max_moves = 50  # 디버깅용 제한
    
    while not game_board.is_terminal() and move_count < max_moves:
        current_player = game_board.current_player
        valid_moves = game_board.get_valid_moves()
        
        # 패스 옵션 추가
        if (-1, -1, -1, -1) not in valid_moves:
            valid_moves.append((-1, -1, -1, -1))
        
        print(f"\n--- Move {move_count + 1} | Player {current_player} ---")
        print(f"Valid moves: {len(valid_moves)} (including pass)")
        
        if len(valid_moves) <= 6:
            print(f"  All moves: {valid_moves}")
        else:
            print(f"  First 5 non-pass: {[m for m in valid_moves if m != (-1, -1, -1, -1)][:5]}")
            print(f"  Pass option: {(-1, -1, -1, -1) in valid_moves}")
        
        try:
            # 1단계: 액션 매핑 검증
            print("Step 1: Testing action encoding...")
            for i, move in enumerate(valid_moves[:3]):  # 처음 3개만 테스트
                encoded = game_board.encode_move(*move)
                print(f"  Move {move} -> Encoded: {encoded}")
                if encoded is None:
                    print(f"  WARNING: Failed to encode move {move}")
            
            # 2단계: 신경망 직접 테스트
            print("Step 2: Testing neural network prediction...")
            state_tensor = game_board.get_state_tensor(current_player)
            print(f"  State tensor shape: {state_tensor.shape}")
            
            try:
                policy_probs, value = generator.neural_network.predict(
                    state_tensor, valid_moves, game_board
                )
                print(f"  Neural network prediction successful!")
                print(f"  Policy probs length: {len(policy_probs)}")
                print(f"  Value: {value:.3f}")
            except Exception as nn_error:
                print(f"  ERROR in neural network: {nn_error}")
                import traceback
                traceback.print_exc()
                break
            
            # 3단계: 최적화된 MCTS 호출 (한 번에 모든 정보 획득)
            print("Step 3: Optimized MCTS calculation...")
            best_move, action_probs, policy_vector, actual_simulations = generator.mcts.get_move_and_probs(
                game_board, current_player, generator.temperature
            )
            print("  All MCTS calculations completed in one pass!")
            
            print(f"MCTS simulations: {actual_simulations} (optimized: 1x instead of 3x)")
            print(f"Selected move: {best_move}")
            
            if best_move == (-1, -1, -1, -1):
                print("  -> PASS selected!")
                if len([m for m in valid_moves if m != (-1, -1, -1, -1)]) > 0:
                    print("  -> WARNING: Non-pass moves available but MCTS chose PASS")
            
            # 상위 3개 액션 확률 표시
            if action_probs:
                # 확률 합계 검증
                total_prob = sum(action_probs.values())
                print(f"Action probabilities (total: {total_prob:.6f}):")
                
                sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
                print("Top 3 action probabilities:")
                for i, (action, prob) in enumerate(sorted_actions[:3]):
                    action_type = "PASS" if action == (-1, -1, -1, -1) else f"RECT{action}"
                    # 방문 횟수도 함께 표시
                    if hasattr(generator.mcts, '_last_root') and generator.mcts._last_root and action in generator.mcts._last_root.children:
                        visit_count = generator.mcts._last_root.children[action].visit_count
                        print(f"  {i+1}. {action_type}: {prob:.3f} (visits: {visit_count})")
                    else:
                        print(f"  {i+1}. {action_type}: {prob:.3f}")
                
                if abs(total_prob - 1.0) > 1e-6:
                    print(f"  WARNING: Probabilities don't sum to 1.0! (sum: {total_prob:.6f})")
            
            # 4단계: 움직임 실행 (Step 4,5 생략으로 번호 변경)
            print("Step 4: Executing move...")
            success = game_board.make_move(*best_move, current_player)
            if not success:
                print(f"ERROR: Invalid move attempted: {best_move}")
                break
            print("  Move executed successfully!")
            
            move_count += 1
            
            # 게임 상태 확인
            if game_board.is_terminal():
                winner = game_board.get_winner()
                score = game_board.get_score()
                print(f"\nGAME ENDED after {move_count} moves")
                print(f"Final score: P0={score[0]}, P1={score[1]}")
                print(f"Winner: {winner}")
                
                if score == (0, 0):
                    print("*** 0-0 DRAW DETECTED! ***")
                    print("Analyzing cause...")
                    
                    # 종료 원인 분석
                    if game_board.pass_count >= 2:
                        print("Cause: Two consecutive passes")
                    elif not game_board.get_valid_moves():
                        print("Cause: No valid moves available")
                    else:
                        print("Cause: Unknown - this shouldn't happen")
                break
                
        except Exception as e:
            print(f"ERROR during move {move_count + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if move_count >= max_moves:
        print(f"\nGame stopped after {max_moves} moves (debug limit)")
    
    final_score = game_board.get_score()
    print(f"\nFinal result: P0={final_score[0]}, P1={final_score[1]}")
    
    return final_score == (0, 0)  # True if 0-0 draw occurred

def quick_self_play_test(neural_network, initial_board: List[List[int]]):
    """빠른 자기대국 테스트"""
    print("Running quick self-play test...")
    
    generator = SelfPlayGenerator(neural_network, num_simulations=100, temperature=1.0)
    
    try:
        game_data = generator.play_game(initial_board, verbose=True)
        print(f"Test game completed with {len(game_data.game_states)} states")
        print(f"Final result: {game_data.final_result}")
        return True
    except Exception as e:
        print(f"Self-play test failed: {e}")
        return False