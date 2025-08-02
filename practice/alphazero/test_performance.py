"""
Python MCTS 성능 테스트 및 최적화 확인
"""

import time
import numpy as np
from game_board import GameBoard
from mcts import MCTS
from neural_network import AlphaZeroNet

def generate_random_board():
    """랜덤 게임 보드 생성"""
    import random
    return [[random.randint(1, 5) for _ in range(17)] for _ in range(10)]

def test_mcts_performance():
    """MCTS 성능 테스트"""
    print("=== MCTS 성능 테스트 ===")
    print()
    
    # 모델 생성
    temp_board = [[1] * 17 for _ in range(10)]
    temp_game = GameBoard(temp_board)
    action_space_size = temp_game.get_action_space_size()
    model = AlphaZeroNet(action_space_size=action_space_size)
    
    # 테스트 보드들 생성
    test_boards = [generate_random_board() for _ in range(5)]
    
    # 시뮬레이션 수 설정
    simulations_list = [50, 100, 200, 400]
    
    print("엔진 타입별 성능 비교:")
    print("-" * 60)
    
    for engine_type in ['heuristic', 'neural']:
        print(f"\n🔧 {engine_type.upper()} 엔진:")
        
        for num_sims in simulations_list:
            mcts = MCTS(model, num_simulations=num_sims, engine_type=engine_type)
            
            times = []
            total_moves = 0
            
            for i, board in enumerate(test_boards):
                game_board = GameBoard(board)
                valid_moves = game_board.get_valid_moves()
                total_moves += len(valid_moves)
                
                start_time = time.time()
                try:
                    best_move, actual_sims = mcts.get_best_move(game_board, 0, 0.0)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                except Exception as e:
                    print(f"    Error in board {i+1}: {e}")
                    continue
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_moves = total_moves / len(test_boards)
                moves_per_sec = avg_moves / avg_time if avg_time > 0 else 0
                
                print(f"  {num_sims:3d} sims: {avg_time:.3f}±{std_time:.3f}s "
                      f"({moves_per_sec:.0f} moves/s)")
            else:
                print(f"  {num_sims:3d} sims: Failed")

def test_valid_moves_performance():
    """get_valid_moves 성능 테스트"""
    print("\n=== get_valid_moves 성능 테스트 ===")
    print()
    
    test_boards = [generate_random_board() for _ in range(100)]
    
    times = []
    move_counts = []
    
    for board in test_boards:
        game_board = GameBoard(board)
        
        start_time = time.time()
        valid_moves = game_board.get_valid_moves()
        elapsed = time.time() - start_time
        
        times.append(elapsed)
        move_counts.append(len(valid_moves))
    
    avg_time = np.mean(times) * 1000  # ms로 변환
    std_time = np.std(times) * 1000
    avg_moves = np.mean(move_counts)
    
    print(f"평균 시간: {avg_time:.2f}±{std_time:.2f}ms")
    print(f"평균 유효 움직임: {avg_moves:.1f}개")
    print(f"처리량: {avg_moves/avg_time*1000:.0f} moves/s")

def compare_engines():
    """휴리스틱 vs 신경망 엔진 비교"""
    print("\n=== 엔진별 품질 비교 ===")
    print()
    
    # 모델 생성
    temp_board = [[1] * 17 for _ in range(10)]
    temp_game = GameBoard(temp_board)
    action_space_size = temp_game.get_action_space_size()
    model = AlphaZeroNet(action_space_size=action_space_size)
    
    # 같은 보드에서 두 엔진 비교
    test_board = generate_random_board()
    game_board = GameBoard(test_board)
    
    print(f"보드 크기: {len(test_board)}x{len(test_board[0])}")
    print(f"유효 움직임: {len(game_board.get_valid_moves())}개")
    print()
    
    for engine_type in ['heuristic', 'neural']:
        mcts = MCTS(model, num_simulations=200, engine_type=engine_type)
        
        start_time = time.time()
        best_move, actual_sims = mcts.get_best_move(game_board, 0, 0.0)
        elapsed = time.time() - start_time
        
        print(f"{engine_type.capitalize():>10}: {best_move} "
              f"({actual_sims} sims, {elapsed:.3f}s)")

def estimate_cpp_speedup():
    """C++ 구현 시 예상 성능 향상 계산"""
    print("\n=== C++ 성능 향상 예측 ===")
    print()
    
    # Python 휴리스틱 성능 측정
    temp_board = [[1] * 17 for _ in range(10)]
    temp_game = GameBoard(temp_board)
    action_space_size = temp_game.get_action_space_size()
    model = AlphaZeroNet(action_space_size=action_space_size)
    
    mcts = MCTS(model, num_simulations=100, engine_type='heuristic')
    
    test_board = generate_random_board()
    game_board = GameBoard(test_board)
    
    # 여러 번 측정
    times = []
    for _ in range(10):
        start_time = time.time()
        best_move, actual_sims = mcts.get_best_move(game_board, 0, 0.0)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    py_avg = np.mean(times)
    
    print(f"Python 휴리스틱 MCTS: {py_avg:.3f}s (100 sims)")
    print()
    print("예상 C++ 성능:")
    print(f"  보수적 예상 (10x):  {py_avg/10:.3f}s")
    print(f"  현실적 예상 (25x):  {py_avg/25:.3f}s")
    print(f"  낙관적 예상 (50x):  {py_avg/50:.3f}s")
    print()
    print("실제 게임에서:")
    print(f"  Python: {py_avg*200:.1f}s (200 moves)")
    print(f"  C++ 25x: {py_avg*200/25:.1f}s (200 moves)")

if __name__ == "__main__":
    test_valid_moves_performance()
    test_mcts_performance() 
    compare_engines()
    estimate_cpp_speedup()
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print()
    print("다음 단계:")
    print("1. Visual Studio Build Tools 설치")
    print("2. C++ 확장 모듈 빌드: cd cpp && pip install .")
    print("3. fast_mcts_wrapper.py로 성능 비교")