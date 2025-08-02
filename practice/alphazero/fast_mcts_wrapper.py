"""
Fast C++ MCTS와 Python 신경망을 연결하는 래퍼 클래스
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import torch

try:
    # C++ 확장 모듈 import
    import fast_mcts
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: fast_mcts C++ module not available. Install with: cd cpp && pip install .")

class FastMCTSWrapper:
    """C++ MCTS를 Python 신경망과 연결하는 래퍼 클래스"""
    
    def __init__(self, neural_network, num_simulations: int = 800, 
                 c_puct: float = 1.0, time_limit: float = None, engine_type: str = 'neural'):
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ MCTS module not available")
            
        self.neural_network = neural_network
        self.cpp_mcts = fast_mcts.MCTS(
            num_simulations=num_simulations,
            c_puct=c_puct, 
            time_limit=time_limit if time_limit else 0.0,
            engine_type=engine_type
        )
        
        # 신경망 콜백 설정
        if engine_type == 'neural':
            self.cpp_mcts.set_neural_callback(self._neural_callback)
    
    def _neural_callback(self, state_tensor_3d: List[List[List[float]]], 
                        valid_moves: List[Tuple[int, int, int, int]]) -> Tuple[List[float], float]:
        """C++에서 호출될 신경망 평가 콜백"""
        try:
            # 3D list를 numpy 배열로 변환
            state_array = np.array(state_tensor_3d, dtype=np.float32)  # (2, 10, 17)
            
            # 임시 GameBoard 객체 생성 (encode_move를 위해)
            from game_board import GameBoard
            temp_board = [[1] * 17 for _ in range(10)]
            temp_game = GameBoard(temp_board)
            
            # 신경망 예측 호출
            policy_probs, value = self.neural_network.predict(
                state_array, valid_moves, temp_game
            )
            
            return policy_probs, float(value)
            
        except Exception as e:
            print(f"Neural callback error: {e}")
            # 오류 시 균등 분포 반환
            uniform_prob = 1.0 / len(valid_moves) if valid_moves else 1.0
            return [uniform_prob] * len(valid_moves), 0.0
    
    def get_best_move(self, game_board, perspective_player: int, 
                     temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], int]:
        """최적 움직임 반환 (기존 MCTS와 호환 인터페이스)"""
        # Python GameBoard를 C++ GameBoard로 변환
        cpp_board = self._convert_to_cpp_board(game_board)
        
        move, simulations = self.cpp_mcts.get_best_move(
            cpp_board, perspective_player, temperature
        )
        
        return move, simulations
    
    def get_move_and_probs(self, game_board, perspective_player: int, 
                          temperature: float = 1.0) -> Tuple[Tuple[int, int, int, int], 
                                                           Dict, np.ndarray, int]:
        """최적화된 메서드: 한 번의 검색으로 모든 정보 반환"""
        # Python GameBoard를 C++ GameBoard로 변환
        cpp_board = self._convert_to_cpp_board(game_board)
        
        result = self.cpp_mcts.get_move_and_probs(
            cpp_board, perspective_player, temperature
        )
        
        # action_probs는 이미 Move가 키인 dict
        action_probs = dict(result.action_probs)
        
        return (result.best_move, action_probs, 
                np.array(result.policy_vector), result.actual_simulations)
    
    def _convert_to_cpp_board(self, python_board) -> 'fast_mcts.GameBoard':
        """Python GameBoard를 C++ GameBoard로 변환"""
        board_data = python_board.board
        return fast_mcts.GameBoard(board_data)
    
    def _convert_cpp_board_back(self, cpp_board, python_board):
        """C++ GameBoard에서 Python GameBoard로 상태 동기화 (필요시)"""
        # 현재는 불변 연산만 하므로 불필요
        pass


class HybridMCTS:
    """Python과 C++ MCTS를 선택적으로 사용할 수 있는 하이브리드 클래스"""
    
    def __init__(self, neural_network, num_simulations: int = 800, 
                 c_puct: float = 1.0, time_limit: float = None, 
                 engine_type: str = 'neural', use_cpp: bool = True):
        self.neural_network = neural_network
        self.use_cpp = use_cpp and CPP_AVAILABLE
        
        if self.use_cpp:
            print("Using C++ MCTS for improved performance")
            self.mcts = FastMCTSWrapper(neural_network, num_simulations, 
                                       c_puct, time_limit, engine_type)
        else:
            print("Using Python MCTS (C++ not available or disabled)")
            from mcts import MCTS
            self.mcts = MCTS(neural_network, num_simulations, c_puct, 
                           time_limit, engine_type)
    
    def get_best_move(self, game_board, perspective_player: int, 
                     temperature: float = 1.0):
        return self.mcts.get_best_move(game_board, perspective_player, temperature)
    
    def get_move_and_probs(self, game_board, perspective_player: int, 
                          temperature: float = 1.0):
        return self.mcts.get_move_and_probs(game_board, perspective_player, temperature)


def benchmark_mcts(neural_network, initial_board, iterations: int = 10):
    """Python vs C++ MCTS 성능 비교"""
    import time
    from game_board import GameBoard
    
    print(f"MCTS 성능 벤치마크 ({iterations}회)")
    print("=" * 50)
    
    # Python MCTS 테스트
    from mcts import MCTS
    py_mcts = MCTS(neural_network, num_simulations=100, engine_type='heuristic')
    
    py_times = []
    for i in range(iterations):
        game_board = GameBoard(initial_board)
        start_time = time.time()
        move, sims = py_mcts.get_best_move(game_board, 0, 0.0)
        elapsed = time.time() - start_time
        py_times.append(elapsed)
    
    py_avg = sum(py_times) / len(py_times)
    print(f"Python MCTS: {py_avg:.3f}s (avg)")
    
    # C++ MCTS 테스트 (가능한 경우)
    if CPP_AVAILABLE:
        cpp_mcts = HybridMCTS(neural_network, num_simulations=100, 
                             engine_type='heuristic', use_cpp=True)
        
        cpp_times = []
        for i in range(iterations):
            game_board = GameBoard(initial_board)
            start_time = time.time()
            move, sims = cpp_mcts.get_best_move(game_board, 0, 0.0)
            elapsed = time.time() - start_time
            cpp_times.append(elapsed)
        
        cpp_avg = sum(cpp_times) / len(cpp_times)
        speedup = py_avg / cpp_avg
        print(f"C++ MCTS:    {cpp_avg:.3f}s (avg)")
        print(f"Speedup:     {speedup:.1f}x")
    else:
        print("C++ MCTS: Not available")
        
    print("=" * 50)


if __name__ == "__main__":
    # 간단한 테스트
    if CPP_AVAILABLE:
        print("C++ MCTS module loaded successfully!")
        
        # 테스트 보드 생성
        test_board = [[i+1 for i in range(17)] for _ in range(10)]
        cpp_board = fast_mcts.GameBoard(test_board)
        
        print(f"Board created, valid moves: {len(cpp_board.get_valid_moves())}")
        print("C++ MCTS integration ready!")
    else:
        print("C++ MCTS module not available. Build with:")
        print("cd cpp && pip install .")