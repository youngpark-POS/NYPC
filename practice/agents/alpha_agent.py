#!/usr/bin/env python3
"""
AlphaZero-style Agent for NYPC following the official protocol

Uses combined Policy-Value network with MCTS for move selection.
This is the complete implementation with value network evaluation.
"""

import sys
import os
import time
import random

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.game_board import GameBoard
from mcts.mcts_search import create_alphazero_mcts, create_value_guided_mcts

# Try to import neural network components
try:
    import torch
    from core.alphazero_net import create_alphazero_net, AlphaZeroNet
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, using fallback mode", file=sys.stderr)
    TORCH_AVAILABLE = False

# ================================
# Game 클래스: 게임 상태 관리 (AlphaZero 버전)
# ================================
class AlphaGame:
    
    def __init__(self, board, first, model_path=None, mcts_simulations=100, mcts_time=1.0):
        self.board_data = board        # 원본 보드 데이터 (2차원 배열)
        self.first = first             # 선공 여부
        self.passed = False            # 마지막 턴에 패스했는지 여부
        
        # GameBoard 객체 생성 (향상된 게임 로직 사용)
        self.game_board = GameBoard(board)
        self.current_player = 0 if first else 1
        
        # MCTS configuration
        self.mcts_simulations = mcts_simulations
        self.mcts_time = mcts_time
        
        # Neural network setup
        self.combined_net = None
        self.value_net = None
        self.use_neural_net = False
        self.use_combined_net = False
        
        if TORCH_AVAILABLE and model_path:
            try:
                # Try to load combined network first
                self.combined_net = create_alphazero_net('cpu')
                
                if os.path.exists(model_path):
                    print(f"Loading AlphaZero model from {model_path}", file=sys.stderr)
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    # Check if this is a combined model
                    if 'model_type' in checkpoint and checkpoint['model_type'] == 'combined':
                        self.combined_net.load_state_dict(checkpoint['model_state_dict'])
                        self.use_combined_net = True
                        print("Combined policy-value network loaded successfully", file=sys.stderr)
                    else:
                        # Try as value network only
                        self.value_net = create_value_net('cpu')
                        self.value_net.load_state_dict(checkpoint['model_state_dict'])
                        print("Value network loaded successfully", file=sys.stderr)
                    
                    self.use_neural_net = True
                else:
                    print(f"Model file not found: {model_path}, using random MCTS", file=sys.stderr)
            except Exception as e:
                print(f"Failed to load neural network: {e}, using random MCTS", file=sys.stderr)
        
        # Initialize MCTS
        if self.use_combined_net:
            # Full AlphaZero: Combined Policy-Value network
            self.mcts = create_alphazero_mcts(
                combined_net_function=self._combined_net_inference,
                max_simulations=mcts_simulations,
                max_time=mcts_time,
                exploration_weight=1.0  # Lower since we have good priors
            )
            print(f"AlphaGame initialized: {'FIRST' if first else 'SECOND'} player (AlphaZero mode)", file=sys.stderr)
        elif self.use_neural_net and self.value_net:
            # Value-guided MCTS only
            self.mcts = create_value_guided_mcts(
                value_function=self._value_evaluation,
                max_simulations=mcts_simulations,
                max_time=mcts_time,
                exploration_weight=1.4  # Higher since no policy priors
            )
            print(f"AlphaGame initialized: {'FIRST' if first else 'SECOND'} player (Value-guided MCTS mode)", file=sys.stderr)
        else:
            # Fallback to basic MCTS
            from mcts.mcts_search import create_basic_mcts
            self.mcts = create_basic_mcts(
                max_simulations=mcts_simulations,
                max_time=mcts_time
            )
            print(f"AlphaGame initialized: {'FIRST' if first else 'SECOND'} player (Basic MCTS mode)", file=sys.stderr)
    
    def _combined_net_inference(self, features, valid_moves):
        """Combined network inference for both policy and value."""
        try:
            if self.combined_net is None:
                if len(valid_moves) > 0:
                    return {move: 1.0/len(valid_moves) for move in valid_moves}, 0.0
                return {}, 0.0
            
            self.combined_net.eval()
            with torch.no_grad():
                # Convert features to tensor format
                if hasattr(features, 'shape') and len(features.shape) == 3:
                    # Neural features format (10, 17, 7)
                    features_tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).float()
                else:
                    # Fallback for other feature formats
                    print("Warning: Unexpected feature format, using zeros", file=sys.stderr)
                    features_tensor = torch.zeros(1, 7, 10, 17)
                
                # Get both policy and value predictions
                move_probs, win_prob = self.combined_net.predict_policy_value(
                    features.copy() if hasattr(features, 'copy') else features, 
                    valid_moves
                )
                
                return move_probs, win_prob
                
        except Exception as e:
            print(f"Combined network inference error: {e}", file=sys.stderr)
            # Fallback to uniform policy and neutral value
            if len(valid_moves) > 0:
                return {move: 1.0/len(valid_moves) for move in valid_moves}, 0.0
            return {}, 0.0
    
    def _value_evaluation(self, features, player):
        """Value network evaluation function for MCTS."""
        try:
            if self.value_net is None:
                return 0.0
            
            # Convert features if needed
            if hasattr(features, 'shape') and len(features.shape) == 3:
                value = self.value_net.predict_value(features, player)
            else:
                print("Warning: Unexpected feature format for value network", file=sys.stderr)
                return 0.0
            
            return value
            
        except Exception as e:
            print(f"Value evaluation error: {e}", file=sys.stderr)
            return 0.0
    
    # ================================================================
    # ===================== [필수 구현] ===============================
    # 직사각형이 유효한지 확인하는 함수
    # ================================================================
    def isValid(self, r1, c1, r2, c2):
        """Check if rectangle is valid according to game rules."""
        return self.game_board.is_valid_move(r1, c1, r2, c2)
    
    # ================================================================
    # ===================== [필수 구현] ===============================
    # 최적의 수를 계산하여 (r1, c1, r2, c2) 튜플로 반환
    # ================================================================
    def calculateMove(self, myTime, oppTime):
        """
        Calculate the best move using AlphaZero-style MCTS.
        """
        try:
            # Get all valid moves from GameBoard
            valid_moves = self.game_board.get_valid_moves()
            
            # If only pass move available
            if len(valid_moves) == 1 and valid_moves[0] == (-1, -1, -1, -1):
                print("Only pass move available", file=sys.stderr)
                return (-1, -1, -1, -1)
            
            # Adjust MCTS time based on available time
            search_time = min(self.mcts_time, myTime * 0.8)  # Use 80% of available time
            
            # Run MCTS search
            start_time = time.time()
            best_move, search_stats = self.mcts.search(
                initial_state=self.game_board,
                player=self.current_player,
                max_time=search_time
            )
            search_duration = time.time() - start_time
            
            # Log search results
            simulations = search_stats.get('simulations', 0)
            sims_per_sec = search_stats.get('simulations_per_second', 0)
            move_str = f"({best_move[0]},{best_move[1]})-({best_move[2]},{best_move[3]})" if best_move[0] != -1 else "PASS"
            
            mode = "AlphaZero" if self.use_combined_net else "Value-guided" if self.use_neural_net else "Basic"
            print(f"{mode} MCTS: {simulations} sims in {search_duration:.3f}s ({sims_per_sec:.1f} sims/s)", file=sys.stderr)
            print(f"Selected move: {move_str}", file=sys.stderr)
            
            # Debug: show network predictions if available
            if self.use_combined_net:
                try:
                    features = self.game_board.to_neural_features(self.current_player)
                    move_probs, win_prob = self._combined_net_inference(features, valid_moves)
                    if best_move in move_probs:
                        print(f"Network: win_prob={win_prob:.3f}, move_prior={move_probs[best_move]:.3f}", file=sys.stderr)
                except:
                    pass
            elif self.use_neural_net:
                try:
                    features = self.game_board.to_neural_features(self.current_player)
                    win_prob = self._value_evaluation(features, self.current_player)
                    print(f"Value network: win_prob={win_prob:.3f}", file=sys.stderr)
                except:
                    pass
            
            return best_move
            
        except Exception as e:
            print(f"Error in AlphaZero calculateMove: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Fallback to random valid move
            valid_moves = self.game_board.get_valid_moves()
            if valid_moves:
                fallback_move = random.choice(valid_moves)
                print(f"Fallback to random move: {fallback_move}", file=sys.stderr)
                return fallback_move
            return (-1, -1, -1, -1)
    
    def updateOpponentAction(self, action, time_taken):
        """상대방의 행동을 업데이트합니다."""
        try:
            r1, c1, r2, c2 = action
            
            # Update game board
            opponent_player = 1 - self.current_player
            success = self.game_board.make_move(r1, c1, r2, c2, opponent_player)
            
            if success:
                move_str = f"({r1},{c1})-({r2},{c2})" if r1 != -1 else "PASS"
                print(f"Opponent move: {move_str} (success)", file=sys.stderr)
                
                # Try to advance MCTS tree if possible
                try:
                    if hasattr(self.mcts, 'tree') and self.mcts.tree:
                        self.mcts.tree.advance_root((r1, c1, r2, c2))
                except:
                    pass  # Tree advancement failed, will rebuild on next search
                    
            else:
                print(f"Opponent move: ({r1},{c1})-({r2},{c2}) (failed)", file=sys.stderr)
                
        except Exception as e:
            print(f"Error updating opponent action: {e}", file=sys.stderr)
    
    def updateMove(self, r1, c1, r2, c2, isMyMove):
        """내 움직임이나 상대방 움직임을 업데이트합니다."""
        try:
            player = self.current_player if isMyMove else (1 - self.current_player)
            success = self.game_board.make_move(r1, c1, r2, c2, player)
            
            move_str = f"({r1},{c1})-({r2},{c2})" if r1 != -1 else "PASS"
            move_type = "My" if isMyMove else "Opponent"
            result = "success" if success else "failed"
            print(f"{move_type} move: {move_str} ({result})", file=sys.stderr)
            
        except Exception as e:
            print(f"Error in updateMove: {e}", file=sys.stderr)

# ================================================================
# ===================== [프로토콜 처리] =============================
# ================================================================
def main():
    """Main protocol handling function."""
    game = None
    first = None
    
    while True:
        try:
            line = input().split()

            if len(line) == 0:
                continue

            command, *param = line

            if command == "READY":
                # 선공 여부 확인
                turn = param[0]
                first = turn == "FIRST"
                print("OK", flush=True)
                continue

            if command == "INIT":
                # 보드 초기화
                board = [list(map(int, row)) for row in param]
                
                # Check for model path in command line arguments
                model_path = None
                if len(sys.argv) > 1:
                    model_path = sys.argv[1]
                
                game = AlphaGame(board, first, model_path)
                print(f"AlphaZero AI initialized as {'FIRST' if first else 'SECOND'} player", file=sys.stderr)
                continue

            if command == "TIME":
                # 내 턴: 수 계산 및 실행
                if game is None:
                    print("Error: Game not initialized", file=sys.stderr)
                    print("-1 -1 -1 -1", flush=True)
                    continue
                
                myTime, oppTime = map(int, param)
                
                ret = game.calculateMove(myTime, oppTime)
                game.updateMove(*ret, True)
                
                print(*ret, flush=True)
                continue

            if command == "OPP":
                # 상대 턴 반영
                if game is None:
                    print("Error: Game not initialized for opponent move", file=sys.stderr)
                    continue
                
                r1, c1, r2, c2, time_taken = map(int, param)
                game.updateOpponentAction((r1, c1, r2, c2), time_taken)
                continue

            if command == "FINISH":
                print("AlphaZero AI game finished", file=sys.stderr)
                if game and game.game_board:
                    scores = game.game_board.get_score()
                    print(f"Final scores: P1={scores[0]}, P2={scores[1]}", file=sys.stderr)
                break

            print(f"Unknown command: {command}", file=sys.stderr)
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error in main loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            break

if __name__ == "__main__":
    main()