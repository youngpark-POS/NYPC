#!/usr/bin/env python3
"""
Basic AI for NYPC following the official protocol

This AI uses simple heuristic strategies.
It follows the exact same input/output protocol as sample_code.py.
"""

import sys
import os

# ================================
# Game 클래스: 게임 상태 관리 (Basic 버전)
# ================================
class BasicGame:
    
    def __init__(self, board, first):
        self.board = board            # 게임 보드 (2차원 배열)
        self.first = first            # 선공 여부
        self.passed = False           # 마지막 턴에 패스했는지 여부
    
    # 사각형 (r1, c1) ~ (r2, c2)이 유효한지 검사 (합이 10이고, 네 변을 모두 포함)
    def isValid(self, r1, c1, r2, c2):
        sums = 0
        r1fit = c1fit = r2fit = c2fit = False

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] != 0:
                    sums += self.board[r][c]
                    if r == r1:
                        r1fit = True
                    if r == r2:
                        r2fit = True
                    if c == c1:
                        c1fit = True
                    if c == c2:
                        c2fit = True
        return sums == 10 and r1fit and r2fit and c1fit and c2fit

    # ================================================================
    # ===================== [필수 구현] ===============================
    # 휴리스틱을 사용한 수 계산 (가장 큰 면적의 사각형 선택)
    # ================================================================
    def calculateMove(self, myTime, oppTime):
        # 전략: 가장 큰 면적의 유효한 사각형을 찾는다
        best_move = None
        best_area = 0
        
        # 모든 가능한 사각형을 확인
        for r1 in range(len(self.board)):
            for c1 in range(len(self.board[r1])):
                for r2 in range(r1, len(self.board)):
                    for c2 in range(c1, len(self.board[r2])):
                        if self.isValid(r1, c1, r2, c2):
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if area > best_area:
                                best_area = area
                                best_move = (r1, c1, r2, c2)
        
        if best_move is not None:
            print(f"Selected move {best_move} with area {best_area}", file=sys.stderr)
            return best_move
        else:
            print("No valid moves found, passing", file=sys.stderr)
            return (-1, -1, -1, -1)
    # =================== [필수 구현 끝] =============================

    # 상대방의 수를 받아 보드에 반영
    def updateOpponentAction(self, action, time_taken):
        self.updateMove(*action, False)

    # 주어진 수를 보드에 반영 (칸을 0으로 지움)
    def updateMove(self, r1, c1, r2, c2, isMyMove):
        if r1 == c1 == r2 == c2 == -1:
            self.passed = True
            return
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
        self.passed = False


# ================================
# main(): 입출력 처리 및 게임 진행
# ================================
def main():
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
                game = BasicGame(board, first)
                print(f"Basic AI initialized as {'FIRST' if first else 'SECOND'} player", file=sys.stderr)
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
                
                move_str = f"({r1},{c1})-({r2},{c2})" if r1 != -1 else "PASS"
                print(f"Opponent played: {move_str}", file=sys.stderr)
                continue

            if command == "FINISH":
                print("Basic AI game finished", file=sys.stderr)
                break

            print(f"Unknown command: {command}", file=sys.stderr)
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error in main loop: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    main()