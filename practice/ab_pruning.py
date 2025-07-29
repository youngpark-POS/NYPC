import math

from copy import deepcopy


# ================================
# Game 클래스: 게임 상태 관리
# ================================

"""

Team 'The geek diaries / 공대생의 혼잣말'
Juwon Kim, Junhyeok Lim, Yeongjae Park

CAUTION: Whenever using print() method, don't forget to set flush argument to True
so as to avoid TLE  e.g. print(*args, flush=True)

"""

GAMETREE_SEARCH_DEPTH = 2
BOARD_ROW = 9
BOARD_COLUMN = 16


class Game:

    def _calculate_board_value(self):
        return sum([sum(row) for row in self.territory_board])

    def _simulate(
        self,
        alpha: int,
        beta: int,
        is_max_player: int,
        depth: int,
    ) -> list[int | list[int]]:
        """
        Simulate all possibilities and find the best value of given state.

        Args:
            alpha: Current alpha value for this state.
            beta: Current beta value for this state.
            is_max_player: True if the max player is playing. False otherwise.
            depth: Current searching depth. The search terminates when this value reached GAMETREE_SEARCH_DEPTH.
        Returns:
            First integer represents the value of this state, and following list represents the best move.
        Raises:
            This method raises no exceptions.

        """

        original_board = deepcopy(self.board)
        original_territory_board = deepcopy(self.territory_board)
        best_value = -math.inf if is_max_player else math.inf
        best_move = [-1, -1, -1, -1]
        is_terminal = True

        if depth == GAMETREE_SEARCH_DEPTH:  #  In case of maximum searching depth
            return self._calculate_board_value(), best_move

        for r1 in range(BOARD_ROW):
            for r2 in range(r1 + 1, BOARD_ROW):
                for c1 in range(BOARD_COLUMN):
                    for c2 in range(c1 + 1, BOARD_COLUMN):
                        if not self._isValid(r1, c1, r2, c2):
                            continue

                        is_terminal = False

                        self.updateMove(r1, c1, r2, c2, is_max_player)
                        state_value = self._simulate(
                            alpha, beta, not is_max_player, depth + 1
                        )

                        if is_max_player and state_value > best_value:
                            best_move = [r1, c1, r2, c2]
                            best_value = state_value
                            alpha = max(alpha, state_value)
                        if not is_max_player and state_value < best_value:
                            best_move = [r1, c1, r2, c2]
                            best_value = state_value
                            beta = min(beta, state_value)

                        self.restoreMove(
                            r1, c1, r2, c2, original_board, original_territory_board
                        )

                        if alpha >= beta:  # pruning
                            return best_value, best_move

        if is_terminal:
            return self._calculate_board_value(), best_move
        else:
            return best_value, best_move

    def __init__(self, board, first):
        self.board = board
        self.territory_board = [[0 for _ in board[0]] for _ in board]
        self.first = first

    # 사각형 (r1, c1) ~ (r2, c2)이 유효한지 검사 (합이 10이고, 네 변을 모두 포함)
    def _isValid(self, r1, c1, r2, c2) -> bool:
        sums = 0
        r1fit = c1fit = r2fit = c2fit = False

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.board[r][c] != 0:
                    sums += self.board[r][c]
                    if sums > 10:
                        break
                    if r == r1:
                        r1fit = True
                    if r == r2:
                        r2fit = True
                    if c == c1:
                        c1fit = True
                    if c == c2:
                        c2fit = True
        return sums == 10 and r1fit and r2fit and c1fit and c2fit

    def calculateMove(self, _myTime, _oppTime) -> list[int]:
        best_value, best_move = self._simulate(
            -math.inf, math.inf, True, GAMETREE_SEARCH_DEPTH
        )
        if self._calculate_board_value() > best_value:
            return [-1, -1, -1, -1]
        else:
            return best_move

    def updateOpponentAction(self, action, _time) -> None:
        self.updateMove(*action, False)

    def restoreMove(
        self, r1, c1, r2, c2, original_board, original_territory_board
    ) -> None:
        if r1 == c1 == r2 == c2 == -1:
            return
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = original_board[r][c]
                self.territory_board[r][c] = original_territory_board[r][c]

    def updateMove(self, r1, c1, r2, c2, is_max_player) -> None:
        if r1 == c1 == r2 == c2 == -1:
            return
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
                self.territory_board[r][c] = 1 if is_max_player else -1


# ================================
# main(): 입출력 처리 및 게임 진행
# ================================
def main():
    while True:
        line = input().split()

        if len(line) == 0:
            continue

        command, *param = line

        if command == "READY":
            # 선공 여부 확인
            turn = param[0]
            global first
            first = turn == "FIRST"
            print("OK", flush=True)
            continue

        if command == "INIT":
            # 보드 초기화
            board = [list(map(int, row)) for row in param]
            global game
            game = Game(board, first)
            continue

        if command == "TIME":
            # 내 턴: 수 계산 및 실행
            myTime, oppTime = map(int, param)
            ret = game.calculateMove(myTime, oppTime)
            game.updateMove(*ret, True)
            print(*ret, flush=True)
            continue

        if command == "OPP":
            # 상대 턴 반영
            r1, c1, r2, c2, time = map(int, param)
            game.updateOpponentAction((r1, c1, r2, c2), time)
            continue

        if command == "FINISH":
            break

        assert False, f"Invalid command {command}"


if __name__ == "__main__":
    main()
