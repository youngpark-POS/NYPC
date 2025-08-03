#include "game_board.h"
#include <algorithm>
#include <iostream>

GameBoard::GameBoard(const std::vector<std::vector<int>>& initial_board)
    : board(initial_board), current_player(0), pass_count(0), game_over(false), winner(-1) {
    build_action_mapping();
}

GameBoard::GameBoard(const GameBoard& other)
    : board(other.board), current_player(other.current_player), 
      pass_count(other.pass_count), game_over(other.game_over), 
      winner(other.winner), action_to_move(other.action_to_move),
      move_to_action(other.move_to_action) {
}

void GameBoard::build_action_mapping() {
    int action_idx = 0;
    
    // 모든 가능한 직사각형에 대해 매핑 생성
    for (int r1 = 0; r1 < R; ++r1) {
        for (int c1 = 0; c1 < C; ++c1) {
            for (int r2 = r1; r2 < R; ++r2) {
                for (int c2 = c1; c2 < C; ++c2) {
                    int area = (r2 - r1 + 1) * (c2 - c1 + 1);
                    if (area >= 2) {  // 최소 2칸 이상
                        Move move = std::make_tuple(r1, c1, r2, c2);
                        action_to_move[action_idx] = move;
                        move_to_action[move_to_key(r1, c1, r2, c2)] = action_idx;
                        action_idx++;
                    }
                }
            }
        }
    }
    
    // 패스 액션 추가
    Move pass_move = std::make_tuple(-1, -1, -1, -1);
    action_to_move[action_idx] = pass_move;
    move_to_action[move_to_key(-1, -1, -1, -1)] = action_idx;
}

long long GameBoard::move_to_key(int r1, int c1, int r2, int c2) const {
    // 음수 처리를 위해 offset 추가
    long long key = ((long long)(r1 + 100) << 24) | 
                    ((long long)(c1 + 100) << 16) | 
                    ((long long)(r2 + 100) << 8) | 
                    (c2 + 100);
    return key;
}

std::vector<GameBoard::Move> GameBoard::get_valid_moves() const {
    if (game_over) {
        return {};
    }
    
    std::vector<Move> valid_moves;
    
    for (int r1 = 0; r1 < R; ++r1) {
        for (int c1 = 0; c1 < C; ++c1) {
            bool skip_larger_r2 = false;
            for (int r2 = r1; r2 < R; ++r2) {
                if (skip_larger_r2) {
                    break;
                }
                for (int c2 = c1; c2 < C; ++c2) {
                    // 면적 체크
                    int area = (r2 - r1 + 1) * (c2 - c1 + 1);
                    if (area < 2) {
                        continue;
                    }
                    
                    // 합계 계산
                    int total_sum = get_box_sum(r1, c1, r2, c2);
                    
                    if (total_sum >= 10) {
                        if (total_sum == 10 && check_edges(r1, c1, r2, c2)) {
                            valid_moves.emplace_back(r1, c1, r2, c2);
                        }
                        // 같은 r2에서 더 큰 c2들은 건너뛰기
                        break;
                    }
                    
                    // 세로 한 줄(c1==c2)에서 합>=10이면 더 큰 r2들도 건너뛰기
                    if (c1 == c2 && total_sum >= 10) {
                        skip_larger_r2 = true;
                    }
                }
            }
        }
    }
    
    return valid_moves;
}

int GameBoard::get_box_sum(int r1, int c1, int r2, int c2) const {
    int total_sum = 0;
    for (int i = r1; i <= r2; ++i) {
        for (int j = c1; j <= c2; ++j) {
            if (board[i][j] > 0) {
                total_sum += board[i][j];
            }
        }
    }
    return total_sum;
}

bool GameBoard::check_edges(int r1, int c1, int r2, int c2) const {
    bool top = false, down = false, left = false, right = false;
    
    // 상단과 하단 변
    for (int j = c1; j <= c2; ++j) {
        if (board[r1][j] > 0) top = true;
        if (board[r2][j] > 0) down = true;
    }
    
    // 좌측과 우측 변
    for (int i = r1; i <= r2; ++i) {
        if (board[i][c1] > 0) left = true;
        if (board[i][c2] > 0) right = true;
    }
    
    return top && down && left && right;
}

bool GameBoard::is_valid_move(int r1, int c1, int r2, int c2) const {
    // 범위 체크
    if (!(0 <= r1 && r1 <= r2 && r2 < R && 0 <= c1 && c1 <= c2 && c2 < C)) {
        return false;
    }
    
    int area = (r2 - r1 + 1) * (c2 - c1 + 1);
    if (area < 2) {  // 최소 2칸 이상
        return false;
    }
    
    // 합이 10인지 확인
    int total_sum = get_box_sum(r1, c1, r2, c2);
    if (total_sum != 10) {
        return false;
    }
    
    // 네 변에 각각 최소 하나 이상의 버섯이 있는지 확인
    return check_edges(r1, c1, r2, c2);
}

bool GameBoard::make_move(int r1, int c1, int r2, int c2, int player) {
    if (game_over) {
        return false;
    }
    
    // 패스인 경우
    if (r1 == -1 && c1 == -1 && r2 == -1 && c2 == -1) {
        pass_count++;
        if (pass_count >= 2) {
            end_game();
        } else {
            current_player = 1 - current_player;
        }
        return true;
    }
    
    // 유효한 움직임인지 확인
    if (!is_valid_move(r1, c1, r2, c2)) {
        return false;
    }
    
    // 영역 점령
    for (int i = r1; i <= r2; ++i) {
        for (int j = c1; j <= c2; ++j) {
            board[i][j] = -(player + 1);  // -1은 플레이어 0, -2는 플레이어 1
        }
    }
    
    pass_count = 0;  // 패스 카운트 초기화
    current_player = 1 - current_player;
    
    // 더 이상 유효한 움직임이 없으면 게임 종료
    if (get_valid_moves().empty()) {
        end_game();
    }
    
    return true;
}

void GameBoard::end_game() {
    game_over = true;
    
    // 점수 계산
    int score[2] = {0, 0};
    for (const auto& row : board) {
        for (int cell : row) {
            if (cell == -1) {
                score[0]++;
            } else if (cell == -2) {
                score[1]++;
            }
        }
    }
    
    // 승자 결정
    if (score[0] > score[1]) {
        winner = 0;
    } else if (score[1] > score[0]) {
        winner = 1;
    } else {
        winner = -1;  // 무승부
    }
}

bool GameBoard::is_terminal() const {
    return game_over;
}

std::pair<int, int> GameBoard::get_score() const {
    int score[2] = {0, 0};
    for (const auto& row : board) {
        for (int cell : row) {
            if (cell == -1) {
                score[0]++;
            } else if (cell == -2) {
                score[1]++;
            }
        }
    }
    return {score[0], score[1]};
}

GameBoard GameBoard::copy() const {
    return GameBoard(*this);
}

int GameBoard::get_action_space_size() const {
    return action_to_move.size();
}

int GameBoard::encode_move(int r1, int c1, int r2, int c2) const {
    auto it = move_to_action.find(move_to_key(r1, c1, r2, c2));
    if (it != move_to_action.end()) {
        return it->second;
    }
    return -1;  // 찾을 수 없음
}

GameBoard::Move GameBoard::decode_action(int action_idx) const {
    auto it = action_to_move.find(action_idx);
    if (it != action_to_move.end()) {
        return it->second;
    }
    return std::make_tuple(-1, -1, -1, -1);  // 기본값
}

std::vector<std::vector<std::vector<float>>> GameBoard::get_state_tensor(int perspective_player) const {
    std::vector<std::vector<std::vector<float>>> state(2, 
        std::vector<std::vector<float>>(R, std::vector<float>(C, 0.0f)));
    
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            int cell = board[i][j];
            if (cell > 0) {
                // 버섯 값을 정규화 (1-9 -> 0.1-0.9)
                state[0][i][j] = cell / 10.0f;
            } else if (cell == -(perspective_player + 1)) {
                // 현재 플레이어가 점령한 칸
                state[1][i][j] = 1.0f;
            } else if (cell == -(2 - perspective_player)) {
                // 상대 플레이어가 점령한 칸
                state[1][i][j] = -1.0f;
            }
        }
    }
    
    return state;
}

float GameBoard::get_reward(int player) const {
    if (!game_over) {
        return 0.0f;
    }
    
    if (winner == player) {
        return 1.0f;
    } else if (winner == -1) {  // 무승부
        return 0.0f;
    } else {
        return -1.0f;
    }
}