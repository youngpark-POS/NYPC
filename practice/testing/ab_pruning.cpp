#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <algorithm>

using namespace std;

const int GAMETREE_SEARCH_DEPTH = 5;
const int BOARD_ROW = 10;
const int BOARD_COLUMN = 17;
const vector<int> MOVE_PASS = {-1, -1, -1, -1};

class Game {
public:
    vector<vector<int>> board;
    vector<vector<int>> territory_board;
    bool opponent_last_passed;
    int turn_count;

    Game(const vector<vector<int>>& b) : board(b), territory_board(BOARD_ROW, vector<int>(BOARD_COLUMN, 0)), opponent_last_passed(false), turn_count(0) {}

    int calculate_board_value() {
        int total = 0;
        for (const auto& row: territory_board) {
            for (int v: row) total += v;
        }
        return total;
    }

    struct Move {
        int r1, c1, r2, c2;
        int score;
        
        Move(int r1, int c1, int r2, int c2, int score) 
            : r1(r1), c1(c1), r2(r2), c2(c2), score(score) {}
    };

    int evaluateMove(int r1, int c1, int r2, int c2, bool is_max_player) {
        int size = (r2 - r1 + 1) * (c2 - c1 + 1);
        int center_r = (r1 + r2) / 2;
        int center_c = (c1 + c2) / 2;
        int center_distance = abs(center_r - BOARD_ROW/2) + abs(center_c - BOARD_COLUMN/2);
        
        int immediate_value = 0;
        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                immediate_value += (is_max_player ? 1 : -1);
            }
        }
        
        return immediate_value * 1000 - size * 10 - center_distance;
    }

    vector<Move> generateOrderedMoves(bool is_max_player) {
        vector<Move> moves;
        
        for (int r1 = 0; r1 < BOARD_ROW; ++r1) {
            for (int r2 = r1; r2 < BOARD_ROW; ++r2) {
                for (int c1 = 0; c1 < BOARD_COLUMN; ++c1) {
                    for (int c2 = c1; c2 < BOARD_COLUMN; ++c2) {
                        if (isValid(r1, c1, r2, c2)) {
                            int score = evaluateMove(r1, c1, r2, c2, is_max_player);
                            moves.emplace_back(r1, c1, r2, c2, score);
                        }
                    }
                }
            }
        }
        
        if (is_max_player) {
            sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
                return a.score > b.score;
            });
        } else {
            sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
                return a.score < b.score;
            });
        }
        
        return moves;
    }

    int getSearchDepth() {
        return 4;  // 고정 depth 4로 속도 개선
    }

    // returns {best_value, r1, c1, r2, c2}
    pair<int, vector<int>> simulate(int alpha, int beta, bool is_max_player, int depth, bool opponent_passed = false) {
        vector<vector<int>> original_board = board;
        vector<vector<int>> original_territory_board = territory_board;
        vector<int> best_move = MOVE_PASS;
        int best_value = is_max_player ? numeric_limits<int>::min() : numeric_limits<int>::max();
        bool is_terminal = true;

        if (depth == getSearchDepth()) {
            return {calculate_board_value(), best_move};
        }

        // Consider pass move strategically
        int current_board_value = calculate_board_value();
        
        // Only consider pass when winning
        if ((is_max_player && current_board_value > 0) || (!is_max_player && current_board_value < 0)) {
            is_terminal = false;
            auto rv = simulate(alpha, beta, !is_max_player, depth + 1, false);
            int state_value = rv.first;
            
            if (is_max_player && state_value > best_value) {
                best_value = state_value;
                best_move = MOVE_PASS;
                alpha = max(alpha, state_value);
            }
            if (!is_max_player && state_value < best_value) {
                best_value = state_value;
                best_move = MOVE_PASS;
                beta = min(beta, state_value);
            }
            
            if (alpha >= beta) {
                return {best_value, best_move};
            }
        }


        for (int r1 = 0; r1 < BOARD_ROW; ++r1) {
            for (int r2 = r1; r2 < BOARD_ROW; ++r2) {
                for (int c1 = 0; c1 < BOARD_COLUMN; ++c1) {
                    for (int c2 = c1; c2 < BOARD_COLUMN; ++c2) {
                        if (!isValid(r1, c1, r2, c2))
                            continue;

                        is_terminal = false;

                        updateMove(r1, c1, r2, c2, is_max_player);
                        auto rv = simulate(alpha, beta, !is_max_player, depth + 1, false);
                        int state_value = rv.first;
                        auto _best_move = rv.second;
                        
                        if (is_max_player && state_value > best_value) {
                            best_value = state_value;
                            best_move = {r1, c1, r2, c2};
                            alpha = max(alpha, state_value);
                        }
                        if (!is_max_player && state_value < best_value) {
                            best_value = state_value;
                            best_move = {r1, c1, r2, c2};
                            beta = min(beta, state_value);
                        }

                        restoreMove(r1, c1, r2, c2, original_board, original_territory_board);

                        if (alpha >= beta) {
                            return {best_value, best_move};
                        }
                    }
                }
            }
        }

        if (is_terminal) {
            return {calculate_board_value(), MOVE_PASS};
        } else {
            return {best_value, best_move};
        }
    }

    // 사각형 (r1, c1) ~ (r2, c2)이 유효한지 검사
    bool isValid(int r1, int c1, int r2, int c2) {
        int sums = 0;
        bool r1fit = false, c1fit = false, r2fit = false, c2fit = false;

        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                if (board[r][c] != 0) {
                    sums += board[r][c];
                    if (sums > 10)
                        return false;
                    if (r == r1) r1fit = true;
                    if (r == r2) r2fit = true;
                    if (c == c1) c1fit = true;
                    if (c == c2) c2fit = true;
                }
            }
        }
        return (sums == 10 && r1fit && r2fit && c1fit && c2fit);
    }

    vector<int> calculateMove(int _myTime, int _oppTime) {
        // If opponent passed and we're winning, pass to end the game
        if (opponent_last_passed) {
            int current_score = calculate_board_value();
            if (current_score > 0) {  // Only when we're winning
                return MOVE_PASS;
            }
        }
        
        turn_count++;
        auto rv = simulate(numeric_limits<int>::min(), numeric_limits<int>::max(), true, 0, false);
        return rv.second;
    }

    void updateOpponentAction(const vector<int>& action, int _time) {
        opponent_last_passed = (action[0] == -1 && action[1] == -1 && action[2] == -1 && action[3] == -1);
        updateMove(action[0], action[1], action[2], action[3], false);
    }

    void updateMove(int r1, int c1, int r2, int c2, bool is_max_player) {
        if (r1 == -1 && c1 == -1 && r2 == -1 && c2 == -1)
            return;
        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                board[r][c] = 0;
                territory_board[r][c] = is_max_player ? 1 : -1;
            }
        }
    }

    void restoreMove(int r1, int c1, int r2, int c2, const vector<vector<int>>& original_board, const vector<vector<int>>& original_territory_board) {
        if (r1 == -1 && c1 == -1 && r2 == -1 && c2 == -1)
            return;
        for (int r = r1; r <= r2; ++r) {
            for (int c = c1; c <= c2; ++c) {
                board[r][c] = original_board[r][c];
                territory_board[r][c] = original_territory_board[r][c];
            }
        }
    }
};

// 전역 변수
bool first;
Game* game = nullptr;

// 입력 처리
vector<string> split(const string& line) {
    vector<string> tokens;
    istringstream iss(line);
    string token;
    while (iss >> token) tokens.push_back(token);
    return tokens;
}

vector<int> parse_board_row(const string& s) {
    vector<int> row;
    for (char ch : s) {
        if ('0' <= ch && ch <= '9') row.push_back(ch - '0');
    }
    return row;
}

int main() {
    string line;
    while (getline(cin, line)) {
        vector<string> tokens = split(line);
        if (tokens.empty()) continue;

        string command = tokens[0];

        if (command == "READY") {
            string turn = tokens[1];
            first = (turn == "FIRST");
            cout << "OK" << endl;
            cout.flush();
            continue;
        }

        if (command == "INIT") {
            vector<vector<int>> board;
            for (size_t i = 1; i < tokens.size(); ++i)
                board.push_back(parse_board_row(tokens[i]));
            if (game) delete game;
            game = new Game(board);
            continue;
        }

        if (command == "TIME") {
            int myTime = stoi(tokens[1]);
            int oppTime = stoi(tokens[2]);
            vector<int> ret = game->calculateMove(myTime, oppTime);
            game->updateMove(ret[0], ret[1], ret[2], ret[3], true);
            for (size_t i = 0; i < ret.size(); ++i) 
                cout << ret[i] << (i + 1 == ret.size() ? "\n" : " ");
            cout.flush();
            continue;
        }

        if (command == "OPP") {
            vector<int> action(4);
            for (int i = 0; i < 4; ++i)
                action[i] = stoi(tokens[i+1]);
            int time = stoi(tokens[5]);
            game->updateOpponentAction(action, time);
            continue;
        }

        if (command == "FINISH") {
            break;
        }

        throw runtime_error("Invalid command: " + command);
    }
    if (game) delete game;
    return 0;
}