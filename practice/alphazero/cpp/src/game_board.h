#pragma once

#include <vector>
#include <tuple>
#include <unordered_map>

class GameBoard {
public:
    static const int R = 10;  // 행 수
    static const int C = 17;  // 열 수
    
    using Move = std::tuple<int, int, int, int>;  // (r1, c1, r2, c2)
    
    // 생성자
    GameBoard(const std::vector<std::vector<int>>& initial_board);
    GameBoard(const GameBoard& other);  // 복사 생성자
    
    // 게임 로직
    std::vector<Move> get_valid_moves() const;
    bool make_move(int r1, int c1, int r2, int c2, int player);
    bool is_terminal() const;
    std::pair<int, int> get_score() const;
    float get_reward(int player) const;
    
    // 유틸리티
    GameBoard copy() const;
    int get_current_player() const { return current_player; }
    int get_pass_count() const { return pass_count; }
    bool is_game_over() const { return game_over; }
    int get_winner() const { return winner; }
    
    // 액션 공간 관리
    int get_action_space_size() const;
    int encode_move(int r1, int c1, int r2, int c2) const;
    Move decode_action(int action_idx) const;
    
    // 상태 텐서 생성 (Python 콜백용)
    std::vector<std::vector<std::vector<float>>> get_state_tensor(int perspective_player) const;
    
    // 보드 접근
    const std::vector<std::vector<int>>& get_board() const { return board; }

private:
    std::vector<std::vector<int>> board;
    int current_player;
    int pass_count;
    bool game_over;
    int winner;
    
    // 액션 매핑 테이블
    std::unordered_map<int, Move> action_to_move;
    std::unordered_map<long long, int> move_to_action;
    
    // 내부 헬퍼 함수들
    void build_action_mapping();
    int get_box_sum(int r1, int c1, int r2, int c2) const;
    bool check_edges(int r1, int c1, int r2, int c2) const;
    bool is_valid_move(int r1, int c1, int r2, int c2) const;
    void end_game();
    long long move_to_key(int r1, int c1, int r2, int c2) const;
};