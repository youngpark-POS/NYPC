#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game_board.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_game_board, m) {
    m.doc() = "Fast C++ GameBoard implementation for AlphaZero";
    
    // GameBoard 클래스 바인딩
    py::class_<GameBoard>(m, "GameBoard")
        .def(py::init<const std::vector<std::vector<int>>&>())
        .def("get_valid_moves", &GameBoard::get_valid_moves)
        .def("make_move", &GameBoard::make_move)
        .def("is_terminal", &GameBoard::is_terminal)
        .def("get_score", &GameBoard::get_score)
        .def("get_reward", &GameBoard::get_reward)
        .def("copy", &GameBoard::copy)
        .def("get_current_player", &GameBoard::get_current_player)
        .def("get_pass_count", &GameBoard::get_pass_count)
        .def("is_game_over", &GameBoard::is_game_over)
        .def("get_winner", &GameBoard::get_winner)
        .def("get_action_space_size", &GameBoard::get_action_space_size)
        .def("encode_move", &GameBoard::encode_move)
        .def("decode_action", &GameBoard::decode_action)
        .def("get_state_tensor", &GameBoard::get_state_tensor)
        .def("get_board", &GameBoard::get_board, py::return_value_policy::reference_internal);
}