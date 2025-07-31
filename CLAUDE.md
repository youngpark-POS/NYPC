# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a competitive programming project for NYPC (New York Programming Contest) focused on implementing AI for a strategic "Mushroom game". The game involves players selecting rectangular regions on a 10x17 grid where rectangles must sum to exactly 10 and have at least one non-zero number on their edges.

**Project Goal**: Optimize AI performance with minimal computational resources. Focus is on execution time efficiency, not game outcome comparison.

## Common Commands

### Testing and Development
- **Run game simulation**: `python practice/testing/testing_tool.py` (from root directory)  
- **Compile optimized C++ AI**: `cd practice/testing && g++ -O2 -o ab_pruning ab_pruning.cpp`
- **View game logs**: Check `practice/testing/log.txt` after running tests
- **Test specific agents**: Configure `practice/testing/setting.ini` and run testing tool

### Agent Development  
- **Run basic agent**: `python practice/agents/basic_agent.py` (for standalone testing)
- **Run MCTS agent**: `python practice/agents/mcts_agent.py` (for standalone testing)
- **Run AlphaZero agent**: `python practice/agents/alpha_agent.py [model_path]` (with trained model)

### AlphaZero Training and Usage
- **Train AlphaZero model**: `python practice/training/iterative_training.py --experiment my_model --iterations 5 --selfplay-games 20 --training-epochs 10`
- **Quick test training**: `python practice/training/iterative_training.py --experiment test --iterations 1 --selfplay-games 3 --training-epochs 2`
- **Use trained model**: Models saved in `experiments/[experiment_name]/latest_model.pth`
- **Test framework**: `python test_alphazero_framework.py` (comprehensive test suite)

### Configuration
- **Game settings**: Edit `practice/testing/setting.ini` to configure:
  - Input file path (`INPUT=./practice/testing/input.txt`)
  - Log file path (`LOG=./practice/testing/log.txt`) 
  - Player 1 executable (`EXEC1=python practice/agents/basic_agent.py`)
  - Player 2 executable (`EXEC2=python practice/agents/mcts_agent.py`)

### AlphaZero Configuration Examples
```ini
# Use trained AlphaZero model vs basic agent
EXEC1=python practice/agents/alpha_agent.py experiments/my_first_model/latest_model.pth
EXEC2=python practice/agents/basic_agent.py

# AlphaZero vs MCTS comparison
EXEC1=python practice/agents/alpha_agent.py experiments/my_model/latest_model.pth  
EXEC2=python practice/agents/mcts_agent.py

# C++ vs AlphaZero benchmark
EXEC1=practice/testing/ab_pruning
EXEC2=python practice/agents/alpha_agent.py experiments/competition_model/latest_model.pth
```

## Architecture

### Core Components
- **`practice/testing/testing_tool.py`**: Game engine that manages two-player games with process communication, timeouts, and move validation
- **`practice/core/game_board.py`**: Unified GameBoard class with move validation, generation, and neural network feature extraction
- **`practice/core/game_rules.py`**: Game constants, validation utilities, and move encoding
- **`practice/core/value_net.py`**: Neural networks (ValueNet, CombinedPolicyValueNet) for AlphaZero framework
- **`practice/agents/`**: Modular AI implementations (basic, MCTS, AlphaZero) inheriting from BaseAgent
- **`practice/mcts/`**: Monte Carlo Tree Search implementation with UCB1 selection and neural network hooks
- **`practice/training/`**: AlphaZero training pipeline (self-play, iterative learning)
- **`practice/testing/ab_pruning.cpp`**: Optimized C++ AI with alpha-beta pruning and progressive expansion
- **`practice/testing/input.txt`**: Game board configuration (10x17 grid of numbers)
- **`experiments/`**: Trained AlphaZero models and training data

### Game Engine (`practice/testing/testing_tool.py`)
The testing framework creates separate processes for each player and communicates via stdin/stdout. Protocol:
- **READY** → **INIT** → **TIME/OPP** cycles → **FINISH**
- Move validation according to game rules
- Timeout management (10 seconds per player per game)
- Turn-based game flow until no valid moves remain
- Comprehensive logging of all game actions and timing

### Modular AI Architecture (`practice/agents/`)
All agents inherit from `BaseAgent` with standard protocol communication:
- **BasicAgent**: Greedy heuristic selecting largest valid rectangles
- **PolicyAgent**: Neural network policy guidance with heuristic fallback
- **MCTSAgent**: Monte Carlo Tree Search with random rollouts and strategic pass consideration  
- **AlphaAgent**: Full AlphaZero implementation with combined policy-value network and MCTS
- **C++ Agent**: MinMax search with alpha-beta pruning (separate executable)

### Game Logic (`practice/core/`)
- **GameBoard**: Unified state management with territory tracking, undo/redo, progressive move generation
- **Move Generation**: Optimized rectangle enumeration starting from top-left corners
- **Feature Engineering**: 683-dimensional vectors + 7-channel spatial features for neural networks
- **Validation**: Comprehensive rule checking (sum=10, edge constraints, territory conflicts)

### AlphaZero Framework (`practice/training/`, `practice/core/value_net.py`)
- **ValueNet**: Neural network for position evaluation (win probability prediction)
- **CombinedPolicyValueNet**: Unified network for both policy and value prediction (memory efficient)
- **Self-play Generator**: Automated game generation using current model + MCTS
- **Iterative Training**: Expert Iteration loop with model improvement
- **MCTS Integration**: Policy priors + value evaluation for enhanced search

## Development Notes

### Architecture Patterns
- **Modular Design**: Clear separation between game logic (`core/`), AI implementations (`agents/`), search algorithms (`mcts/`), and testing framework
- **Protocol Communication**: Standard stdin/stdout interface allows mixing Python and C++ implementations
- **Inheritance Hierarchy**: All Python agents inherit from `BaseAgent` for consistent protocol handling
- **Performance Focus**: Optimized move generation, progressive expansion, and time-bounded search

### Adding New Agents
1. Create new file in `practice/agents/` inheriting from `BaseAgent`
2. Implement `get_move(board, time_limit)` method
3. Update `practice/testing/setting.ini` to test against other agents
4. Use `GameBoard` class for state management and move validation

### Research Features  
- **Neural Network Ready**: GameBoard provides 683-dimensional feature vectors
- **MCTS Framework**: Extensible tree search with simulation and neural evaluation hooks
- **Comprehensive Logging**: All moves, timing, and game states recorded for analysis
- **Performance Benchmarking**: Focus on execution time optimization rather than win rates

## Performance Optimization Guidelines
- **Primary Metric**: Execution time efficiency (check time values in log.txt)
- **Secondary Consideration**: Memory usage and algorithmic complexity
- **Not Measured**: Game win/loss ratios against simple baseline AI
- **Testing Context**: Performance comparisons against simple example code are not meaningful for optimization purposes
- **Focus**: Minimize computational resources while maintaining reasonable move quality