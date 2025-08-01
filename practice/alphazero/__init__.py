"""
AlphaZero Implementation for Mushroom Game
==========================================

This package contains a complete AlphaZero implementation for the NYPC Mushroom game.

Key Components:
- GameBoard: Game state management and rule validation
- AlphaZeroNet: Neural network with residual blocks and policy/value heads  
- MCTS: Monte Carlo Tree Search with neural network integration
- SelfPlayGenerator: Self-play data generation for training
- TrainingManager: Training pipeline and model management
- AlphaZeroAgent: Competition-ready agent following sample_code.py protocol

Usage:
    # Quick test
    python main_training.py --test
    
    # Training
    python main_training.py --iterations 5 --selfplay-games 20 --training-epochs 10
    
    # Run agent
    python alphazero_agent.py [model_path]
"""

from .game_board import GameBoard
from .neural_network import AlphaZeroNet, AlphaZeroTrainer
from .mcts import MCTS, MCTSNode
from .self_play import SelfPlayGenerator, SelfPlayData
from .training import TrainingManager
from .alphazero_agent import AlphaZeroAgent

__version__ = "1.0.0"
__author__ = "AlphaZero Implementation"

__all__ = [
    'GameBoard',
    'AlphaZeroNet', 
    'AlphaZeroTrainer',
    'MCTS',
    'MCTSNode', 
    'SelfPlayGenerator',
    'SelfPlayData',
    'TrainingManager',
    'AlphaZeroAgent'
]