#!/usr/bin/env python3
"""
Iterative Training for AlphaZero-style learning

Implements the Expert Iteration loop:
1. Self-play data generation using current model
2. Train combined policy-value network on collected data
3. Evaluate new model and update if better
4. Repeat for multiple iterations
"""

import sys
import os
import pickle
import time
import json
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

from core.game_board import GameBoard
from core.alphazero_net import create_alphazero_net, AlphaZeroNet
from training.self_play import SelfPlayGenerator, SelfPlayDataPoint, load_board_config

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length valid_moves and policies."""
    # Extract individual components
    features = torch.stack([item['features'] for item in batch])
    value_targets = torch.stack([item['value_target'] for item in batch])
    
    # Keep policy targets and valid moves as lists (variable length)
    policy_targets = [item['policy_target'] for item in batch]
    valid_moves_batch = [item['valid_moves'] for item in batch]
    
    return {
        'features': features,
        'policy_target': policy_targets,  # List of tensors
        'value_target': value_targets,
        'valid_moves': valid_moves_batch  # List of move lists
    }

class SelfPlayDataset(Dataset):
    """Dataset for training on self-play data."""
    
    def __init__(self, data_points: List[SelfPlayDataPoint]):
        self.data_points = []
        
        # Filter and validate data points
        for dp in data_points:
            if len(dp.valid_moves) > 0 and dp.mcts_policy.sum() > 0:
                self.data_points.append(dp)
        
        print(f"SelfPlayDataset: {len(self.data_points)} valid training examples")
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        dp = self.data_points[idx]
        
        # Convert features to tensor format
        features = torch.from_numpy(dp.board_features).permute(2, 0, 1).float()  # (7, 10, 17)
        
        # Policy target (MCTS visit distribution)
        policy_target = torch.from_numpy(dp.mcts_policy).float()
        
        # Value target (game outcome)
        value_target = torch.tensor([dp.game_result], dtype=torch.float32)
        
        return {
            'features': features,
            'policy_target': policy_target,
            'value_target': value_target,
            'valid_moves': dp.valid_moves
        }

class IterativeTrainer:
    """Manages the iterative training process."""
    
    def __init__(self, 
                 experiment_name: str = "alphazero_experiment",
                 device: str = 'cpu'):
        """
        Initialize iterative trainer.
        
        Args:
            experiment_name: Name for this training experiment
            device: Device to train on ('cpu' or 'cuda')
        """
        self.experiment_name = experiment_name
        self.device = device
        
        # Create experiment directory
        self.exp_dir = Path(f"experiments/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = []
        
        print(f"IterativeTrainer initialized:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Device: {device}")
        print(f"  Directory: {self.exp_dir}")
    
    def train_on_self_play_data(self, 
                               model: AlphaZeroNet,
                               data_points: List[SelfPlayDataPoint],
                               num_epochs: int = 20,
                               batch_size: int = 32,
                               learning_rate: float = 0.001,
                               weight_decay: float = 1e-4) -> AlphaZeroNet:
        """
        Train the model on self-play data.
        
        Args:
            model: Model to train
            data_points: Self-play training data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            
        Returns:
            Trained model
        """
        print(f"\nTraining model on {len(data_points)} data points...")
        
        # Create dataset and dataloader
        dataset = SelfPlayDataset(data_points)
        if len(dataset) == 0:
            print("No valid training data!")
            return model
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
        
        # Setup training
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss functions
        policy_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()
        
        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            num_batches = 0
            
            for batch in dataloader:
                features = batch['features'].to(self.device)  # (batch_size, 7, 10, 17)
                policy_targets = batch['policy_target']  # List of tensors (variable length)
                value_targets = batch['value_target'].to(self.device)  # (batch_size, 1)
                valid_moves_batch = batch['valid_moves']
                
                optimizer.zero_grad()
                
                # Forward pass
                batch_losses = []
                batch_policy_losses = []
                batch_value_losses = []
                
                for i in range(len(features)):
                    sample_features = features[i:i+1]
                    sample_policy_target = policy_targets[i].to(self.device)  # Move to device
                    sample_value_target = value_targets[i]
                    sample_valid_moves = valid_moves_batch[i]
                    
                    # Get model predictions
                    policy_scores, value_pred = model(sample_features, sample_valid_moves)
                    
                    # Policy loss (cross-entropy between MCTS policy and model policy)
                    if policy_scores is not None and len(sample_policy_target) == policy_scores.shape[1]:
                        policy_probs = F.softmax(policy_scores, dim=1)
                        # Use KL divergence loss instead of cross-entropy
                        policy_loss = F.kl_div(
                            F.log_softmax(policy_scores, dim=1),
                            sample_policy_target.unsqueeze(0),
                            reduction='batchmean'
                        )
                    else:
                        policy_loss = torch.tensor(0.0, device=self.device)
                    
                    # Value loss (MSE between game outcome and predicted value)
                    value_loss = value_loss_fn(value_pred, sample_value_target.unsqueeze(0))
                    
                    # Combined loss
                    total_sample_loss = policy_loss + value_loss
                    
                    batch_losses.append(total_sample_loss)
                    batch_policy_losses.append(policy_loss)
                    batch_value_losses.append(value_loss)
                
                if batch_losses:
                    batch_loss = torch.stack(batch_losses).mean()
                    batch_policy_loss = torch.stack(batch_policy_losses).mean()
                    batch_value_loss = torch.stack(batch_value_losses).mean()
                    
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                    policy_loss_sum += batch_policy_loss.item()
                    value_loss_sum += batch_value_loss.item()
                    num_batches += 1
            
            # Print epoch results
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                avg_policy_loss = policy_loss_sum / num_batches
                avg_value_loss = value_loss_sum / num_batches
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={avg_loss:.4f} (Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f})")
        
        print("Training completed!")
        return model
    
    def evaluate_model(self, 
                      model: AlphaZeroNet,
                      board_data: List[List[int]],
                      num_games: int = 20) -> Dict[str, float]:
        """
        Evaluate model by self-play games.
        
        Args:
            model: Model to evaluate
            board_data: Initial board configuration
            num_games: Number of evaluation games
            
        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating model with {num_games} games...")
        
        generator = SelfPlayGenerator(
            model=model,
            mcts_simulations=30,   # Fewer sims for faster evaluation
            mcts_time=0.3,         # Faster evaluation
            temperature=0.1        # Low temperature for evaluation
        )
        
        games_data = []
        total_turns = 0
        game_lengths = []
        
        for i in range(num_games):
            try:
                game_data = generator.play_game(board_data, verbose=False)
                games_data.extend(game_data)
                
                game_length = len(game_data)
                game_lengths.append(game_length)
                total_turns += game_length
                
            except Exception as e:
                print(f"Evaluation game {i} failed: {e}")
                continue
        
        # Calculate metrics
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        total_games_played = len(game_lengths)
        
        # Win rate distribution
        win_count = sum(1 for dp in games_data if dp.game_result > 0)
        loss_count = sum(1 for dp in games_data if dp.game_result < 0)
        tie_count = sum(1 for dp in games_data if dp.game_result == 0)
        
        metrics = {
            'games_played': total_games_played,
            'avg_game_length': avg_game_length,
            'total_positions': len(games_data),
            'win_rate': win_count / len(games_data) if games_data else 0,
            'loss_rate': loss_count / len(games_data) if games_data else 0,
            'tie_rate': tie_count / len(games_data) if games_data else 0
        }
        
        print(f"Evaluation results:")
        print(f"  Games played: {metrics['games_played']}")
        print(f"  Avg game length: {metrics['avg_game_length']:.1f}")
        print(f"  Win/Loss/Tie rates: {metrics['win_rate']:.3f}/{metrics['loss_rate']:.3f}/{metrics['tie_rate']:.3f}")
        
        return metrics
    
    def save_model(self, model: AlphaZeroNet, iteration: int, metrics: Dict[str, Any] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': 'combined',
            'iteration': iteration,
            'metrics': metrics or {},
            'timestamp': time.time()
        }
        
        model_path = self.exp_dir / f"model_iter_{iteration:03d}.pth"
        torch.save(checkpoint, model_path)
        
        # Also save as latest
        latest_path = self.exp_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"Model saved: {model_path}")
    
    def load_model(self, iteration: int = None) -> AlphaZeroNet:
        """Load model checkpoint."""
        if iteration is None:
            model_path = self.exp_dir / "latest_model.pth"
        else:
            model_path = self.exp_dir / f"model_iter_{iteration:03d}.pth"
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None
        
        model = create_alphazero_net(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded: {model_path}")
        return model
    
    def run_iteration(self, 
                     iteration: int,
                     model: AlphaZeroNet,
                     board_data: List[List[int]],
                     selfplay_games: int = 50,
                     training_epochs: int = 20,
                     evaluation_games: int = 20) -> AlphaZeroNet:
        """
        Run a single iteration of the training loop.
        
        Args:
            iteration: Iteration number
            model: Current model
            board_data: Board configuration
            selfplay_games: Number of self-play games
            training_epochs: Number of training epochs
            evaluation_games: Number of evaluation games
            
        Returns:
            Updated model
        """
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Step 1: Generate self-play data
        print(f"Step 1: Generating {selfplay_games} self-play games...")
        generator = SelfPlayGenerator(
            model=model,
            mcts_simulations=400, 
            mcts_time=2.0,
            temperature=1.0,
            temperature_threshold=10
        )
        
        self_play_data = generator.generate_self_play_data(
            board_data=board_data,
            num_games=selfplay_games,
            verbose=True
        )
        
        # Save self-play data
        data_path = self.exp_dir / f"selfplay_data_iter_{iteration:03d}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(self_play_data, f)
        print(f"Self-play data saved: {data_path}")
        
        # Step 2: Train model
        print(f"Step 2: Training model for {training_epochs} epochs...")
        trained_model = self.train_on_self_play_data(
            model=model,
            data_points=self_play_data,
            num_epochs=training_epochs,
            batch_size=16,
            learning_rate=0.001
        )
        
        # Step 3: Evaluate model
        print(f"Step 3: Evaluating model...")
        metrics = self.evaluate_model(
            model=trained_model,
            board_data=board_data,
            num_games=evaluation_games
        )
        
        # Step 4: Save model and record metrics
        self.save_model(trained_model, iteration, metrics)
        
        # Update training history
        iteration_record = {
            'iteration': iteration,
            'selfplay_games': selfplay_games,
            'training_data_points': len(self_play_data),
            'training_epochs': training_epochs,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.training_history.append(iteration_record)
        
        # Save training history
        history_path = self.exp_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Iteration {iteration} completed!")
        return trained_model
    
    def run_full_training(self,
                         num_iterations: int = 10,
                         selfplay_games: int = 50,
                         training_epochs: int = 20,
                         evaluation_games: int = 20,
                         resume_from: int = None):
        """
        Run the full iterative training process.
        
        Args:
            num_iterations: Number of iterations to run
            selfplay_games: Self-play games per iteration
            training_epochs: Training epochs per iteration
            evaluation_games: Evaluation games per iteration
            resume_from: Iteration to resume from (None for new training)
        """
        print(f"Starting iterative training:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Self-play games per iteration: {selfplay_games}")
        print(f"  Training epochs per iteration: {training_epochs}")
        
        # Load board configuration
        board_data = load_board_config()
        
        # Initialize or load model
        if resume_from is not None:
            model = self.load_model(resume_from)
            if model is None:
                print(f"Could not load model from iteration {resume_from}, starting fresh")
                model = create_alphazero_net(self.device)
            start_iteration = resume_from + 1
        else:
            model = create_alphazero_net(self.device)
            start_iteration = 0
        
        # Run iterations
        for iteration in range(start_iteration, num_iterations):
            try:
                model = self.run_iteration(
                    iteration=iteration,
                    model=model,
                    board_data=board_data,
                    selfplay_games=selfplay_games,
                    training_epochs=training_epochs,
                    evaluation_games=evaluation_games
                )
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nTraining completed! Final model saved as latest_model.pth")
        print(f"Experiment data saved in: {self.exp_dir}")

def main():
    """Main function for iterative training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaZero-style iterative training')
    parser.add_argument('--experiment', type=str, default='alphazero_nypc', 
                       help='Experiment name')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of training iterations')
    parser.add_argument('--selfplay-games', type=int, default=30,
                       help='Self-play games per iteration')
    parser.add_argument('--training-epochs', type=int, default=15,
                       help='Training epochs per iteration')
    parser.add_argument('--evaluation-games', type=int, default=10,
                       help='Evaluation games per iteration')
    parser.add_argument('--resume-from', type=int, 
                       help='Resume from specific iteration')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = IterativeTrainer(
        experiment_name=args.experiment,
        device=args.device
    )
    
    # Run training
    trainer.run_full_training(
        num_iterations=args.iterations,
        selfplay_games=args.selfplay_games,
        training_epochs=args.training_epochs,
        evaluation_games=args.evaluation_games,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main()