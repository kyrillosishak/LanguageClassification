#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.training.trainer import MultiGPULanguageClassificationTrainer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_with_args(config, args):
    """Update config with command line arguments"""
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.max_length:
        config['model']['max_length'] = args.max_length
    
    # Resume settings
    if args.resume:
        config['resume']['enabled'] = True
        if args.checkpoint_path:
            config['resume']['checkpoint_path'] = args.checkpoint_path
    
    if args.from_scratch:
        config['resume']['enabled'] = False
        config['resume']['from_scratch'] = True
    
    # Experiment name
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Language Classification Model')
    
    # Data paths
    parser.add_argument('--x_text_path', type=str, required=True,
                       help='Path to text data file')
    parser.add_argument('--y_labels_path', type=str, required=True,
                       help='Path to labels file')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--model_name', type=str,
                       help='Pre-trained model name')
    parser.add_argument('--max_length', type=int,
                       help='Maximum sequence length')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Specific checkpoint path to resume from')
    parser.add_argument('--from_scratch', action='store_true',
                       help='Train from scratch (ignore existing checkpoints)')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str,
                       help='Name for the experiment')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only evaluate')
    
    args = parser.parse_args()

    # Load and update config
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    trainer = MultiGPULanguageClassificationTrainer(config=config)
    train_loader, val_loader, test_loader = trainer.prepare_data(
        args.x_text_path,
        args.y_labels_path
    )

    if not args.skip_training:
        trainer.train(train_loader, val_loader)

    if args.evaluate:
        trainer.evaluate(test_loader)


if __name__ == '__main__':
    main()

