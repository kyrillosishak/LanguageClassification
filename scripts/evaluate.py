#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.evaluator import MultiGPULanguageClassificationEvaluator


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_config_with_args(config, args):
    if args.checkpoint_path:
        config['resume']['checkpoint_path'] = args.checkpoint_path
    if args.model_name:
        config['model']['name'] = args.model_name
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    
    parser.add_argument('--x_text_path', type=str, required=True,
                        help='Path to text data file')
    parser.add_argument('--y_labels_path', type=str, required=True,
                        help='Path to labels file')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--checkpoint_path', type=str,
                        help='Checkpoint path to load model from')
    parser.add_argument('--model_name', type=str,
                        help='Pre-trained model name to override config')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    evaluator = MultiGPULanguageClassificationEvaluator(
        config=config,
        x_text_path=args.x_text_path,
        y_labels_path=args.y_labels_path
    )
    evaluator.evaluate()


if __name__ == '__main__':
    main()
