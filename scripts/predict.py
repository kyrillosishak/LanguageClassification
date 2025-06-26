#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference.predictor import MultiGPULanguageClassificationPredictor


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
    parser = argparse.ArgumentParser(description='Run Prediction')
    
    parser.add_argument('--input_text_path', type=str, required=True,
                        help='Path to input text file (unlabeled)')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--checkpoint_path', type=str,
                        help='Checkpoint path to load model from')
    parser.add_argument('--model_name', type=str,
                        help='Pre-trained model name to override config')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to write predictions to')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    predictor = MultiGPULanguageClassificationPredictor(
        config=config,
        input_text_path=args.input_text_path
    )
    predictions = predictor.predict()
    
    # Save predictions
    with open(args.output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")


if __name__ == '__main__':
    main()
