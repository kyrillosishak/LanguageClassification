"""
Logging utilities for training and evaluation
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Comprehensive logging for training process
    """
    
    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self.setup_file_logger()
        
        # Setup TensorBoard logger
        self.tb_writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        
    def setup_file_logger(self):
        """Setup file logger"""
        self.logger = logging.getLogger('training_logger')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.experiment_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config):
        """Log configuration"""
        config_file = self.experiment_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info("Configuration saved")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value}")
            else:
                self.logger.info(f"{key}: {value}")
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log epoch start"""
        self.logger.info(f"Starting Epoch {epoch+1}/{total_epochs}")
        self.current_epoch = epoch
    
    def log_batch_metrics(self, batch_idx, total_batches, loss, accuracy, learning_rate=None):
        """Log batch-level metrics"""
        if batch_idx % 100 == 0:  # Log every 100 batches
            self.logger.info(
                f"Batch {batch_idx}/{total_batches} - "
                f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )
    
    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                         learning_rate, epoch_time):
        """Log epoch-level metrics"""
        # Store metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_accuracy'].append(val_acc)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['epoch_times'].append(epoch_time)
        
        # Log to file and console
        self.logger.info(
            f"Epoch {epoch+1} Results - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {learning_rate:.2e}, Time: {epoch_time:.2f}s"
        )
        
        # Log to TensorBoard
        self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        self.tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.tb_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.tb_writer.add_scalar('Learning_Rate', learning_rate, epoch)
        self.tb_writer.add_scalar('Epoch_Time', epoch_time, epoch)
        
        # Save metrics to JSON
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.experiment_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_test_results(self, test_accuracy, classification_report=None):
        """Log test results"""
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        if classification_report:
            self.logger.info("Classification Report:")
            self.logger.info(str(classification_report))
            
            # Save classification report
            report_file = self.experiment_dir / 'classification_report.json'
            with open(report_file, 'w') as f:
                json.dump(classification_report, f, indent=2)
    
    def log_model_save(self, epoch, checkpoint_path, is_best=False):
        """Log model save"""
        if is_best:
            self.logger.info(f"New best model saved at epoch {epoch+1}: {checkpoint_path}")
        else:
            self.logger.info(f"Model checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
    
    def log_training_complete(self, total_time, best_val_acc):
        """Log training completion"""
        self.logger.info(f"Training completed!")
        self.logger.info(f"Total training time: {total_time:.2f}s")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    def close(self):
        """Close logger"""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
        
        # Remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints
    """
    
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_score = 0.0
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, 
                       label_encoder, config, is_best=False):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Training metrics
            label_encoder: Label encoder
            config: Model configuration
            is_best: Whether this is the best model
        """
        # Handle DataParallel models
        model_to_save = model.module if hasattr(model, 'module') else model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'label_encoder': label_encoder,
            'config': config,
            'best_score': self.best_score
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_score = metrics.get('val_accuracy', 0.0)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Loaded checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pth'
        return best_path if best_path.exists() else None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            checkpoint.unlink()