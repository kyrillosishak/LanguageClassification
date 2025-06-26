import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

from ..models.classifier import create_model, get_tokenizer
from ..data.dataset import DataManager
from ..utils.logger import TrainingLogger, CheckpointManager
from ..utils.metrics import MetricsCalculator


class MultiGPULanguageClassificationTrainer:
    """
    Enhanced trainer with checkpoint support and resume functionality
    """
    
    def __init__(self, config):
        self.config = config
        self.setup_device()
        
        # Initialize components
        self.tokenizer = get_tokenizer(config['model']['name'])
        self.data_manager = DataManager(self.tokenizer, config['model']['max_length'])
        
        # Initialize logging and checkpointing
        self.logger = TrainingLogger(
            config['paths']['log_dir'],
            experiment_name=config.get('experiment_name')
        )
        self.checkpoint_manager = CheckpointManager(
            config['paths']['checkpoint_dir'],
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.best_val_acc = 0.0
        
        # Log configuration
        self.logger.log_config(config)
        
    def setup_device(self):
        """Setup device configuration"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.num_gpus = torch.cuda.device_count()
            print(f"Found {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            if self.num_gpus > 1 and self.config['system']['use_multi_gpu']:
                print(f"Using {self.num_gpus} GPUs with DataParallel")
                self.multi_gpu = True
            else:
                print("Using single GPU")
                self.multi_gpu = False
        else:
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.multi_gpu = False
            print("Using CPU")
    
    def prepare_data(self, x_text_path, y_labels_path):
        """Prepare datasets and data loaders"""
        # Load data
        texts, labels = self.data_manager.load_data(x_text_path, y_labels_path)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, num_classes, label_encoder = \
            self.data_manager.prepare_data(
                texts, labels,
                test_size=self.config['data']['test_size'],
                val_size=self.config['data']['val_size'],
                random_state=self.config['data']['random_state']
            )
        
        # Store for later use
        self.num_classes = num_classes
        self.label_encoder = label_encoder
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.data_manager.create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            multi_gpu=self.multi_gpu
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """Create model with multi-GPU support"""
        self.model = create_model(
            model_name=self.config['model']['name'],
            num_classes=self.num_classes,
            dropout_rate=self.config['model']['dropout_rate'],
            device=self.device,
            multi_gpu=self.multi_gpu
        )
        return self.model
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training'].get('weight_decay', 0.01))
        )
        
        total_steps = len(train_loader) * self.config['training']['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        return self.optimizer, self.scheduler
    
    def load_checkpoint_if_exists(self):
        """Load checkpoint if resume is enabled"""
        if not self.config['resume']['enabled']:
            return False
        
        checkpoint_path = None
        
        # Check for specific checkpoint path
        if self.config['resume']['checkpoint_path']:
            checkpoint_path = self.config['resume']['checkpoint_path']
        else:
            # Look for latest checkpoint
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = self.checkpoint_manager.load_checkpoint(
                checkpoint_path, self.model, self.optimizer, self.scheduler
            )
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_score', 0.0)
            self.label_encoder = checkpoint['label_encoder']
            
            print(f"Resumed from epoch {self.start_epoch}")
            print(f"Best validation accuracy so far: {self.best_val_acc:.4f}")
            return True
        
        return False
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config['training']['gradient_clipping']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct_predictions / total_samples
            current_lr = self.scheduler.get_last_lr()[0]
            
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log batch metrics periodically
            self.logger.log_batch_metrics(
                batch_idx, len(train_loader), current_loss, current_acc, current_lr
            )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        learning_rate = self.scheduler.get_last_lr()[0]
        
        return avg_loss, accuracy, learning_rate
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        val_pbar = tqdm(val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                current_acc = correct_predictions / total_samples
                val_pbar.set_postfix({'acc': f'{current_acc:.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Main training loop with checkpointing"""
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(train_loader)
        
        # Load checkpoint if resuming
        self.load_checkpoint_if_exists()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            self.logger.log_epoch_start(epoch, self.config['training']['epochs'])
            
            # Training
            train_loss, train_acc, learning_rate = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            self.logger.log_epoch_metrics(
                epoch, train_loss, train_acc, val_loss, val_acc, 
                learning_rate, epoch_time
            )
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': learning_rate
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, metrics,
                self.label_encoder, self.config, is_best
            )
            
            self.logger.log_model_save(epoch, checkpoint_path, is_best)
        
        total_time = time.time() - start_time
        self.logger.log_training_complete(total_time, self.best_val_acc)
        
        return self.model
    
    def evaluate(self, test_loader, generate_report=True):
        """Evaluate the model on test set"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Load best model for evaluation
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint and os.path.exists(best_checkpoint):
            print("Loading best model for evaluation...")
            self.checkpoint_manager.load_checkpoint(best_checkpoint, self.model)
        
        test_loss, test_accuracy = self.validate(test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        results = {'test_accuracy': test_accuracy, 'test_loss': test_loss}
        
        if generate_report:
            # Generate detailed classification report
            report = self.generate_classification_report(test_loader)
            results['classification_report'] = report
            
            # Generate detailed predictions
            detailed_results = self.get_detailed_predictions(test_loader)
            results.update(detailed_results)
        
        # Log results
        self.logger.log_test_results(test_accuracy, results.get('classification_report'))
        
        return results
    
    def generate_classification_report(self, data_loader):
        """Generate detailed classification report"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        eval_pbar = tqdm(data_loader, desc='Generating Classification Report', leave=True)
        
        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        language_names = self.label_encoder.classes_
        
        report_dict = classification_report(
            all_labels, 
            all_predictions, 
            target_names=language_names,
            output_dict=True,
            zero_division=0
        )
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*60}")
        
        report_str = classification_report(
            all_labels, 
            all_predictions, 
            target_names=language_names,
            digits=4,
            zero_division=0
        )
        print(report_str)
        
        return report_dict
    
    def get_detailed_predictions(self, data_loader, save_errors=True):
        """Get detailed predictions with confidence scores"""
        self.model.eval()
        results = {
            'predictions': [],
            'true_labels': [],
            'confidences': [],
            'correct': []
        }
        
        eval_pbar = tqdm(data_loader, desc='Generating Detailed Predictions', leave=True)
        
        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                results['predictions'].extend(predicted.cpu().numpy())
                results['true_labels'].extend(labels.cpu().numpy())
                results['confidences'].extend(confidences.cpu().numpy())
                results['correct'].extend((predicted == labels).cpu().numpy())
        
        pred_languages = self.label_encoder.inverse_transform(results['predictions'])
        true_languages = self.label_encoder.inverse_transform(results['true_labels'])
        
        if save_errors:
            error_file = os.path.join(self.config['paths']['results_dir'], 'classification_errors.txt')
            os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
            
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("CLASSIFICATION ERROR ANALYSIS\n")
                f.write("="*50 + "\n\n")
                
                error_count = 0
                for i, (pred, true, conf, correct) in enumerate(zip(
                    pred_languages, true_languages, results['confidences'], results['correct']
                )):
                    if not correct:
                        error_count += 1
                        f.write(f"Error #{error_count}:\n")
                        f.write(f"  True Language: {true}\n")
                        f.write(f"  Predicted Language: {pred}\n")
                        f.write(f"  Confidence: {conf:.4f}\n")
                        f.write("-" * 30 + "\n")
                
                f.write(f"\nTotal Errors: {error_count}\n")
                f.write(f"Total Samples: {len(results['correct'])}\n")
                f.write(f"Error Rate: {error_count/len(results['correct']):.4f}\n")
            
            print(f"Error analysis saved to: {error_file}")
        
        return {
            'predicted_languages': pred_languages,
            'true_languages': true_languages,
            'confidences': results['confidences'],
            'correct': results['correct'],
            'accuracy': sum(results['correct']) / len(results['correct'])
        }
    
    def predict(self, texts):
        """Predict languages for given texts"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        predictions = []
        confidences = []
        
        predict_pbar = tqdm(texts, desc='Predicting', leave=True)
        
        with torch.no_grad():
            for text in predict_pbar:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['model']['max_length'],
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_label = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
                predictions.append(predicted_label)
                confidences.append(confidence.cpu().numpy()[0])
        
        return predictions, confidences
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'logger'):
            self.logger.close()