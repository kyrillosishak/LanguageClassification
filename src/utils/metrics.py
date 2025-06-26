import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import json
import os


class MetricsCalculator:
    """
    Calculate and visualize various metrics for language classification
    """
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def update_metrics(self, train_loss, val_loss, train_acc, val_acc, lr):
        """Update metrics history"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_accuracy'].append(train_acc)
        self.metrics_history['val_accuracy'].append(val_acc)
        self.metrics_history['learning_rate'].append(lr)
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics_history['train_accuracy'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.metrics_history['val_accuracy'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.metrics_history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference
        loss_diff = np.array(self.metrics_history['val_loss']) - np.array(self.metrics_history['train_loss'])
        axes[1, 1].plot(loss_diff, color='purple')
        axes[1, 1].set_title('Overfitting Monitor (Val Loss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        return plt.gcf()
    
    def calculate_per_class_metrics(self, y_true, y_pred, class_names):
        """Calculate detailed per-class metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {}
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics['overall'] = {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1_score': float(overall_f1),
            'accuracy': float(np.mean(np.array(y_true) == np.array(y_pred)))
        }
        
        return metrics
    
    def plot_per_class_performance(self, metrics, save_path=None):
        """Plot per-class performance metrics"""
        class_names = [k for k in metrics.keys() if k != 'overall']
        
        precision_scores = [metrics[k]['precision'] for k in class_names]
        recall_scores = [metrics[k]['recall'] for k in class_names]
        f1_scores = [metrics[k]['f1_score'] for k in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Language Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class performance plot saved to: {save_path}")
        
        return fig
    
    def generate_metrics_report(self, y_true, y_pred, class_names, 
                              predictions_with_confidence=None, save_dir=None):
        """Generate comprehensive metrics report"""
        report = {}
        
        # Basic metrics
        report['accuracy'] = float(np.mean(np.array(y_true) == np.array(y_pred)))
        report['total_samples'] = len(y_true)
        report['num_classes'] = len(class_names)
        
        # Per-class metrics
        report['per_class_metrics'] = self.calculate_per_class_metrics(
            y_true, y_pred, class_names
        )
        
        # Classification report
        report['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report['confusion_matrix'] = cm.tolist()
        
        # Error analysis
        errors = []
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                error_info = {
                    'sample_index': i,
                    'true_label': class_names[true_label],
                    'predicted_label': class_names[pred_label]
                }
                
                if predictions_with_confidence:
                    error_info['confidence'] = float(predictions_with_confidence[i])
                
                errors.append(error_info)
        
        report['errors'] = errors
        report['error_rate'] = len(errors) / len(y_true)
        
        # Save report
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save JSON report
            report_path = os.path.join(save_dir, 'metrics_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Metrics report saved to: {report_path}")
            
            # Generate and save plots
            self.plot_confusion_matrix(
                y_true, y_pred, class_names,
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )
            
            self.plot_per_class_performance(
                report['per_class_metrics'],
                save_path=os.path.join(save_dir, 'per_class_performance.png')
            )
            
            if hasattr(self, 'metrics_history') and len(self.metrics_history['train_loss']) > 0:
                self.plot_training_curves(
                    save_path=os.path.join(save_dir, 'training_curves.png')
                )
        
        return report
    
    def save_metrics_history(self, save_path):
        """Save metrics history to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Metrics history saved to: {save_path}")
    
    def load_metrics_history(self, load_path):
        """Load metrics history from JSON"""
        with open(load_path, 'r') as f:
            self.metrics_history = json.load(f)
        print(f"Metrics history loaded from: {load_path}")


def calculate_model_size(model):
    """Calculate model size in parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }


def print_model_summary(model, input_size=None):
    """Print model summary"""
    model_info = calculate_model_size(model)
    
    print("="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {model_info['non_trainable_parameters']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    print("="*50)
    
    return model_info