import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LanguageClassifier(nn.Module):
    """
    BERT-based Language Classification Model
    """
    
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(LanguageClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Classification logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'hidden_size': self.bert.config.hidden_size
        }


def create_model(model_name, num_classes, dropout_rate=0.3, device='cuda', multi_gpu=False):
    """
    Create and setup the classification model
    
    Args:
        model_name: Pre-trained model name
        num_classes: Number of classes
        dropout_rate: Dropout rate
        device: Device to move model to
        multi_gpu: Whether to use DataParallel
        
    Returns:
        Configured model
    """
    model = LanguageClassifier(model_name, num_classes, dropout_rate)
    model = model.to(device)
    
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model


def get_tokenizer(model_name):
    """
    Get tokenizer for the model
    
    Args:
        model_name: Pre-trained model name
        
    Returns:
        Tokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_name)
