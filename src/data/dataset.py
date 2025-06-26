import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


class LanguageDataset(Dataset):
    """
    Dataset class for language classification
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataManager:
    """
    Data management class for loading and preparing datasets
    """
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
    def load_data(self, x_text_path, y_labels_path):
        """
        Load text data and corresponding labels
        
        Args:
            x_text_path: Path to text file
            y_labels_path: Path to labels file
            
        Returns:
            texts, labels
        """
        # Read text data
        with open(x_text_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

        # Read labels
        with open(y_labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]

        # Ensure same length
        min_len = min(len(texts), len(labels))
        texts = texts[:min_len]
        labels = labels[:min_len]

        print(f"Loaded {len(texts)} samples")
        print(f"Number of unique languages: {len(set(labels))}")

        return texts, labels
    
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Prepare data for training, validation, and testing
        
        Args:
            texts: List of texts
            labels: List of labels
            test_size: Test set size
            val_size: Validation set size
            random_state: Random state for reproducibility
            
        Returns:
            train_dataset, val_dataset, test_dataset, num_classes, label_encoder
        """
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=encoded_labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_temp
        )

        print(f"Data split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Testing samples: {len(X_test)}")

        # Create datasets
        train_dataset = LanguageDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = LanguageDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = LanguageDataset(X_test, y_test, self.tokenizer, self.max_length)

        return train_dataset, val_dataset, test_dataset, num_classes, self.label_encoder
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, 
                          batch_size=16, num_workers=4, pin_memory=True, multi_gpu=False):
        """
        Create data loaders
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Whether to pin memory
            multi_gpu: Whether using multiple GPUs
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # Adjust batch size for multi-GPU
        if multi_gpu:
            num_gpus = torch.cuda.device_count()
            effective_batch_size = batch_size * num_gpus
            print(f"Multi-GPU: Using effective batch size of {effective_batch_size} ({batch_size} per GPU)")
        else:
            effective_batch_size = batch_size
            
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        
        print(f"Data loaders created:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Testing batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
