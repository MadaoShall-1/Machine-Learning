"""
Phase 2: Model Architecture for Streaming Code-Switching Prediction
=====================================================================
Implements:
1. CausalCodeSwitchModel - Dual-head transformer with causal attention
2. Multitask loss function with weighted switch + duration losses
3. Training loop with validation and early stopping
4. Comprehensive evaluation metrics including universality analysis

Architecture:
    [Input Tokens] -> [Multilingual Encoder (XLM-R/mBERT)] -> [Causal Mask]
                                    |
                    +---------------+---------------+
                    |                               |
              [Switch Head]                   [Duration Head]
              (Binary: 0/1)                   (Multiclass: 0/1/2)
                    |                               |
              [y_sw prediction]              [y_dur prediction]
              
Author: Shuhuan Ye, Zhihang Cheng, Qi Zhou
Course: CS 5100/7100 - Northeastern University
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    get_linear_schedule_with_warmup
)

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the CausalCodeSwitchModel"""
    
    # Model backbone
    model_name: str = "xlm-roberta-base"  # or "bert-base-multilingual-cased"
    hidden_size: int = 768  # XLM-R base hidden size
    
    # Prediction heads
    switch_hidden_dim: int = 256
    duration_hidden_dim: int = 256
    num_duration_classes: int = 3  # Small, Medium, Large
    dropout_rate: float = 0.1
    
    # Causal attention
    max_context_window: int = 128
    use_causal_mask: bool = True
    
    # Multitask loss weights
    lambda_switch: float = 1.0
    lambda_duration: float = 0.5  # Lower weight since duration only applies at switches
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("./model_outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect hidden size based on model
        if "large" in self.model_name.lower():
            self.hidden_size = 1024
        elif "base" in self.model_name.lower():
            self.hidden_size = 768


# ============================================================================
# Causal Attention Mask Utilities
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (look-ahead) attention mask.
    
    For streaming prediction, token at position t should only attend to
    positions [0, 1, ..., t], not [t+1, t+2, ...].
    
    Returns:
        mask: [seq_len, seq_len] tensor where mask[i,j] = 0 if i >= j, else -inf
    """
    # Create lower triangular matrix (1s where attention is allowed)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Convert to attention mask format: 0 where allowed, -inf where blocked
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    
    return mask


def create_combined_attention_mask(
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combine padding attention mask with causal mask.
    
    Args:
        attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
        causal_mask: [seq_len, seq_len] - causal attention pattern
    
    Returns:
        combined: [batch, 1, seq_len, seq_len] - combined mask for transformer
    """
    batch_size, seq_len = attention_mask.shape
    
    # Expand padding mask: [batch, 1, 1, seq_len]
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    
    # Expand causal mask: [1, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Combine: padding mask blocks attention to padded positions
    # causal mask blocks attention to future positions
    combined = padding_mask.float() * (causal_mask == 0).float()
    
    # Convert back to mask format
    combined = combined.masked_fill(combined == 0, float('-inf'))
    combined = combined.masked_fill(combined == 1, 0.0)
    
    return combined


# ============================================================================
# Prediction Heads
# ============================================================================

class SwitchPredictionHead(nn.Module):
    """
    Binary classification head for predicting code-switching events.
    
    At position t, predicts P(LID(x_{t+1}) != LID(x_t))
    """
    
    def __init__(self, hidden_size: int, intermediate_dim: int, dropout: float):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, 2)  # Binary: switch or no-switch
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            logits: [batch, seq_len, 2] - switch prediction logits
        """
        return self.classifier(hidden_states)


class DurationPredictionHead(nn.Module):
    """
    Multi-class classification head for predicting switch duration.
    
    Predicts the duration class (Small/Medium/Large) of the upcoming
    code-switching burst, only evaluated at actual switch positions.
    
    Classes:
        0 (Small):  1-2 tokens  - lexical insertions
        1 (Medium): 3-6 tokens  - phrase-level switches
        2 (Large):  7+ tokens   - clausal/sentential switches
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_dim: int, 
        num_classes: int,
        dropout: float
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            logits: [batch, seq_len, num_classes] - duration prediction logits
        """
        return self.classifier(hidden_states)


# ============================================================================
# Main Model: CausalCodeSwitchModel
# ============================================================================

class CausalCodeSwitchModel(nn.Module):
    """
    Causal Transformer Model for Streaming Code-Switching Prediction.
    
    Architecture:
        1. Multilingual encoder backbone (XLM-R or mBERT)
        2. Causal attention masking for streaming compatibility
        3. Dual prediction heads:
           - Switch head: Binary prediction of code-switch at next position
           - Duration head: Multi-class prediction of switch burst length
    
    The model enforces the streaming constraint: at position t, predictions
    are made using only the context [x_1, ..., x_t], never seeing x_{t+1}.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Load pretrained multilingual encoder
        logger.info(f"Loading encoder: {config.model_name}")
        self.encoder = AutoModel.from_pretrained(
            config.model_name,
            add_pooling_layer=False  # We don't need [CLS] pooling
        )
        
        # Freeze bottom layers for efficiency (optional)
        # self._freeze_encoder_layers(num_layers=6)
        
        # Prediction heads
        self.switch_head = SwitchPredictionHead(
            hidden_size=config.hidden_size,
            intermediate_dim=config.switch_hidden_dim,
            dropout=config.dropout_rate
        )
        
        self.duration_head = DurationPredictionHead(
            hidden_size=config.hidden_size,
            intermediate_dim=config.duration_hidden_dim,
            num_classes=config.num_duration_classes,
            dropout=config.dropout_rate
        )
        
        # Layer normalization before heads (helps training stability)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the bottom N layers of the encoder"""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze bottom layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        logger.info(f"Froze embeddings and first {num_layers} encoder layers")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        apply_causal_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with causal attention masking.
        
        Args:
            input_ids: [batch, seq_len] - token IDs
            attention_mask: [batch, seq_len] - padding mask (1=valid, 0=pad)
            apply_causal_mask: Whether to apply causal (left-to-right) masking
        
        Returns:
            dict with:
                - switch_logits: [batch, seq_len, 2]
                - duration_logits: [batch, seq_len, num_duration_classes]
                - hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # For HuggingFace models, we pass the 2D attention_mask directly
        # The model will handle the expansion internally
        # 
        # For causal masking, we have two options:
        # 1. Use a causal LM head (but XLM-R is not causal by default)
        # 2. Create a proper 4D mask and pass it differently
        #
        # Since XLM-R doesn't natively support causal attention in the same way
        # as GPT models, we'll implement causal behavior at the APPLICATION level:
        # - During training/inference, we only use the prediction at the LAST position
        # - This naturally enforces causality since we're predicting for position t
        #   using context [0, 1, ..., t-1, t]
        
        # Standard forward pass with padding mask only
        # Causal constraint is enforced by only using last position's output
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask  # [batch, seq_len] - 2D mask
        )
        
        # Get hidden states
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Predict with both heads
        switch_logits = self.switch_head(hidden_states)
        duration_logits = self.duration_head(hidden_states)
        
        return {
            'switch_logits': switch_logits,
            'duration_logits': duration_logits,
            'hidden_states': hidden_states
        }
    
    def _create_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create extended attention mask combining padding and causal constraints.
        
        For XLM-R/BERT, the mask should have:
        - Shape: [batch, 1, seq_len, seq_len]
        - Values: 0.0 for allowed attention, -10000.0 for blocked
        """
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        
        # Expand padding mask to [batch, 1, 1, seq_len]
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
        
        # Expand causal mask to [1, 1, seq_len, seq_len]  
        # causal_mask is already 0 for allowed, -inf for blocked
        causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Create combined mask
        # Attention is allowed only where both masks allow it
        # padding_mask: 1 = valid token, 0 = padding
        # causal_mask: 0 = allowed, -inf = blocked
        
        # First, convert causal to binary: 1 = allowed, 0 = blocked
        causal_binary = (causal_mask_expanded == 0).float()
        
        # Combine: attention allowed only if both allow
        # Broadcast padding_mask from [batch, 1, 1, seq_len] to full shape
        combined = padding_mask * causal_binary
        
        # Convert to transformer format: 0 for allowed, large negative for blocked
        extended_mask = (1.0 - combined) * -10000.0
        
        return extended_mask


# ============================================================================
# Multitask Loss Function
# ============================================================================

class MultitaskCodeSwitchLoss(nn.Module):
    """
    Combined loss function for switch prediction and duration prediction.
    
    L_total = λ₁ * L_switch + λ₂ * L_duration
    
    Key features:
    - Switch loss: Standard cross-entropy for binary classification
    - Duration loss: Cross-entropy with ignore_index=-1, only computed
      at positions where a switch actually occurs
    - Class weighting for imbalanced switch distribution
    """
    
    def __init__(
        self,
        lambda_switch: float = 1.0,
        lambda_duration: float = 0.5,
        switch_class_weights: Optional[torch.Tensor] = None,
        duration_class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1
    ):
        super().__init__()
        
        self.lambda_switch = lambda_switch
        self.lambda_duration = lambda_duration
        self.ignore_index = ignore_index
        
        # Switch loss (binary cross-entropy)
        self.switch_criterion = nn.CrossEntropyLoss(
            weight=switch_class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        # Duration loss (only at switch positions)
        self.duration_criterion = nn.CrossEntropyLoss(
            weight=duration_class_weights,
            ignore_index=ignore_index,  # Critical: ignore non-switch positions
            reduction='mean'
        )
    
    def forward(
        self,
        switch_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        switch_labels: torch.Tensor,
        duration_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            switch_logits: [batch, seq_len, 2]
            duration_logits: [batch, seq_len, num_classes]
            switch_labels: [batch, seq_len] - 0/1 for no-switch/switch, -1 for ignore
            duration_labels: [batch, seq_len] - 0/1/2 for duration class, -1 for ignore
        
        Returns:
            dict with loss_total, loss_switch, loss_duration
        """
        batch_size, seq_len, _ = switch_logits.shape
        
        # Reshape for cross-entropy: [batch*seq_len, num_classes]
        switch_logits_flat = switch_logits.view(-1, 2)
        duration_logits_flat = duration_logits.view(-1, duration_logits.size(-1))
        
        # Flatten labels: [batch*seq_len]
        switch_labels_flat = switch_labels.view(-1)
        duration_labels_flat = duration_labels.view(-1)
        
        # Compute losses
        loss_switch = self.switch_criterion(switch_logits_flat, switch_labels_flat)
        loss_duration = self.duration_criterion(duration_logits_flat, duration_labels_flat)
        
        # Handle case where no valid duration labels exist
        if torch.isnan(loss_duration):
            loss_duration = torch.tensor(0.0, device=switch_logits.device)
        
        # Combined loss
        loss_total = self.lambda_switch * loss_switch + self.lambda_duration * loss_duration
        
        return {
            'loss_total': loss_total,
            'loss_switch': loss_switch,
            'loss_duration': loss_duration
        }


# ============================================================================
# Evaluation Metrics
# ============================================================================

class CodeSwitchMetrics:
    """
    Compute evaluation metrics for code-switching prediction.
    
    Metrics:
    - Anticipatory F1: F1 score for SWITCH class at position t
    - Duration Accuracy: Accuracy of duration prediction at switch positions
    - Macro-averaged metrics across switch types
    - Universality metrics (per-pair performance and variance)
    """
    
    def __init__(self, num_duration_classes: int = 3):
        self.num_duration_classes = num_duration_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        # Switch prediction
        self.switch_tp = 0  # True positives (predicted switch, actual switch)
        self.switch_fp = 0  # False positives (predicted switch, no actual switch)
        self.switch_fn = 0  # False negatives (no predicted switch, actual switch)
        self.switch_tn = 0  # True negatives
        
        # Duration prediction
        self.duration_correct = defaultdict(int)  # Correct per class
        self.duration_total = defaultdict(int)    # Total per class
        self.duration_predictions = []
        self.duration_targets = []
        
        # Per-pair tracking (for universality)
        self.pair_metrics = defaultdict(lambda: {
            'switch_tp': 0, 'switch_fp': 0, 'switch_fn': 0, 'switch_tn': 0,
            'duration_correct': 0, 'duration_total': 0
        })
    
    def update(
        self,
        switch_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        switch_labels: torch.Tensor,
        duration_labels: torch.Tensor,
        language_pair: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            switch_logits: [batch, seq_len, 2]
            duration_logits: [batch, seq_len, num_classes]
            switch_labels: [batch, seq_len]
            duration_labels: [batch, seq_len]
            language_pair: Optional pair identifier for universality tracking
            threshold: Classification threshold for switch prediction (default 0.5)
        """
        # Get predictions using threshold
        switch_probs = torch.softmax(switch_logits, dim=-1)[:, :, 1]  # P(switch=1)
        switch_preds = (switch_probs >= threshold).long()  # [batch, seq_len]
        duration_preds = torch.argmax(duration_logits, dim=-1)
        
        # Flatten
        switch_preds = switch_preds.view(-1).cpu().numpy()
        switch_labels = switch_labels.view(-1).cpu().numpy()
        duration_preds = duration_preds.view(-1).cpu().numpy()
        duration_labels = duration_labels.view(-1).cpu().numpy()
        
        # Filter out ignored positions
        valid_switch_mask = switch_labels != -1
        valid_duration_mask = duration_labels != -1
        
        # Update switch metrics
        for pred, label in zip(switch_preds[valid_switch_mask], 
                               switch_labels[valid_switch_mask]):
            if pred == 1 and label == 1:
                self.switch_tp += 1
                if language_pair:
                    self.pair_metrics[language_pair]['switch_tp'] += 1
            elif pred == 1 and label == 0:
                self.switch_fp += 1
                if language_pair:
                    self.pair_metrics[language_pair]['switch_fp'] += 1
            elif pred == 0 and label == 1:
                self.switch_fn += 1
                if language_pair:
                    self.pair_metrics[language_pair]['switch_fn'] += 1
            else:
                self.switch_tn += 1
                if language_pair:
                    self.pair_metrics[language_pair]['switch_tn'] += 1
        
        # Update duration metrics (only at switch positions)
        for pred, label in zip(duration_preds[valid_duration_mask],
                               duration_labels[valid_duration_mask]):
            self.duration_total[int(label)] += 1
            if language_pair:
                self.pair_metrics[language_pair]['duration_total'] += 1
            
            if pred == label:
                self.duration_correct[int(label)] += 1
                if language_pair:
                    self.pair_metrics[language_pair]['duration_correct'] += 1
            
            self.duration_predictions.append(int(pred))
            self.duration_targets.append(int(label))
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        results = {}
        
        # ===== Switch Prediction Metrics =====
        
        # Precision, Recall, F1 for SWITCH class
        precision = self.switch_tp / (self.switch_tp + self.switch_fp + 1e-10)
        recall = self.switch_tp / (self.switch_tp + self.switch_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        results['switch_precision'] = precision
        results['switch_recall'] = recall
        results['switch_f1'] = f1  # This is the "Anticipatory F1"
        
        # Overall switch accuracy
        total = self.switch_tp + self.switch_fp + self.switch_fn + self.switch_tn
        accuracy = (self.switch_tp + self.switch_tn) / (total + 1e-10)
        results['switch_accuracy'] = accuracy
        
        # ===== Duration Prediction Metrics =====
        
        # Per-class accuracy
        for cls in range(self.num_duration_classes):
            if self.duration_total[cls] > 0:
                acc = self.duration_correct[cls] / self.duration_total[cls]
                results[f'duration_acc_class_{cls}'] = acc
        
        # Overall duration accuracy
        total_dur_correct = sum(self.duration_correct.values())
        total_dur = sum(self.duration_total.values())
        results['duration_accuracy'] = total_dur_correct / (total_dur + 1e-10)
        
        # Macro-averaged duration accuracy
        class_accs = []
        for cls in range(self.num_duration_classes):
            if self.duration_total[cls] > 0:
                class_accs.append(self.duration_correct[cls] / self.duration_total[cls])
        results['duration_macro_accuracy'] = np.mean(class_accs) if class_accs else 0.0
        
        # ===== Universality Metrics (per-pair analysis) =====
        
        pair_f1_scores = []
        for pair, metrics in self.pair_metrics.items():
            tp = metrics['switch_tp']
            fp = metrics['switch_fp']
            fn = metrics['switch_fn']
            
            pair_precision = tp / (tp + fp + 1e-10)
            pair_recall = tp / (tp + fn + 1e-10)
            pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall + 1e-10)
            
            results[f'{pair}_f1'] = pair_f1
            pair_f1_scores.append(pair_f1)
        
        # σ_universality: Standard deviation of F1 across pairs
        # Lower is better (more consistent across languages)
        if len(pair_f1_scores) > 1:
            results['sigma_universality'] = np.std(pair_f1_scores)
            results['mean_pair_f1'] = np.mean(pair_f1_scores)
        
        return results
    
    def get_confusion_matrix(self, task: str = 'switch') -> np.ndarray:
        """Get confusion matrix for specified task"""
        if task == 'switch':
            return np.array([
                [self.switch_tn, self.switch_fp],
                [self.switch_fn, self.switch_tp]
            ])
        else:  # duration
            from sklearn.metrics import confusion_matrix
            if self.duration_targets:
                return confusion_matrix(
                    self.duration_targets, 
                    self.duration_predictions,
                    labels=list(range(self.num_duration_classes))
                )
            return np.zeros((self.num_duration_classes, self.num_duration_classes))


# ============================================================================
# Enhanced Dataset with Language Pair Tracking
# ============================================================================

class EnhancedStreamingDataset(Dataset):
    """
    Enhanced dataset that tracks language pairs for universality evaluation.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_context_window: int = 128,
        include_pair_info: bool = True,
        max_samples: Optional[int] = None
    ):
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        self.max_context_window = max_context_window
        self.include_pair_info = include_pair_info
        
        # Flatten samples into individual prediction points
        all_points = []
        
        for sample_idx, sample in enumerate(self.samples):
            token_ids = sample['token_ids']
            switch_labels = sample['switch_labels']
            duration_labels = sample['duration_labels']
            pair = sample.get('pair', 'unknown')
            
            for t in range(1, len(token_ids) - 1):
                if switch_labels[t] != -1:
                    all_points.append({
                        'token_ids': token_ids,
                        'position': t,
                        'switch_label': switch_labels[t],
                        'duration_label': duration_labels[t],
                        'pair': pair,
                        'sample_idx': sample_idx
                    })
        
        # 限制样本数量 (随机采样以保持分布)
        if max_samples is not None and len(all_points) > max_samples:
            import random
            random.seed(42)  # 固定种子保证可复现
            self.prediction_points = random.sample(all_points, max_samples)
            logger.info(f"⚡ Sampled {max_samples} from {len(all_points)} total points")
        else:
            self.prediction_points = all_points
        
        logger.info(f"Created dataset with {len(self.prediction_points)} prediction points")
        
        # Log distribution
        pair_counts = defaultdict(int)
        for p in self.prediction_points:
            pair_counts[p['pair']] += 1
        
        logger.info("Distribution by language pair:")
        for pair, count in sorted(pair_counts.items()):
            logger.info(f"  {pair}: {count}")
    
    def __len__(self):
        return len(self.prediction_points)
    
    def __getitem__(self, idx):
        point = self.prediction_points[idx]
        
        t = point['position']
        token_ids = point['token_ids']
        
        # Extract causal context: tokens [max(0, t-window), t]
        start_idx = max(0, t - self.max_context_window + 1)
        context_ids = token_ids[start_idx:t+1]
        
        # Pad if necessary (left padding)
        if len(context_ids) < self.max_context_window:
            padding = [0] * (self.max_context_window - len(context_ids))
            context_ids = padding + context_ids
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tid != 0 else 0 for tid in context_ids]
        
        result = {
            'input_ids': torch.tensor(context_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'switch_label': torch.tensor(point['switch_label'], dtype=torch.long),
            'duration_label': torch.tensor(point['duration_label'], dtype=torch.long),
        }
        
        if self.include_pair_info:
            result['pair'] = point['pair']
        
        return result


# ============================================================================
# Trainer Class
# ============================================================================

class TrainingProgressCallback:
    """
    回调函数：追踪和保存训练进度
    Callback for tracking and saving training progress
    """
    
    def __init__(self, log_dir: Path, log_every_n_steps: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_every_n_steps
        
        # 进度文件
        self.progress_file = self.log_dir / "training_progress.json"
        self.step_log_file = self.log_dir / "step_logs.jsonl"
        
        # 状态追踪
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.best_metric = 0.0
        self.history = []
        
    def on_train_begin(self, total_epochs: int, steps_per_epoch: int):
        """训练开始时调用"""
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        
        self._save_progress({
            'status': 'training',
            'current_epoch': 0,
            'total_epochs': total_epochs,
            'current_step': 0,
            'total_steps': self.total_steps,
            'progress_pct': 0.0
        })
        
        logger.info(f"📊 Training progress will be saved to: {self.progress_file}")
    
    def on_step_end(self, epoch: int, step: int, loss: float, metrics: dict = None):
        """每个step结束时调用"""
        self.current_epoch = epoch
        self.current_step = epoch * self.steps_per_epoch + step
        
        # 每N步记录一次
        if step % self.log_every_n_steps == 0:
            progress_pct = (self.current_step / self.total_steps) * 100
            
            step_info = {
                'epoch': epoch,
                'step': step,
                'global_step': self.current_step,
                'loss': loss,
                'progress_pct': round(progress_pct, 2)
            }
            if metrics:
                step_info.update(metrics)
            
            # 追加到日志文件
            with open(self.step_log_file, 'a') as f:
                f.write(json.dumps(step_info) + '\n')
            
            # 更新进度文件
            self._save_progress({
                'status': 'training',
                'current_epoch': epoch + 1,
                'total_epochs': self.total_epochs,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'progress_pct': round(progress_pct, 2),
                'current_loss': loss
            })
    
    def on_epoch_end(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """每个epoch结束时调用"""
        progress_pct = ((epoch + 1) / self.total_epochs) * 100
        
        epoch_info = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'progress_pct': round(progress_pct, 2)
        }
        self.history.append(epoch_info)
        
        # 更新进度
        self._save_progress({
            'status': 'training',
            'current_epoch': epoch + 1,
            'total_epochs': self.total_epochs,
            'progress_pct': round(progress_pct, 2),
            'best_val_f1': self.best_metric,
            'last_val_f1': val_metrics.get('switch_f1', 0),
            'history': self.history
        })
    
    def on_train_end(self, final_metrics: dict):
        """训练结束时调用"""
        self._save_progress({
            'status': 'completed',
            'current_epoch': self.total_epochs,
            'total_epochs': self.total_epochs,
            'progress_pct': 100.0,
            'final_metrics': final_metrics,
            'history': self.history
        })
        logger.info(f"✅ Training complete! Final progress saved to {self.progress_file}")
    
    def _save_progress(self, data: dict):
        """保存进度到JSON文件"""
        import time
        data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)


class CodeSwitchTrainer:
    """
    Trainer for CausalCodeSwitchModel with multitask learning.
    
    Features:
    - Mixed precision training (optional)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint saving
    - Comprehensive logging
    - 📊 Progress tracking (check training_progress.json)
    - 🎯 Custom threshold for evaluation
    """
    
    def __init__(
        self,
        model: CausalCodeSwitchModel,
        config: ModelConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        device: str = 'auto',
        threshold: float = 0.75
    ):
        self.model = model
        self.config = config
        self.threshold = threshold  # 评估时使用的阈值
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Evaluation threshold: {self.threshold}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Determine num_workers based on OS
        # Windows has issues with multiprocessing, so we use fewer workers
        import platform
        if platform.system() == 'Windows':
            num_workers = 0  # Windows multiprocessing can be problematic
        else:
            num_workers = 4  # Linux/Mac can use more workers
        
        logger.info(f"DataLoader num_workers: {num_workers}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=self._collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.test_loader = None
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self._collate_fn,
                persistent_workers=True if num_workers > 0 else False
            )
        
        # Calculate class weights for imbalanced data
        switch_weights = self._calculate_class_weights(train_dataset, 'switch')
        duration_weights = self._calculate_class_weights(train_dataset, 'duration')
        
        logger.info(f"Switch class weights: {switch_weights}")
        logger.info(f"Duration class weights: {duration_weights}")
        
        # Initialize loss function
        self.criterion = MultitaskCodeSwitchLoss(
            lambda_switch=config.lambda_switch,
            lambda_duration=config.lambda_duration,
            switch_class_weights=switch_weights.to(self.device) if switch_weights is not None else None,
            duration_class_weights=duration_weights.to(self.device) if duration_weights is not None else None
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        num_training_steps = len(self.train_loader) * config.max_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize metrics
        self.metrics = CodeSwitchMetrics(num_duration_classes=config.num_duration_classes)
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = []
    
    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        switch_labels = torch.stack([item['switch_label'] for item in batch])
        duration_labels = torch.stack([item['duration_label'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'switch_labels': switch_labels,
            'duration_labels': duration_labels
        }
        
        # Include pair info if available
        if 'pair' in batch[0]:
            result['pairs'] = [item['pair'] for item in batch]
        
        return result
    
    def _calculate_class_weights(
        self, 
        dataset: Dataset, 
        task: str
    ) -> Optional[torch.Tensor]:
        """Calculate inverse frequency class weights"""
        label_key = 'switch_label' if task == 'switch' else 'duration_label'
        num_classes = 2 if task == 'switch' else self.config.num_duration_classes
        
        class_counts = defaultdict(int)
        
        for item in dataset:
            label = item[label_key].item()
            if label != -1:  # Ignore ignored labels
                class_counts[label] += 1
        
        if not class_counts:
            return None
        
        total = sum(class_counts.values())
        weights = []
        
        for cls in range(num_classes):
            count = class_counts.get(cls, 1)
            weight = total / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_switch_loss = 0.0
        total_duration_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.max_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            switch_labels = batch['switch_labels'].to(self.device)
            duration_labels = batch['duration_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_causal_mask=True
            )
            
            # For single-position prediction, we need to get the last position's logits
            # The model outputs logits for all positions, but we predict from the last token
            # Since each sample is a context window ending at position t, we use the last position
            switch_logits = outputs['switch_logits'][:, -1, :]  # [batch, 2]
            duration_logits = outputs['duration_logits'][:, -1, :]  # [batch, num_classes]
            
            # Reshape labels to match (they're already single values)
            # Add sequence dimension for loss computation
            switch_logits = switch_logits.unsqueeze(1)  # [batch, 1, 2]
            duration_logits = duration_logits.unsqueeze(1)  # [batch, 1, num_classes]
            switch_labels = switch_labels.unsqueeze(1)  # [batch, 1]
            duration_labels = duration_labels.unsqueeze(1)  # [batch, 1]
            
            # Compute loss
            losses = self.criterion(
                switch_logits=switch_logits,
                duration_logits=duration_logits,
                switch_labels=switch_labels,
                duration_labels=duration_labels
            )
            
            loss = losses['loss_total']
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track losses
            total_loss += losses['loss_total'].item()
            total_switch_loss += losses['loss_switch'].item()
            total_duration_loss += losses['loss_duration'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['loss_total'].item():.4f}",
                'sw': f"{losses['loss_switch'].item():.4f}",
                'dur': f"{losses['loss_duration'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'switch_loss': total_switch_loss / num_batches,
            'duration_loss': total_duration_loss / num_batches
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split_name: str = 'val') -> Dict[str, float]:
        """Evaluate on validation or test set"""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            switch_labels = batch['switch_labels'].to(self.device)
            duration_labels = batch['duration_labels'].to(self.device)
            pairs = batch.get('pairs', None)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_causal_mask=True
            )
            
            # Get last position predictions
            switch_logits = outputs['switch_logits'][:, -1, :]
            duration_logits = outputs['duration_logits'][:, -1, :]
            
            # Compute loss
            switch_logits_loss = switch_logits.unsqueeze(1)
            duration_logits_loss = duration_logits.unsqueeze(1)
            switch_labels_loss = switch_labels.unsqueeze(1)
            duration_labels_loss = duration_labels.unsqueeze(1)
            
            losses = self.criterion(
                switch_logits=switch_logits_loss,
                duration_logits=duration_logits_loss,
                switch_labels=switch_labels_loss,
                duration_labels=duration_labels_loss
            )
            
            total_loss += losses['loss_total'].item()
            num_batches += 1
            
            # Update metrics with threshold
            # Expand logits for metrics (expects seq_len dimension)
            switch_logits_metrics = switch_logits.unsqueeze(1)
            duration_logits_metrics = duration_logits.unsqueeze(1)
            switch_labels_metrics = switch_labels.unsqueeze(1)
            duration_labels_metrics = duration_labels.unsqueeze(1)
            
            # Update per sample with pair info
            if pairs:
                for i, pair in enumerate(pairs):
                    self.metrics.update(
                        switch_logits_metrics[i:i+1],
                        duration_logits_metrics[i:i+1],
                        switch_labels_metrics[i:i+1],
                        duration_labels_metrics[i:i+1],
                        language_pair=pair,
                        threshold=self.threshold
                    )
            else:
                self.metrics.update(
                    switch_logits_metrics,
                    duration_logits_metrics,
                    switch_labels_metrics,
                    duration_labels_metrics,
                    threshold=self.threshold
                )
        
        # Compute metrics
        results = self.metrics.compute()
        results['loss'] = total_loss / num_batches
        
        return results
    
    def train(self) -> Dict:
        """Full training loop with early stopping"""
        logger.info("="*60)
        logger.info("Starting Training")
        logger.info("="*60)
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Loss weights: λ_switch={self.config.lambda_switch}, λ_duration={self.config.lambda_duration}")
        logger.info("="*60)
        
        best_model_state = None
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader, 'validation')
            
            # Log metrics
            logger.info(f"\nEpoch {epoch+1} Results:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Switch F1: {val_metrics['switch_f1']:.4f}")
            logger.info(f"  Val Switch Precision: {val_metrics['switch_precision']:.4f}")
            logger.info(f"  Val Switch Recall: {val_metrics['switch_recall']:.4f}")
            logger.info(f"  Val Duration Accuracy: {val_metrics['duration_accuracy']:.4f}")
            
            if 'sigma_universality' in val_metrics:
                logger.info(f"  σ_universality: {val_metrics['sigma_universality']:.4f}")
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Check for improvement
            current_f1 = val_metrics['switch_f1']
            
            if current_f1 > self.best_val_f1 + self.config.min_delta:
                logger.info(f"  ✓ New best F1: {current_f1:.4f} (prev: {self.best_val_f1:.4f})")
                self.best_val_f1 = current_f1
                self.patience_counter = 0
                
                # Save best model
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                
                checkpoint_path = self.config.checkpoint_dir / 'best_model.pt'
                torch.save(best_model_state, checkpoint_path)
                logger.info(f"  Saved checkpoint to {checkpoint_path}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.config.patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model for final evaluation
        if best_model_state:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            logger.info("Loaded best model for final evaluation")
        
        # Final test evaluation
        final_results = {
            'training_history': self.training_history,
            'best_val_f1': self.best_val_f1
        }
        
        if self.test_loader:
            logger.info("\n" + "="*60)
            logger.info("Final Test Evaluation")
            logger.info("="*60)
            
            test_metrics = self.evaluate(self.test_loader, 'test')
            final_results['test_metrics'] = test_metrics
            
            logger.info(f"Test Switch F1: {test_metrics['switch_f1']:.4f}")
            logger.info(f"Test Switch Precision: {test_metrics['switch_precision']:.4f}")
            logger.info(f"Test Switch Recall: {test_metrics['switch_recall']:.4f}")
            logger.info(f"Test Duration Accuracy: {test_metrics['duration_accuracy']:.4f}")
            
            # Per-pair results
            logger.info("\nPer-Language-Pair Results:")
            for key, value in test_metrics.items():
                if key.endswith('_f1') and key != 'switch_f1':
                    pair_name = key.replace('_f1', '')
                    logger.info(f"  {pair_name}: F1 = {value:.4f}")
            
            if 'sigma_universality' in test_metrics:
                logger.info(f"\nUniversality Metric:")
                logger.info(f"  Mean Pair F1: {test_metrics.get('mean_pair_f1', 0):.4f}")
                logger.info(f"  σ_universality: {test_metrics['sigma_universality']:.4f}")
        
        return final_results


# ============================================================================
# Inference / Streaming Simulation
# ============================================================================

class StreamingPredictor:
    """
    Streaming predictor for real-time code-switching prediction.
    
    Simulates processing text token-by-token, outputting:
    - P(switch at next position)
    - Predicted duration class (if switch predicted)
    """
    
    def __init__(
        self,
        model: CausalCodeSwitchModel,
        tokenizer,
        max_context_window: int = 128,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_window = max_context_window
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Context buffer
        self.token_buffer = []
        
        # Duration class names
        self.duration_classes = ['Small (1-2 tokens)', 'Medium (3-6 tokens)', 'Large (7+ tokens)']
    
    def reset(self):
        """Reset the context buffer"""
        self.token_buffer = []
    
    @torch.no_grad()
    def predict_next(self, text: str) -> Dict:
        """
        Process text and predict code-switching for the next token.
        
        Args:
            text: The text processed so far
        
        Returns:
            dict with:
                - switch_prob: P(code-switch at next position)
                - predicted_switch: Boolean
                - duration_class: Predicted duration class (if switch)
                - duration_probs: Probability distribution over duration classes
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_context_window
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            apply_causal_mask=True
        )
        
        # Get predictions for last position
        switch_logits = outputs['switch_logits'][0, -1, :]  # [2]
        duration_logits = outputs['duration_logits'][0, -1, :]  # [3]
        
        # Convert to probabilities
        switch_probs = F.softmax(switch_logits, dim=0)
        duration_probs = F.softmax(duration_logits, dim=0)
        
        # Get predictions
        switch_prob = switch_probs[1].item()  # P(switch)
        predicted_switch = switch_prob > 0.5
        duration_class = torch.argmax(duration_probs).item()
        
        return {
            'switch_prob': switch_prob,
            'predicted_switch': predicted_switch,
            'duration_class': self.duration_classes[duration_class] if predicted_switch else None,
            'duration_probs': duration_probs.cpu().numpy(),
            'num_tokens': input_ids.shape[1]
        }
    
    def stream_predict(self, text: str, word_by_word: bool = True) -> List[Dict]:
        """
        Stream through text and output predictions at each step.
        
        Args:
            text: Full text to process
            word_by_word: If True, process word by word; else token by token
        
        Returns:
            List of predictions at each step
        """
        predictions = []
        
        if word_by_word:
            words = text.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text = " ".join(words[:i+1])
                
                if len(current_text.strip()) > 0:
                    pred = self.predict_next(current_text)
                    pred['position'] = i
                    pred['current_word'] = word
                    pred['context'] = current_text
                    predictions.append(pred)
        else:
            # Token by token
            encoding = self.tokenizer(text, add_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            
            for i in range(1, len(tokens)):
                partial_ids = encoding['input_ids'][:i+1]
                partial_text = self.tokenizer.decode(partial_ids)
                
                pred = self.predict_next(partial_text)
                pred['position'] = i
                pred['current_token'] = tokens[i]
                pred['context'] = partial_text
                predictions.append(pred)
        
        return predictions


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main training script for Phase 2.
    
    Usage:
        python model_architecture_phase2.py --data_dir ./processed_data --output_dir ./outputs
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CausalCodeSwitchModel')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Directory containing processed .pkl files')
    parser.add_argument('--output_dir', type=str, default='./model_outputs',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                        help='Pretrained model name (xlm-roberta-base or bert-base-multilingual-cased)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum training epochs')
    parser.add_argument('--lambda_switch', type=float, default=1.0,
                        help='Weight for switch prediction loss')
    parser.add_argument('--lambda_duration', type=float, default=0.5,
                        help='Weight for duration prediction loss')
    parser.add_argument('--context_window', type=int, default=128,
                        help='Maximum context window size')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (auto/cuda/cpu). Use "cpu" if you have CUDA compatibility issues')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum training samples (for faster testing). E.g., 50000 or 100000')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Classification threshold for switch prediction (default 0.5, recommended 0.75)')
    
    args = parser.parse_args()
    
    # ========== 启动提示 ==========
    print("\n" + "="*60)
    print("🚀 CODE-SWITCHING PREDICTION MODEL - PHASE 2")
    print("="*60)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_name}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Max epochs: {args.max_epochs}")
    print(f"🎯 Threshold: {args.threshold}")
    print("="*60 + "\n")
    
    # ========== 检查数据文件 ==========
    data_dir = Path(args.data_dir)
    
    print("📂 Checking data files...")
    required_files = ['train.pkl', 'val.pkl']
    missing_files = []
    
    for f in required_files:
        file_path = data_dir / f
        if file_path.exists():
            print(f"   ✅ Found: {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
            missing_files.append(f)
    
    if missing_files:
        print(f"\n❌ ERROR: Missing required data files!")
        print(f"   Please run data_processing_phase2.py first:")
        print(f"   >>> python data_processing_phase2.py")
        return None
    
    print()
    
    # Create config
    config = ModelConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        lambda_switch=args.lambda_switch,
        lambda_duration=args.lambda_duration,
        max_context_window=args.context_window,
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.output_dir) / 'checkpoints'
    )
    
    # ========== 设置设备 ==========
    if args.device == 'cpu':
        print("⚠️  Forcing CPU mode (CUDA disabled)")
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # Load datasets
    print("\n📥 Loading datasets (this may take a moment)...")
    
    if args.max_samples:
        print(f"⚡ Limiting to {args.max_samples} training samples for faster training")
    
    train_dataset = EnhancedStreamingDataset(
        data_dir / 'train.pkl',
        max_context_window=config.max_context_window,
        max_samples=args.max_samples
    )
    
    # Validation/test sets: use 10% of max_samples
    val_max = args.max_samples // 10 if args.max_samples else None
    
    val_dataset = EnhancedStreamingDataset(
        data_dir / 'val.pkl',
        max_context_window=config.max_context_window,
        max_samples=val_max
    )
    
    test_dataset = None
    if (data_dir / 'test.pkl').exists():
        test_dataset = EnhancedStreamingDataset(
            data_dir / 'test.pkl',
            max_context_window=config.max_context_window,
            max_samples=val_max
        )
    
    # Create model (on CPU first, then move to device)
    print("\n🤖 Loading pretrained model (downloading if needed, ~1GB)...")
    print("   This may take several minutes on first run...")
    model = CausalCodeSwitchModel(config)
    
    # Create trainer (will move model to correct device)
    trainer = CodeSwitchTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=str(device),  # Pass as string
        threshold=args.threshold  # 使用自定义阈值
    )
    
    # Train
    results = trainer.train()
    
    # Save results
    results_path = config.output_dir / 'training_results.json'
    
    # Convert results to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("Training complete!")
    
    return results


if __name__ == "__main__":
    main()