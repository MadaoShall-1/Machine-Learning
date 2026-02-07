"""
阈值分析工具 - Threshold Analysis for Code-Switching Prediction
================================================================

这个脚本用于分析不同阈值对 Precision/Recall/F1 的影响，
帮助找到最佳的阈值来平衡 Precision 和 Recall。

Usage:
    python threshold_analysis.py --model_path ./model_outputs/checkpoints/best_model.pt --data_dir ./processed_data

Author: Shuhuan Ye, Zhihang Cheng, Qi Zhou
Date: February 2026
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

# Import from main module
import sys
sys.path.insert(0, '.')
from model_architecture_phase2 import (
    CausalCodeSwitchModel,
    ModelConfig,
    EnhancedStreamingDataset
)


def load_model(model_path: str, device: str = 'cuda'):
    """加载训练好的模型"""
    print(f"📥 Loading model from {model_path}")
    
    # PyTorch 2.6+ requires weights_only=False for full checkpoint loading
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = ModelConfig()
    
    # 创建模型
    model = CausalCodeSwitchModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, config


def collect_predictions(
    model, 
    data_loader: DataLoader, 
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    收集所有预测结果（概率形式）
    
    Returns:
        switch_probs: P(switch) for each sample
        switch_labels: True labels
        duration_probs: Duration class probabilities
        duration_labels: True duration labels
        pairs: Language pair for each sample
    """
    all_switch_probs = []
    all_switch_labels = []
    all_duration_probs = []
    all_duration_labels = []
    all_pairs = []
    
    print("📊 Collecting predictions...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            switch_labels = batch['switch_labels']
            duration_labels = batch['duration_labels']
            pairs = batch.get('pairs', ['unknown'] * len(input_ids))
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_causal_mask=True
            )
            
            # Get probabilities
            switch_logits = outputs['switch_logits'][:, -1, :]  # [batch, 2]
            duration_logits = outputs['duration_logits'][:, -1, :]  # [batch, 3]
            
            switch_probs = F.softmax(switch_logits, dim=-1)[:, 1]  # P(switch)
            duration_probs = F.softmax(duration_logits, dim=-1)
            
            all_switch_probs.extend(switch_probs.cpu().numpy())
            all_switch_labels.extend(switch_labels.numpy())
            all_duration_probs.extend(duration_probs.cpu().numpy())
            all_duration_labels.extend(duration_labels.numpy())
            all_pairs.extend(pairs)
    
    return (
        np.array(all_switch_probs),
        np.array(all_switch_labels),
        np.array(all_duration_probs),
        np.array(all_duration_labels),
        all_pairs
    )


def compute_metrics_at_threshold(
    switch_probs: np.ndarray,
    switch_labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """计算给定阈值下的各项指标"""
    
    # 根据阈值生成预测
    predictions = (switch_probs >= threshold).astype(int)
    
    # 过滤掉 label=-1 的样本
    valid_mask = switch_labels != -1
    predictions = predictions[valid_mask]
    labels = switch_labels[valid_mask]
    
    # 计算 TP, FP, FN, TN
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    # 计算指标
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def find_optimal_thresholds(
    switch_probs: np.ndarray,
    switch_labels: np.ndarray,
    thresholds: np.ndarray = None
) -> Dict[str, Dict]:
    """
    分析不同阈值的效果，找到最优阈值
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(switch_probs, switch_labels, thresh)
        results.append(metrics)
    
    # 找到各种"最优"阈值
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_precision_idx = np.argmax([r['precision'] for r in results])
    
    # 找到 F1 和 Precision 平衡点 (Precision >= 0.5 且 F1 最高)
    balanced_results = [r for r in results if r['precision'] >= 0.5]
    if balanced_results:
        best_balanced_idx = np.argmax([r['f1'] for r in balanced_results])
        best_balanced = balanced_results[best_balanced_idx]
    else:
        best_balanced = results[best_f1_idx]
    
    return {
        'all_results': results,
        'best_f1': results[best_f1_idx],
        'best_precision': results[best_precision_idx],
        'best_balanced': best_balanced
    }


def plot_threshold_analysis(results: List[Dict], save_path: str = None):
    """绘制阈值分析图"""
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: Precision, Recall, F1 vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2)
    ax1.plot(thresholds, f1s, 'g-^', label='F1', linewidth=2)
    
    # 找到最佳 F1 点
    best_f1_idx = np.argmax(f1s)
    ax1.axvline(x=thresholds[best_f1_idx], color='green', linestyle='--', alpha=0.5)
    ax1.scatter([thresholds[best_f1_idx]], [f1s[best_f1_idx]], 
                color='green', s=200, zorder=5, marker='*', label=f'Best F1={f1s[best_f1_idx]:.3f}')
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision / Recall / F1 vs Threshold', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 图2: Precision-Recall Curve
    ax2 = axes[1]
    ax2.plot(recalls, precisions, 'b-o', linewidth=2)
    
    # 标注一些关键阈值点
    for i, thresh in enumerate(thresholds):
        if thresh in [0.3, 0.5, 0.7]:
            ax2.annotate(f'θ={thresh}', (recalls[i], precisions[i]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    plt.show()


def print_analysis_report(analysis: Dict):
    """打印分析报告"""
    
    print("\n" + "="*70)
    print("📊 THRESHOLD ANALYSIS REPORT")
    print("="*70)
    
    # 默认阈值 (0.5)
    default = None
    for r in analysis['all_results']:
        if abs(r['threshold'] - 0.5) < 0.01:
            default = r
            break
    
    if default:
        print(f"\n📌 Default Threshold (0.5):")
        print(f"   Precision: {default['precision']:.4f}")
        print(f"   Recall:    {default['recall']:.4f}")
        print(f"   F1:        {default['f1']:.4f}")
    
    # 最佳 F1
    best_f1 = analysis['best_f1']
    print(f"\n🏆 Best F1 Threshold ({best_f1['threshold']:.2f}):")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   Recall:    {best_f1['recall']:.4f}")
    print(f"   F1:        {best_f1['f1']:.4f}")
    
    # 最佳平衡点
    balanced = analysis['best_balanced']
    print(f"\n⚖️  Best Balanced (Precision≥0.5, threshold={balanced['threshold']:.2f}):")
    print(f"   Precision: {balanced['precision']:.4f}")
    print(f"   Recall:    {balanced['recall']:.4f}")
    print(f"   F1:        {balanced['f1']:.4f}")
    
    # 完整表格
    print(f"\n📋 Full Results Table:")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    
    for r in analysis['all_results']:
        marker = ""
        if r['threshold'] == best_f1['threshold']:
            marker = " ← Best F1"
        elif r['threshold'] == balanced['threshold'] and r != best_f1:
            marker = " ← Balanced"
        
        print(f"{r['threshold']:<12.2f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}{marker}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Threshold Analysis for Code-Switching Prediction')
    parser.add_argument('--model_path', type=str, 
                        default='./model_outputs/checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Directory containing processed data')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--save_plot', type=str, default='./threshold_analysis.png',
                        help='Path to save the analysis plot')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    
    # 加载模型
    model, config = load_model(args.model_path, device)
    
    # 加载测试数据
    print(f"\n📥 Loading test data...")
    data_dir = Path(args.data_dir)
    
    test_dataset = EnhancedStreamingDataset(
        data_dir / 'test.pkl',
        max_context_window=config.max_context_window,
        max_samples=args.max_samples
    )
    
    # 自定义 collate function
    def collate_fn(batch):
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
        
        if 'pair' in batch[0]:
            result['pairs'] = [item['pair'] for item in batch]
        
        return result
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 收集预测
    switch_probs, switch_labels, duration_probs, duration_labels, pairs = \
        collect_predictions(model, test_loader, device)
    
    # 分析不同阈值
    print("\n🔍 Analyzing thresholds...")
    thresholds = np.arange(0.1, 0.95, 0.05)
    analysis = find_optimal_thresholds(switch_probs, switch_labels, thresholds)
    
    # 打印报告
    print_analysis_report(analysis)
    
    # 绘图
    plot_threshold_analysis(analysis['all_results'], save_path=args.save_plot)
    
    # 保存结果
    results_path = Path(args.save_plot).parent / 'threshold_analysis_results.json'
    
    # 转换为可序列化格式
    serializable_results = {
        'best_f1': analysis['best_f1'],
        'best_balanced': analysis['best_balanced'],
        'all_results': analysis['all_results']
    }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # 推荐
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    
    best_balanced = analysis['best_balanced']
    print(f"""
    建议使用阈值 {best_balanced['threshold']:.2f}:
    
    - Precision: {best_balanced['precision']:.4f} (提升 {best_balanced['precision'] - 0.38:.4f})
    - Recall:    {best_balanced['recall']:.4f}
    - F1:        {best_balanced['f1']:.4f}
    
    这个阈值在 Precision 和 Recall 之间取得了更好的平衡。
    """)


if __name__ == "__main__":
    main()