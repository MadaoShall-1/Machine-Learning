"""
Threshold analysis for code-switching prediction.

This script selects a switch threshold on the validation split and can
optionally report final metrics on the test split using the chosen threshold.
"""

import __main__
import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, ".")
from model import CausalCodeSwitchModel, EnhancedStreamingDataset, ModelConfig, last_active_index


def make_checkpoint_compatible() -> None:
    """Allow legacy checkpoints saved on Linux or via __main__ dataclasses."""
    pathlib.PosixPath = pathlib.WindowsPath
    __main__.ModelConfig = ModelConfig


def load_model_config(payload):
    if isinstance(payload, dict):
        return ModelConfig.from_dict(payload)
    return payload


def load_model(model_path: str, device: str = "cuda"):
    """Load a trained model checkpoint."""
    print(f"Loading model from {model_path}")
    make_checkpoint_compatible()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = load_model_config(checkpoint.get("config", {})) if "config" in checkpoint else ModelConfig()

    net = CausalCodeSwitchModel(config)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()

    print("Model loaded successfully")
    return net, config


def collate_fn(batch):
    result = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "switch_labels": torch.stack([item["switch_label"] for item in batch]),
        "duration_labels": torch.stack([item["duration_label"] for item in batch]),
    }
    if "pair" in batch[0]:
        result["pairs"] = [item["pair"] for item in batch]
    return result


def collect_predictions(
    net,
    data_loader: DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Collect switch and duration predictions for a split."""
    all_switch_probs = []
    all_switch_labels = []
    all_duration_probs = []
    all_duration_labels = []
    all_pairs = []

    print("Collecting predictions...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            switch_labels = batch["switch_labels"]
            duration_labels = batch["duration_labels"]
            pairs = batch.get("pairs", ["unknown"] * len(input_ids))

            outputs = net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_causal_mask=True,
            )

            last_positions = last_active_index(attention_mask)
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            switch_logits = outputs["switch_logits"][batch_indices, last_positions, :]
            duration_logits = outputs["duration_logits"][batch_indices, last_positions, :]

            switch_probs = F.softmax(switch_logits, dim=-1)[:, 1]
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
        all_pairs,
    )


def compute_metrics_at_threshold(
    switch_probs: np.ndarray,
    switch_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predictions = (switch_probs >= threshold).astype(int)

    valid_mask = switch_labels != -1
    predictions = predictions[valid_mask]
    labels = switch_labels[valid_mask]

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_thresholds(
    switch_probs: np.ndarray,
    switch_labels: np.ndarray,
    thresholds: np.ndarray = None,
) -> Dict[str, Dict]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = [compute_metrics_at_threshold(switch_probs, switch_labels, thresh) for thresh in thresholds]
    best_f1 = max(results, key=lambda item: item["f1"])
    best_precision = max(results, key=lambda item: item["precision"])
    balanced_results = [item for item in results if item["precision"] >= 0.5]
    best_balanced = max(balanced_results, key=lambda item: item["f1"]) if balanced_results else best_f1

    return {
        "all_results": results,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_balanced": best_balanced,
    }


def plot_threshold_analysis(results: List[Dict], save_path: str = None):
    thresholds = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(thresholds, precisions, "b-o", label="Precision", linewidth=2)
    ax1.plot(thresholds, recalls, "r-s", label="Recall", linewidth=2)
    ax1.plot(thresholds, f1s, "g-^", label="F1", linewidth=2)

    best_f1_idx = int(np.argmax(f1s))
    ax1.axvline(x=thresholds[best_f1_idx], color="green", linestyle="--", alpha=0.5)
    ax1.scatter(
        [thresholds[best_f1_idx]],
        [f1s[best_f1_idx]],
        color="green",
        s=200,
        zorder=5,
        marker="*",
        label=f"Best F1={f1s[best_f1_idx]:.3f}",
    )

    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision / Recall / F1 vs Threshold", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2 = axes[1]
    ax2.plot(recalls, precisions, "b-o", linewidth=2)
    for i, thresh in enumerate(thresholds):
        if thresh in [0.3, 0.5, 0.7]:
            ax2.annotate(f"t={thresh:.2f}", (recalls[i], precisions[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curve", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def print_analysis_report(split_name: str, analysis: Dict):
    print("\n" + "=" * 70)
    print(f"THRESHOLD ANALYSIS REPORT ({split_name.upper()})")
    print("=" * 70)

    default = next((r for r in analysis["all_results"] if abs(r["threshold"] - 0.5) < 0.01), None)
    if default:
        print("\nDefault Threshold (0.5):")
        print(f"  Precision: {default['precision']:.4f}")
        print(f"  Recall:    {default['recall']:.4f}")
        print(f"  F1:        {default['f1']:.4f}")

    best_f1 = analysis["best_f1"]
    print(f"\nBest F1 Threshold ({best_f1['threshold']:.2f}):")
    print(f"  Precision: {best_f1['precision']:.4f}")
    print(f"  Recall:    {best_f1['recall']:.4f}")
    print(f"  F1:        {best_f1['f1']:.4f}")

    balanced = analysis["best_balanced"]
    print(f"\nBest Balanced (Precision >= 0.5, threshold={balanced['threshold']:.2f}):")
    print(f"  Precision: {balanced['precision']:.4f}")
    print(f"  Recall:    {balanced['recall']:.4f}")
    print(f"  F1:        {balanced['f1']:.4f}")


def build_loader(data_path: Path, config: ModelConfig, batch_size: int, max_samples: Optional[int]) -> DataLoader:
    dataset = EnhancedStreamingDataset(
        data_path,
        max_context_window=config.max_context_window,
        max_samples=max_samples,
        pad_token_id=config.pad_token_id,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def main():
    parser = argparse.ArgumentParser(description="Threshold Analysis for Code-Switching Prediction")
    parser.add_argument("--model_path", type=str, default="./model_outputs/checkpoints/best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./processed_data", help="Directory containing processed data")
    parser.add_argument("--selection_split", type=str, default="val", choices=["val", "test"], help="Split used to select the threshold. Use validation by default.")
    parser.add_argument("--report_test", action="store_true", help="Also evaluate the selected threshold on the test split.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum samples to evaluate per split")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--save_plot", type=str, default="./threshold_analysis.png", help="Path to save the analysis plot")

    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"Using device: {device}")
    net, config = load_model(args.model_path, device)

    data_dir = Path(args.data_dir)
    selection_path = data_dir / f"{args.selection_split}.pkl"
    print(f"\nLoading {args.selection_split} data from {selection_path}...")
    selection_loader = build_loader(selection_path, config, args.batch_size, args.max_samples)

    switch_probs, switch_labels, duration_probs, duration_labels, pairs = collect_predictions(net, selection_loader, device)
    print(f"\nAnalyzing thresholds on {args.selection_split}...")
    thresholds = np.arange(0.1, 0.95, 0.05)
    analysis = find_optimal_thresholds(switch_probs, switch_labels, thresholds)
    print_analysis_report(args.selection_split, analysis)
    plot_threshold_analysis(analysis["all_results"], save_path=args.save_plot)

    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "data_dir": str(data_dir.resolve()),
        "selection_split": args.selection_split,
        "selected_threshold": analysis["best_balanced"],
        "all_results": analysis["all_results"],
    }

    if args.report_test:
        test_path = data_dir / "test.pkl"
        if test_path.exists():
            print("\nEvaluating selected threshold on test split...")
            test_loader = build_loader(test_path, config, args.batch_size, args.max_samples)
            test_switch_probs, test_switch_labels, _, _, _ = collect_predictions(net, test_loader, device)
            payload["test_metrics"] = compute_metrics_at_threshold(
                test_switch_probs,
                test_switch_labels,
                analysis["best_balanced"]["threshold"],
            )
        else:
            print("\nSkipping test evaluation: test.pkl not found.")

    results_path = Path(args.save_plot).parent / "threshold_analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to {results_path}")
    best_balanced = analysis["best_balanced"]
    print("\nRecommendation")
    print(f"  Select threshold {best_balanced['threshold']:.2f} based on the {args.selection_split} split.")
    print(f"  Precision: {best_balanced['precision']:.4f}")
    print(f"  Recall:    {best_balanced['recall']:.4f}")
    print(f"  F1:        {best_balanced['f1']:.4f}")


if __name__ == "__main__":
    main()
