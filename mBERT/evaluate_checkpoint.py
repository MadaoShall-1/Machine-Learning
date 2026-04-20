import argparse
import json
import pathlib
import __main__

import torch
from torch.utils.data import DataLoader

import model


def make_checkpoint_compatible() -> None:
    """Allow Linux-saved checkpoints to load on Windows."""
    pathlib.PosixPath = pathlib.WindowsPath
    __main__.ModelConfig = model.ModelConfig


def load_model_config(payload):
    """Support both legacy pickled config objects and dict-based configs."""
    if isinstance(payload, dict):
        return model.ModelConfig.from_dict(payload)
    return payload


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


def evaluate_model(net, loader, device, threshold, num_duration_classes):
    metrics = model.CodeSwitchMetrics(num_duration_classes=num_duration_classes)
    net.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            switch_labels = batch["switch_labels"].to(device)
            duration_labels = batch["duration_labels"].to(device)
            pairs = batch.get("pairs", None)

            outputs = net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                apply_causal_mask=True,
            )

            last_positions = model.last_active_index(attention_mask)
            batch_indices = torch.arange(input_ids.size(0), device=device)
            switch_logits = outputs["switch_logits"][batch_indices, last_positions, :].unsqueeze(1)
            duration_logits = outputs["duration_logits"][batch_indices, last_positions, :].unsqueeze(1)
            switch_labels = switch_labels.unsqueeze(1)
            duration_labels = duration_labels.unsqueeze(1)

            if pairs:
                for i, pair in enumerate(pairs):
                    metrics.update(
                        switch_logits[i : i + 1],
                        duration_logits[i : i + 1],
                        switch_labels[i : i + 1],
                        duration_labels[i : i + 1],
                        language_pair=pair,
                        threshold=threshold,
                    )
            else:
                metrics.update(
                    switch_logits,
                    duration_logits,
                    switch_labels,
                    duration_labels,
                    threshold=threshold,
                )

    return metrics.compute()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the processed test set."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed_data/test.pkl",
        help="Path to processed test split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.8],
        help="One or more switch thresholds to evaluate.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on prediction points for faster checks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to write metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    make_checkpoint_compatible()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    config = load_model_config(checkpoint["config"])

    net = model.CausalCodeSwitchModel(config)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)

    dataset = model.EnhancedStreamingDataset(
        pathlib.Path(args.data_path),
        max_context_window=config.max_context_window,
        max_samples=args.max_samples,
        pad_token_id=config.pad_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = {}
    for threshold in args.thresholds:
        metrics = evaluate_model(
            net=net,
            loader=loader,
            device=device,
            threshold=threshold,
            num_duration_classes=config.num_duration_classes,
        )
        results[str(threshold)] = metrics
        print(f"Threshold {threshold}")
        print(json.dumps(metrics, indent=2))

    payload = {
        "model_path": str(pathlib.Path(args.model_path).resolve()),
        "data_path": str(pathlib.Path(args.data_path).resolve()),
        "device": str(device),
        "epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics", {}),
        "thresholds": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
