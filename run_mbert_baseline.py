import argparse
import subprocess
import sys
from pathlib import Path


MBERT_MODEL_NAME = "bert-base-multilingual-cased"


def run_step(command):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convenience runner for the mBERT baseline pipeline."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="processed_data_mbert",
        help="Directory for mBERT-processed data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_outputs_mbert",
        help="Directory for mBERT checkpoints/results.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_mbert",
        help="Directory for processing cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Training device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Evaluation threshold during training.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on training points for faster experiments.",
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Skip data preprocessing if mBERT processed files already exist.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python_exe = args.python

    if not args.skip_processing:
        processing_cmd = [
            python_exe,
            "data_processing.py",
            "--model_name",
            MBERT_MODEL_NAME,
            "--output_dir",
            args.processed_dir,
            "--cache_dir",
            args.cache_dir,
        ]
        run_step(processing_cmd)

    if not args.skip_training:
        training_cmd = [
            python_exe,
            "model.py",
            "--model_name",
            MBERT_MODEL_NAME,
            "--data_dir",
            args.processed_dir,
            "--output_dir",
            args.output_dir,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--max_epochs",
            str(args.max_epochs),
            "--threshold",
            str(args.threshold),
        ]

        if args.max_samples is not None:
            training_cmd.extend(["--max_samples", str(args.max_samples)])

        run_step(training_cmd)


if __name__ == "__main__":
    main()
