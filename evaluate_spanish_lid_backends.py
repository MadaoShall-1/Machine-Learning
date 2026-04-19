import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

from tqdm import tqdm

from data_processing import LanguageDetector


LINCE_BASE_URL = (
    "https://ritual.uh.edu/lince/libaccess/"
    "eyJ1c2VybmFtZSI6ICJodWdnaW5nZmFjZSBubHAiLCAidXNlcl9pZCI6IDExMSwgImVtYWlsIjogImR1bW15QGVtYWlsLmNvbSJ9"
)
SPAENG_ZIP_URL = f"{LINCE_BASE_URL}/lid_spaeng.zip"


def ensure_dataset(cache_dir: Path, zip_path: Optional[Path] = None, dataset_dir: Optional[Path] = None) -> Path:
    if dataset_dir is not None:
        dataset_dir = dataset_dir.resolve()
        if (dataset_dir / "dev.conll").exists():
            return dataset_dir
        if (dataset_dir / "lid_spaeng" / "dev.conll").exists():
            return dataset_dir / "lid_spaeng"
        raise FileNotFoundError(f"Could not find dev.conll under {dataset_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_path.resolve() if zip_path is not None else cache_dir / "lid_spaeng.zip"
    extract_dir = cache_dir / "lid_spaeng"

    if not zip_path.exists():
        print(f"Downloading LinCE spa-eng LID archive to {zip_path}")
        try:
            urlretrieve(SPAENG_ZIP_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download LinCE spa-eng from ritual.uh.edu. "
                "You can retry later or pass --zip-path/--dataset-dir with a local copy."
            ) from exc

    if not extract_dir.exists():
        print(f"Extracting {zip_path} to {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir / "lid_spaeng"


def parse_conll(filepath: Path) -> List[Dict]:
    sentences = []
    words: List[str] = []
    labels: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if words:
                    sentences.append({"words": words, "labels": labels})
                    words = []
                    labels = []
                continue

            if line.startswith("# sent_enum"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            words.append(parts[0])
            labels.append(parts[1])

    if words:
        sentences.append({"words": words, "labels": labels})

    return sentences


def gold_to_lang(label: str) -> str:
    if label == "lang1":
        return "en"
    if label == "lang2":
        return "es"
    return "other"


def evaluate_backend(sentences: List[Dict], backend: str, limit: int = 0) -> Dict:
    detector = LanguageDetector(
        spanish_lid_backend=backend,
        cache_dir=Path.cwd() / "cache" / "hf_models" / backend,
    )
    subset = sentences[:limit] if limit > 0 else sentences

    total_lang_tokens = 0
    correct_lang_tokens = 0
    strict_total = 0
    strict_correct = 0
    confusion = Counter()
    mismatches = []

    for sentence in tqdm(subset, desc=f"Evaluating {backend}"):
        words = sentence["words"]
        gold_labels = sentence["labels"]
        text = " ".join(words)

        prediction = detector.detect_sentence_languages(text, "spanish_eng")
        if prediction is None or len(prediction["langs"]) != len(words):
            mismatches.append(
                {
                    "text": text,
                    "reason": "prediction_length_mismatch",
                    "gold_words": words,
                    "predicted_langs": prediction["langs"] if prediction else None,
                }
            )
            continue

        sentence_correct = True
        sentence_has_lang_token = False
        for word, gold_label, pred_lang in zip(words, gold_labels, prediction["langs"]):
            mapped_gold = gold_to_lang(gold_label)
            if mapped_gold == "other":
                continue

            sentence_has_lang_token = True
            total_lang_tokens += 1
            strict_total += 1
            confusion[(mapped_gold, pred_lang)] += 1

            if pred_lang == mapped_gold:
                correct_lang_tokens += 1
            else:
                sentence_correct = False

        if sentence_has_lang_token and sentence_correct:
            strict_correct += 1

    per_class = {}
    for cls in ["en", "es"]:
        tp = confusion[(cls, cls)]
        fp = sum(confusion[(other, cls)] for other in ["en", "es"] if other != cls)
        fn = sum(confusion[(cls, other)] for other in ["en", "es"] if other != cls)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[(cls, pred)] for pred in ["en", "es"]),
        }

    return {
        "backend": backend,
        "sentence_count": len(subset),
        "evaluated_lang_tokens": total_lang_tokens,
        "token_accuracy_lang_only": correct_lang_tokens / (total_lang_tokens + 1e-10),
        "sentence_exact_match_lang_only": strict_correct / (len(subset) + 1e-10),
        "per_class": per_class,
        "confusion": {
            f"{gold}->{pred}": count
            for (gold, pred), count in sorted(confusion.items())
        },
        "prediction_failures": len(mismatches),
        "sample_failures": mismatches[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Spanish-English LID backends on LinCE spa-eng dev.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache_lince_eval"),
        help="Directory for downloaded LinCE files.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional local path to a previously downloaded lid_spaeng.zip archive.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional local directory containing dev.conll/train.conll or a lid_spaeng subdirectory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev"],
        help="Which labeled split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sentence limit for a quick comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spanish_lid_backend_eval.json"),
        help="Path to write evaluation results.",
    )
    args = parser.parse_args()

    dataset_dir = ensure_dataset(args.cache_dir, zip_path=args.zip_path, dataset_dir=args.dataset_dir)
    filename = "dev.conll" if args.split == "dev" else "train.conll"
    sentences = parse_conll(dataset_dir / filename)

    report = {
        "dataset": "LinCE lid_spaeng",
        "split": args.split,
        "sentence_count": len(sentences) if args.limit == 0 else min(args.limit, len(sentences)),
        "results": {
            backend: evaluate_backend(sentences, backend=backend, limit=args.limit)
            for backend in ["papluca", "codeswitch"]
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for backend, metrics in report["results"].items():
        print(f"\nBackend: {backend}")
        print(f"  token_accuracy_lang_only: {metrics['token_accuracy_lang_only']:.4f}")
        print(f"  sentence_exact_match_lang_only: {metrics['sentence_exact_match_lang_only']:.4f}")
        for cls, cls_metrics in metrics["per_class"].items():
            print(
                f"  {cls}: precision={cls_metrics['precision']:.4f} "
                f"recall={cls_metrics['recall']:.4f} f1={cls_metrics['f1']:.4f}"
            )

    print(f"\nSaved evaluation to {args.output}")


if __name__ == "__main__":
    main()
