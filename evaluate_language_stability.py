import argparse
import ast
import json
import hashlib
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from data_processing import LanguageDetector


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DATA_DIR = Path("data")
DEFAULT_OUTPUT = Path("language_stability_report.json")
SUPPORTED_FILES = {
    "spanish_eng": DATA_DIR / "Spanish_eng.csv",
    "arabic_eng": DATA_DIR / "Arabic_eng.csv",
    "hindi_eng": DATA_DIR / "Hindi_eng.csv",
    "chinese_eng": DATA_DIR / "Chinese_eng.csv",
}


def resolve_supported_file(path: Path) -> Path:
    if path.exists():
        return path

    parent = path.parent
    target = path.name.lower()
    for candidate in parent.glob('*.csv'):
        if candidate.name.lower() == target:
            return candidate
    return path


def infer_language_pair(first_lang: str, second_lang: str) -> str:
    lang_map = {
        "Spanish": "spanish",
        "Arabic": "arabic",
        "Hindi": "hindi",
        "Chinese": "chinese",
        "Mandarin": "chinese",
        "English": "eng",
    }

    first = lang_map.get(first_lang, first_lang.lower())
    second = lang_map.get(second_lang, second_lang.lower())

    if first == "eng":
        first, second = second, first

    return f"{first}_{second}"


def extract_sentences(df: pd.DataFrame) -> List[Dict]:
    sentences = []

    for idx, row in df.iterrows():
        result = row.get("data_generation_result", [])
        first_lang = row.get("first_language", "Unknown")
        second_lang = row.get("second_language", "Unknown")
        inferred_pair = infer_language_pair(first_lang, second_lang)

        sentence_list: List[str] = []
        if isinstance(result, list):
            sentence_list = [s for s in result if isinstance(s, str) and s.strip()]
        elif isinstance(result, str):
            try:
                parsed = ast.literal_eval(result)
                if isinstance(parsed, list):
                    sentence_list = [s for s in parsed if isinstance(s, str) and s.strip()]
                elif result.strip():
                    sentence_list = [result]
            except (ValueError, SyntaxError):
                if result.strip():
                    sentence_list = [result]

        for sent in sentence_list:
            sentences.append(
                {
                    "text": sent,
                    "first_language": first_lang,
                    "second_language": second_lang,
                    "inferred_pair": inferred_pair,
                    "original_idx": int(idx),
                }
            )

    return sentences


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def add_boundary_spaces(text: str) -> str:
    return f"  {text.strip()}  "


def lowercase_text(text: str) -> str:
    return text.lower()


def strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[.!?;,:，。！？；：]+$", "", text.strip())


def build_variants(text: str) -> Dict[str, str]:
    words = text.split()
    drop_idx = deterministic_drop_index(text, len(words))

    return {
        "original": text,
        "collapsed_spaces": collapse_spaces(text),
        "boundary_spaces": add_boundary_spaces(text),
        "lowercase": lowercase_text(text),
        "strip_terminal_punct": strip_terminal_punctuation(text),
        "unicode_nfkc": unicodedata.normalize("NFKC", text),
        "drop_first_word": " ".join(words[1:]) if len(words) > 1 else text,
        "drop_last_word": " ".join(words[:-1]) if len(words) > 1 else text,
        "drop_one_word": (
            " ".join(words[:drop_idx] + words[drop_idx + 1 :]) if len(words) > 2 else text
        ),
        "prefix_half": " ".join(words[: max(1, len(words) // 2)]) if words else text,
        "suffix_half": " ".join(words[-max(1, len(words) // 2) :]) if words else text,
    }


def deterministic_drop_index(text: str, word_count: int) -> int:
    if word_count <= 2:
        return 0

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % word_count


def detect_word_languages(
    detector: LanguageDetector, text: str, pair: str
) -> Optional[Tuple[List[str], List[str]]]:
    words = text.split()
    if not words:
        return None

    sentence_pattern = detector.detect_sentence_languages(text, pair)
    langs = []
    for idx, word in enumerate(words):
        lang = detector.detect_token_language(
            token_text=word,
            full_sentence=text,
            token_idx=idx,
            pair=pair,
            sentence_pattern=sentence_pattern,
        )
        langs.append(lang)

    return words, langs


def language_sequence_agreement(
    baseline: Optional[Tuple[List[str], List[str]]],
    candidate: Optional[Tuple[List[str], List[str]]],
) -> Dict[str, Optional[float]]:
    if baseline is None or candidate is None:
        return {
            "word_count_match": False,
            "language_accuracy": None,
            "exact_match": False,
        }

    base_words, base_langs = baseline
    cand_words, cand_langs = candidate

    if len(base_words) != len(cand_words):
        return {
            "word_count_match": False,
            "language_accuracy": None,
            "exact_match": False,
        }

    correct = sum(1 for b, c in zip(base_langs, cand_langs) if b == c)
    accuracy = correct / len(base_langs) if base_langs else 1.0
    return {
        "word_count_match": True,
        "language_accuracy": accuracy,
        "exact_match": base_langs == cand_langs,
    }


def language_overlap_agreement(
    baseline: Optional[Tuple[List[str], List[str]]],
    candidate: Optional[Tuple[List[str], List[str]]],
    mode: str,
) -> Dict[str, Optional[float]]:
    if baseline is None or candidate is None:
        return {
            "word_count_match": False,
            "overlap_token_count": 0,
            "language_accuracy": None,
            "exact_match": False,
        }

    base_words, base_langs = baseline
    cand_words, cand_langs = candidate

    if mode == "drop_first_word":
        base_words = base_words[1:]
        base_langs = base_langs[1:]
    elif mode == "drop_last_word":
        base_words = base_words[:-1]
        base_langs = base_langs[:-1]
    elif mode == "drop_one_word":
        drop_idx = deterministic_drop_index(" ".join(base_words), len(base_words))
        base_words = base_words[:drop_idx] + base_words[drop_idx + 1 :]
        base_langs = base_langs[:drop_idx] + base_langs[drop_idx + 1 :]
    elif mode == "prefix_half":
        keep = max(1, len(base_words) // 2)
        base_words = base_words[:keep]
        base_langs = base_langs[:keep]
    elif mode == "suffix_half":
        keep = max(1, len(base_words) // 2)
        base_words = base_words[-keep:]
        base_langs = base_langs[-keep:]

    if len(base_words) != len(cand_words):
        return {
            "word_count_match": False,
            "overlap_token_count": 0,
            "language_accuracy": None,
            "exact_match": False,
        }

    if base_words != cand_words:
        return {
            "word_count_match": False,
            "overlap_token_count": 0,
            "language_accuracy": None,
            "exact_match": False,
        }

    correct = sum(1 for b, c in zip(base_langs, cand_langs) if b == c)
    accuracy = correct / len(base_langs) if base_langs else 1.0
    return {
        "word_count_match": True,
        "overlap_token_count": len(base_langs),
        "language_accuracy": accuracy,
        "exact_match": base_langs == cand_langs,
    }


def evaluate_sentence(
    detector: LanguageDetector, sentence_meta: Dict, repeats: int
) -> Dict:
    text = sentence_meta["text"]
    pair = sentence_meta["inferred_pair"]

    baseline = detect_word_languages(detector, text, pair)
    if baseline is None:
        return {
            "pair": pair,
            "text": text,
            "skipped": True,
            "reason": "empty_after_split",
        }

    repeated_runs = [detect_word_languages(detector, text, pair) for _ in range(repeats)]
    repeated_exact = all(run == baseline for run in repeated_runs)

    variant_results = {}
    for name, variant_text in build_variants(text).items():
        candidate = detect_word_languages(detector, variant_text, pair)
        if name in {"drop_first_word", "drop_last_word", "drop_one_word", "prefix_half", "suffix_half"}:
            variant_results[name] = language_overlap_agreement(baseline, candidate, name)
        else:
            variant_results[name] = language_sequence_agreement(baseline, candidate)

    return {
        "pair": pair,
        "text": text,
        "skipped": False,
        "baseline_words": baseline[0],
        "baseline_langs": baseline[1],
        "repeated_run_exact_match": repeated_exact,
        "variants": variant_results,
    }


def summarize(results: List[Dict]) -> Dict:
    summary = {
        "overall": defaultdict(float),
        "by_pair": defaultdict(lambda: defaultdict(float)),
    }

    kept = [r for r in results if not r.get("skipped")]
    summary["overall"]["sentence_count"] = len(kept)

    if not kept:
        return {
            "overall": dict(summary["overall"]),
            "by_pair": {},
        }

    variant_names = list(kept[0]["variants"].keys())

    overall_repeat = sum(1 for r in kept if r["repeated_run_exact_match"])
    summary["overall"]["repeat_exact_match_rate"] = overall_repeat / len(kept)

    for variant in variant_names:
        valid = [
            r["variants"][variant]["language_accuracy"]
            for r in kept
            if r["variants"][variant]["language_accuracy"] is not None
        ]
        exact = sum(1 for r in kept if r["variants"][variant]["exact_match"])
        word_count_match = sum(1 for r in kept if r["variants"][variant]["word_count_match"])

        summary["overall"][f"{variant}_exact_match_rate"] = exact / len(kept)
        summary["overall"][f"{variant}_word_count_match_rate"] = word_count_match / len(kept)
        summary["overall"][f"{variant}_avg_language_accuracy"] = (
            sum(valid) / len(valid) if valid else None
        )

    pair_groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in kept:
        pair_groups[item["pair"]].append(item)

    by_pair = {}
    for pair, group in pair_groups.items():
        pair_summary = {
            "sentence_count": len(group),
            "repeat_exact_match_rate": sum(
                1 for r in group if r["repeated_run_exact_match"]
            )
            / len(group),
        }

        for variant in variant_names:
            valid = [
                r["variants"][variant]["language_accuracy"]
                for r in group
                if r["variants"][variant]["language_accuracy"] is not None
            ]
            pair_summary[f"{variant}_exact_match_rate"] = sum(
                1 for r in group if r["variants"][variant]["exact_match"]
            ) / len(group)
            pair_summary[f"{variant}_word_count_match_rate"] = sum(
                1 for r in group if r["variants"][variant]["word_count_match"]
            ) / len(group)
            pair_summary[f"{variant}_avg_language_accuracy"] = (
                sum(valid) / len(valid) if valid else None
            )

        by_pair[pair] = pair_summary

    return {
        "overall": dict(summary["overall"]),
        "by_pair": by_pair,
    }


def load_sentence_sample(limit_per_pair: int, seed: int) -> List[Dict]:
    random.seed(seed)
    all_sentences: List[Dict] = []

    for pair, csv_path in SUPPORTED_FILES.items():
        csv_path = resolve_supported_file(csv_path)
        if not csv_path.exists():
            print(f"[skip] Missing file for {pair}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        sentences = [s for s in extract_sentences(df) if s["inferred_pair"] == pair]
        if limit_per_pair > 0 and len(sentences) > limit_per_pair:
            sentences = random.sample(sentences, limit_per_pair)

        print(f"[load] {pair}: {len(sentences)} sampled sentences")
        all_sentences.extend(sentences)

    return all_sentences


def print_summary(summary: Dict) -> None:
    overall = summary["overall"]
    print("\nOverall")
    print(f"  sentences: {int(overall.get('sentence_count', 0))}")
    print(f"  repeat exact match: {overall.get('repeat_exact_match_rate', 0.0):.4f}")

    for key, value in overall.items():
        if key in {"sentence_count", "repeat_exact_match_rate"}:
            continue
        if value is None:
            print(f"  {key}: null")
        else:
            print(f"  {key}: {value:.4f}")

    print("\nBy pair")
    for pair, pair_summary in summary["by_pair"].items():
        print(f"  {pair}")
        for key, value in pair_summary.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")


def run_stability_eval(
    sentences: List[Dict],
    backend: str,
    repeats: int,
    cache_dir: Path,
) -> Tuple[Dict, List[Dict]]:
    detector = LanguageDetector(
        spanish_lid_backend=backend,
        cache_dir=cache_dir,
    )
    results = []
    for sentence_meta in tqdm(sentences, desc=f"Evaluating stability ({backend})"):
        results.append(evaluate_sentence(detector, sentence_meta, repeats=repeats))
    return summarize(results), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate language detection stability.")
    parser.add_argument(
        "--limit-per-pair",
        type=int,
        default=0,
        help="0 means use all sentences for each language pair.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--spanish-lid-backend",
        type=str,
        default="codeswitch",
        choices=["papluca", "codeswitch"],
        help="Spanish-English LID backend to evaluate when not comparing both.",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Run both papluca and codeswitch backends and save side-by-side summaries.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Include per-sentence details in the JSON report.",
    )
    args = parser.parse_args()

    sentences = load_sentence_sample(limit_per_pair=args.limit_per_pair, seed=args.seed)

    report = {
        "config": {
            "limit_per_pair": args.limit_per_pair,
            "repeats": args.repeats,
            "seed": args.seed,
        }
    }
    detector_cache_root = Path.cwd() / "cache" / "hf_models"

    if args.compare_backends:
        summaries = {}
        details = {}
        for backend in ["papluca", "codeswitch"]:
            summary, results = run_stability_eval(
                sentences,
                backend=backend,
                repeats=args.repeats,
                cache_dir=detector_cache_root / backend,
            )
            summaries[backend] = summary
            if args.save_details:
                details[backend] = results
            print(f"\nBackend: {backend}")
            print_summary(summary)
        report["summaries"] = summaries
        if args.save_details:
            report["details"] = details
    else:
        summary, results = run_stability_eval(
            sentences,
            backend=args.spanish_lid_backend,
            repeats=args.repeats,
            cache_dir=detector_cache_root / args.spanish_lid_backend,
        )
        report["summary"] = summary
        report["config"]["spanish_lid_backend"] = args.spanish_lid_backend
        if args.save_details:
            report["details"] = results
        print_summary(summary)

    print(f"\nSaved report to {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
