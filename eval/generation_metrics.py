"""Compute automatic metrics for fixed generation samples."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fixed generation samples with simple text metrics")
    parser.add_argument("--input-json", required=True, help="Generation JSON from eval/generate_samples.py")
    parser.add_argument("--output-json", required=True, help="Path for metrics JSON")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary path")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def distinct_n(tokens: list[str], n: int) -> float | None:
    items = ngrams(tokens, n)
    if not items:
        return None
    return len(set(items)) / len(items)


def repeated_ngram_fraction(tokens: list[str], n: int) -> float | None:
    items = ngrams(tokens, n)
    if not items:
        return None
    counts = Counter(items)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(items)


def max_ngram_count(tokens: list[str], n: int) -> int:
    items = ngrams(tokens, n)
    if not items:
        return 0
    return max(Counter(items).values(), default=0)


def strip_prompt(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def line_repeat_count(completion: str) -> int:
    lines = [line.strip().lower() for line in completion.splitlines() if line.strip()]
    if not lines:
        return 0
    return max(Counter(lines).values(), default=0)


def safe_mean(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return None
    return mean(numeric)


def score_completion(prompt: str, completion_row: dict[str, Any]) -> dict[str, Any]:
    full_text = str(completion_row.get("text", ""))
    completion = strip_prompt(full_text, prompt)
    tokens = tokenize(completion)
    unique_tokens = len(set(tokens))
    token_count = len(tokens)
    max_4gram = max_ngram_count(tokens, 4)
    line_repeat = line_repeat_count(completion)
    rep_bigram = repeated_ngram_fraction(tokens, 2)
    rep_trigram = repeated_ngram_fraction(tokens, 3)
    d1 = distinct_n(tokens, 1)
    d2 = distinct_n(tokens, 2)
    unique_ratio = unique_tokens / token_count if token_count else None
    bad_loop = (
        max_4gram >= 3
        or line_repeat >= 3
        or (rep_trigram is not None and rep_trigram >= 0.20)
        or (d2 is not None and d2 <= 0.55 and token_count >= 40)
    )

    return {
        "model": completion_row["model"],
        "chars": len(completion),
        "tokens": token_count,
        "new_tokens_reported": completion_row.get("new_tokens"),
        "unique_token_ratio": unique_ratio,
        "distinct_1": d1,
        "distinct_2": d2,
        "repeated_bigram_fraction": rep_bigram,
        "repeated_trigram_fraction": rep_trigram,
        "max_repeated_4gram_count": max_4gram,
        "max_repeated_line_count": line_repeat,
        "bad_loop_detected": bad_loop,
    }


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_tokens: list[str] = []
    for row in rows:
        all_tokens.extend(row["_tokens"])

    return {
        "samples": len(rows),
        "avg_chars": safe_mean([row["chars"] for row in rows]),
        "avg_tokens": safe_mean([row["tokens"] for row in rows]),
        "avg_new_tokens_reported": safe_mean([row["new_tokens_reported"] for row in rows]),
        "corpus_distinct_1": distinct_n(all_tokens, 1),
        "corpus_distinct_2": distinct_n(all_tokens, 2),
        "mean_unique_token_ratio": safe_mean([row["unique_token_ratio"] for row in rows]),
        "mean_distinct_1": safe_mean([row["distinct_1"] for row in rows]),
        "mean_distinct_2": safe_mean([row["distinct_2"] for row in rows]),
        "mean_repeated_bigram_fraction": safe_mean([row["repeated_bigram_fraction"] for row in rows]),
        "mean_repeated_trigram_fraction": safe_mean([row["repeated_trigram_fraction"] for row in rows]),
        "max_repeated_4gram_count": max((int(row["max_repeated_4gram_count"]) for row in rows), default=0),
        "bad_loop_count": sum(1 for row in rows if row["bad_loop_detected"]),
    }


def build_metrics(generation_results: dict[str, Any]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_sample: list[dict[str, Any]] = []

    for sample in generation_results.get("samples", []):
        prompt = str(sample.get("prompt", ""))
        sample_metrics = {
            "prompt_id": sample.get("prompt_id"),
            "category": sample.get("category"),
            "prompt": prompt,
            "completions": [],
        }
        for completion_row in sample.get("completions", []):
            scored = score_completion(prompt, completion_row)
            completion_text = strip_prompt(str(completion_row.get("text", "")), prompt)
            model_row = {**scored, "_tokens": tokenize(completion_text)}
            by_model[scored["model"]].append(model_row)
            sample_metrics["completions"].append(scored)
        per_sample.append(sample_metrics)

    summary = {model: summarize_model(rows) for model, rows in sorted(by_model.items())}
    for rows in by_model.values():
        for row in rows:
            row.pop("_tokens", None)

    return {
        "source_settings": generation_results.get("settings", {}),
        "source_models": generation_results.get("models", {}),
        "summary": summary,
        "per_sample": per_sample,
    }


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 4 Generation Metrics",
        "",
        "Automatic metrics over the fixed prompt generations. These are support metrics, not a replacement for reading the raw samples.",
        "",
        "| Model | Samples | Avg tokens | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in metrics["summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    format_value(row["samples"], 0),
                    format_value(row["avg_tokens"], 1),
                    format_value(row["corpus_distinct_1"]),
                    format_value(row["corpus_distinct_2"]),
                    format_value(row["mean_repeated_bigram_fraction"]),
                    format_value(row["mean_repeated_trigram_fraction"]),
                    format_value(row["bad_loop_count"], 0),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Lower repeated-ngram fractions and fewer bad-loop detections are better. Higher distinct scores are usually better, but very high diversity can also reflect incoherence on tiny sample sets.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json)
    output_json = Path(args.output_json)
    generation_results = json.loads(input_path.read_text(encoding="utf-8"))
    metrics = build_metrics(generation_results)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), metrics)
    print(json.dumps({"output_json": str(output_json), "output_md": args.output_md}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
