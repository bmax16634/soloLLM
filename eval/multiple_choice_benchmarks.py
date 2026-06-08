"""Run lightweight multiple-choice base-LM benchmarks for v2 and GPT-2."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval import resolve_device, resolve_path
from eval.external_benchmarks import DEFAULT_V2_CHECKPOINT, DEFAULT_V2_CONFIG, load_model, model_logits


BENCHMARKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple-choice base-LM comparisons")
    parser.add_argument("--models", nargs="+", default=["v2", "gpt2"], choices=["v2", "gpt2"])
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument("--v2-checkpoint", default=str(DEFAULT_V2_CHECKPOINT))
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means full validation split")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def add_leading_space(text: str) -> str:
    if not text:
        return text
    if text[0].isspace() or text[0] in ".,;:!?)]}":
        return text
    return " " + text


def load_hellaswag() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("hellaswag", split="validation")
    examples = []
    for row in dataset:
        endings = [add_leading_space(str(ending)) for ending in row["endings"]]
        examples.append(
            {
                "id": str(row.get("ind", len(examples))),
                "context": str(row["ctx"]),
                "choices": endings,
                "label": int(row["label"]),
            }
        )
    return examples


def load_piqa() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("gimmaru/piqa", split="validation")
    examples = []
    for index, row in enumerate(dataset):
        examples.append(
            {
                "id": str(index),
                "context": str(row["goal"]),
                "choices": [add_leading_space(str(row["sol1"])), add_leading_space(str(row["sol2"]))],
                "label": int(row["label"]),
            }
        )
    return examples


def load_arc(config_name: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/ai2_arc", config_name, split="validation")
    examples = []
    for row in dataset:
        labels = [str(label) for label in row["choices"]["label"]]
        choices = [add_leading_space(str(text)) for text in row["choices"]["text"]]
        answer_key = str(row["answerKey"])
        if answer_key not in labels:
            continue
        examples.append(
            {
                "id": str(row.get("id", len(examples))),
                "context": f"Question: {row['question']}\nAnswer:",
                "choices": choices,
                "label": labels.index(answer_key),
            }
        )
    return examples


def load_winogrande() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    examples = []
    for index, row in enumerate(dataset):
        sentence = str(row["sentence"])
        if "_" not in sentence:
            continue
        prefix, suffix = sentence.split("_", maxsplit=1)
        choices = [str(row["option1"]) + suffix, str(row["option2"]) + suffix]
        examples.append(
            {
                "id": str(index),
                "context": prefix,
                "choices": choices,
                "label": int(row["answer"]) - 1,
            }
        )
    return examples


def load_examples(benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "hellaswag":
        return load_hellaswag()
    if benchmark == "piqa":
        return load_piqa()
    if benchmark == "arc_easy":
        return load_arc("ARC-Easy")
    if benchmark == "arc_challenge":
        return load_arc("ARC-Challenge")
    if benchmark == "winogrande":
        return load_winogrande()
    raise ValueError(f"unknown benchmark: {benchmark}")


def score_continuation(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
    context: str,
    continuation: str,
    *,
    device: torch.device,
    context_length: int,
) -> tuple[float, float, int]:
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if not context_ids or not continuation_ids:
        return -math.inf, -math.inf, 0

    full_ids = context_ids + continuation_ids
    offset = max(0, len(full_ids) - context_length)
    window = full_ids[offset:]
    continuation_start = max(len(context_ids) - offset, 1)
    score_positions = range(continuation_start, len(window))
    if not score_positions:
        return -math.inf, -math.inf, 0

    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    logits = model_logits(model, input_ids)[:, :-1, :]
    log_probs = torch.log_softmax(logits[0], dim=-1)

    total_logprob = 0.0
    scored_tokens = 0
    for token_index in score_positions:
        target_id = int(window[token_index])
        total_logprob += float(log_probs[token_index - 1, target_id].item())
        scored_tokens += 1

    normalized_logprob = total_logprob / scored_tokens if scored_tokens else -math.inf
    return total_logprob, normalized_logprob, scored_tokens


@torch.no_grad()
def evaluate_benchmark(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
    benchmark: str,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    model_name: str,
) -> dict[str, Any]:
    correct_raw = 0
    correct_norm = 0
    evaluated = 0
    skipped = 0
    total_choice_tokens = 0

    model.eval()
    iterator = tqdm(examples, desc=f"{model_name} {benchmark}", dynamic_ncols=True, disable=disable_progress)
    for example in iterator:
        raw_scores: list[float] = []
        norm_scores: list[float] = []
        choice_token_counts: list[int] = []
        for choice in example["choices"]:
            raw, norm, tokens = score_continuation(
                model,
                tokenizer,
                str(example["context"]),
                str(choice),
                device=device,
                context_length=context_length,
            )
            raw_scores.append(raw)
            norm_scores.append(norm)
            choice_token_counts.append(tokens)

        if not raw_scores or all(not math.isfinite(score) for score in raw_scores):
            skipped += 1
            continue

        label = int(example["label"])
        raw_pred = max(range(len(raw_scores)), key=lambda index: raw_scores[index])
        norm_pred = max(range(len(norm_scores)), key=lambda index: norm_scores[index])
        correct_raw += int(raw_pred == label)
        correct_norm += int(norm_pred == label)
        evaluated += 1
        total_choice_tokens += sum(choice_token_counts)

    return {
        "benchmark": benchmark,
        "examples": len(examples),
        "evaluated_examples": evaluated,
        "skipped_examples": skipped,
        "accuracy": correct_raw / evaluated if evaluated else None,
        "accuracy_norm": correct_norm / evaluated if evaluated else None,
        "correct": correct_raw,
        "correct_norm": correct_norm,
        "avg_choice_tokens": total_choice_tokens / evaluated if evaluated else None,
        "context_length": context_length,
    }


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def format_max_examples(value: Any) -> str:
    if isinstance(value, int) and value <= 0:
        return "full validation split"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Multiple-Choice Base-LM Benchmark Results",
        "",
        "Choices are scored by conditional log-likelihood under the base LM. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.",
        "",
        f"- Context length: `{result['settings']['context_length']}`",
        f"- Max examples: `{format_max_examples(result['settings']['max_examples'])}`",
        "",
        "| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["benchmark"],
                    row["model"],
                    format_value(row["evaluated_examples"], 0),
                    format_value(row["accuracy"]),
                    format_value(row["accuracy_norm"]),
                    format_value(row["avg_choice_tokens"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    loaded_examples: dict[str, list[dict[str, Any]]] = {}
    for benchmark in args.benchmarks:
        examples = load_examples(benchmark)
        if args.max_examples and args.max_examples > 0:
            examples = examples[: args.max_examples]
        loaded_examples[benchmark] = examples

    started = time.time()
    models: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for model_name in args.models:
        model, metadata = load_model(
            model_name,
            device=device,
            v2_checkpoint=resolve_path(args.v2_checkpoint, Path.cwd()),
            v2_config=resolve_path(args.v2_config, Path.cwd()),
        )
        models[model_name] = metadata
        for benchmark in args.benchmarks:
            results.append(
                {
                    "model": model_name,
                    **evaluate_benchmark(
                        model,
                        tokenizer,
                        benchmark,
                        loaded_examples[benchmark],
                        device=device,
                        context_length=args.context_length,
                        disable_progress=args.no_progress,
                        model_name=model_name,
                    ),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "settings": {
            "device": device.type,
            "context_length": args.context_length,
            "max_examples": args.max_examples,
        },
        "models": models,
        "results": results,
        "elapsed_sec": time.time() - started,
    }

    output_json = resolve_path(args.output_json, Path.cwd())
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(resolve_path(args.output_md, Path.cwd()), result)
    print(json.dumps({"output_json": str(output_json), "output_md": args.output_md, "elapsed_sec": result["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
