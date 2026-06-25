"""Run lightweight external base-LM comparisons for SoloLLM and HF models."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval import resolve_device, resolve_path
from eval.model_registry import load_eval_model, model_logits


DEFAULT_V2_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "sologpt_v2"
    / "stretch_5p85b_modern_small_from3b"
    / "checkpoints"
    / "latest.pt"
)
DEFAULT_V2_CONFIG = REPO_ROOT / "sologpt_v2" / "config_modern_small.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact external base-LM benchmarks")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v2", "gpt2"],
        help="Model specs: v2, gpt2, distilgpt2, hf:org/model, or label=hf:org/model",
    )
    parser.add_argument("--benchmarks", nargs="+", default=["wikitext2", "lambada"], choices=["wikitext2", "lambada"])
    parser.add_argument("--v2-checkpoint", default=str(DEFAULT_V2_CHECKPOINT))
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--v2-label", default="v2", help="Display label for the v2-loaded model")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--context-length", type=int, default=512, help="Shared context window for all models")
    parser.add_argument("--max-wikitext-tokens", type=int, default=0, help="0 means all available WikiText-2 tokens")
    parser.add_argument("--max-lambada-examples", type=int, default=1_000)
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow HF custom model code")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def load_wikitext2_texts() -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [str(row["text"]) for row in dataset if str(row["text"]).strip()]


def load_lambada_texts() -> list[str]:
    from datasets import load_dataset

    errors: list[str] = []
    for name, config in [("EleutherAI/lambada_openai", None), ("lambada", None)]:
        try:
            dataset = load_dataset(name, config, split="test") if config else load_dataset(name, split="test")
            return [str(row["text"]) for row in dataset if str(row["text"]).strip()]
        except Exception as exc:  # pragma: no cover - depends on external dataset availability
            errors.append(f"{name}: {exc}")
    raise RuntimeError("could not load a LAMBADA dataset: " + " | ".join(errors))


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def tokenize_corpus(tokenizer: Any, texts: list[str], max_tokens: int | None) -> tuple[list[int], int, bool]:
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    token_ids: list[int] = []
    byte_count = 0
    byte_count_is_exact = True
    for text in texts:
        if eos_ids and token_ids:
            token_ids.extend(eos_ids)
        text_ids = tokenizer(text, add_special_tokens=False).input_ids
        token_ids.extend(text_ids)
        byte_count += utf8_len(text)
        if max_tokens is not None and max_tokens > 0 and len(token_ids) >= max_tokens:
            byte_count_is_exact = len(token_ids) == max_tokens
            return token_ids[:max_tokens], byte_count, byte_count_is_exact
    return token_ids, byte_count, byte_count_is_exact


@torch.no_grad()
def score_token_ids(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    desc: str,
    byte_count: int | None = None,
    byte_count_is_exact: bool | None = None,
) -> dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    chunks = [
        token_ids[start : start + context_length]
        for start in range(0, max(0, len(token_ids) - 1), context_length)
        if len(token_ids[start : start + context_length]) >= 2
    ]

    model.eval()
    for chunk in tqdm(chunks, desc=desc, dynamic_ncols=True, disable=disable_progress):
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        logits = model_logits(model, input_ids)
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())

    mean_loss = total_loss / total_tokens if total_tokens else float("inf")
    result = {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf"),
        "tokens": total_tokens,
        "chunks": len(chunks),
    }
    if byte_count:
        result.update(
            {
                "bytes": byte_count,
                "bits_per_byte": total_loss / math.log(2) / byte_count,
                "nats_per_byte": total_loss / byte_count,
                "byte_count_is_exact": byte_count_is_exact,
            }
        )
    return result


@torch.no_grad()
def score_lambada_last_token(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    desc: str,
) -> dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_examples = 0
    correct = 0

    model.eval()
    for text in tqdm(texts, desc=desc, dynamic_ncols=True, disable=disable_progress):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) < 2:
            continue
        window = token_ids[-context_length:]
        prefix = window[:-1]
        target = window[-1]
        if not prefix:
            continue
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        target_tensor = torch.tensor([target], dtype=torch.long, device=device)
        logits = model_logits(model, input_ids)[:, -1, :]
        total_loss += float(criterion(logits, target_tensor).item())
        correct += int(torch.argmax(logits, dim=-1).item() == target)
        total_examples += 1

    mean_loss = total_loss / total_examples if total_examples else float("inf")
    return {
        "last_token_loss": mean_loss,
        "last_token_perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf"),
        "last_token_accuracy": correct / total_examples if total_examples else None,
        "last_token_correct": correct,
        "last_token_examples": total_examples,
    }


def split_final_word(text: str) -> tuple[str, str] | None:
    match = re.search(r"(\s+\S+)\s*$", text)
    if not match:
        return None
    prefix = text[: match.start(1)]
    target_piece = match.group(1).rstrip()
    if not prefix.strip() or not target_piece.strip():
        return None
    return prefix, target_piece


@torch.no_grad()
def score_lambada_last_word(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    desc: str,
) -> dict[str, Any]:
    total_examples = 0
    correct = 0

    model.eval()
    for text in tqdm(texts, desc=desc, dynamic_ncols=True, disable=disable_progress):
        pieces = split_final_word(text)
        if pieces is None:
            continue
        prefix_text, target_text = pieces
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        if not prefix_ids or not target_ids:
            continue

        generated_ids: list[int] = []
        current_ids = prefix_ids[:]
        for _ in target_ids:
            window = current_ids[-context_length:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model_logits(model, input_ids)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_id)
            current_ids.append(next_id)

        total_examples += 1
        correct += int(generated_ids == target_ids)

    return {
        "last_word_greedy_exact_accuracy": correct / total_examples if total_examples else None,
        "last_word_greedy_exact_correct": correct,
        "last_word_greedy_exact_examples": total_examples,
    }


def run_wikitext2(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    texts: list[str],
    device: torch.device,
    context_length: int,
    max_tokens: int,
    disable_progress: bool,
    model_name: str,
) -> dict[str, Any]:
    token_ids, byte_count, byte_count_is_exact = tokenize_corpus(tokenizer, texts, max_tokens)
    result = score_token_ids(
        model,
        token_ids,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} wikitext2",
        byte_count=byte_count,
        byte_count_is_exact=byte_count_is_exact,
    )
    result.update(
        {
            "benchmark": "wikitext2",
            "dataset": "wikitext/wikitext-2-raw-v1/test",
            "max_tokens": max_tokens,
            "context_length": context_length,
        }
    )
    return result


def run_lambada(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    texts: list[str],
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    model_name: str,
) -> dict[str, Any]:
    token_ids, byte_count, byte_count_is_exact = tokenize_corpus(tokenizer, texts, None)
    ppl_result = score_token_ids(
        model,
        token_ids,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} lambada-ppl",
        byte_count=byte_count,
        byte_count_is_exact=byte_count_is_exact,
    )
    last_token = score_lambada_last_token(
        model,
        tokenizer,
        texts,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} lambada-last-token",
    )
    last_word = score_lambada_last_word(
        model,
        tokenizer,
        texts,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} lambada-last-word",
    )
    return {
        "benchmark": "lambada",
        "dataset": "EleutherAI/lambada_openai/test",
        "examples": len(texts),
        "context_length": context_length,
        **ppl_result,
        **last_token,
        **last_word,
    }


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def format_cap(value: Any) -> str:
    if value is None:
        return "full"
    if isinstance(value, int) and value <= 0:
        return "full"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# External Base-LM Benchmark Results",
        "",
        "Compact external comparison using each model's native tokenizer and a shared context window.",
        "",
        "Raw token perplexity is tokenizer-local. Use bits-per-byte and downstream accuracies for cross-tokenizer comparisons.",
        "",
        f"- Context length: `{result['settings']['context_length']}`",
        f"- Device: `{result['settings']['device']}`",
        f"- WikiText token cap: `{format_cap(result['settings']['max_wikitext_tokens'])}`",
        f"- LAMBADA example cap: `{format_cap(result['settings']['max_lambada_examples'])}`",
        "",
        "| Benchmark | Model | Tokens | Bytes | BPB | PPL | Loss | Last-token acc | Last-word exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["benchmark"],
                    row["model"],
                    format_value(row.get("tokens"), 0),
                    format_value(row.get("bytes"), 0),
                    format_value(row.get("bits_per_byte")),
                    format_value(row.get("perplexity")),
                    format_value(row.get("loss")),
                    format_value(row.get("last_token_accuracy")),
                    format_value(row.get("last_word_greedy_exact_accuracy")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)

    wikitext_texts: list[str] | None = None
    lambada_texts: list[str] | None = None
    if "wikitext2" in args.benchmarks:
        wikitext_texts = load_wikitext2_texts()
    if "lambada" in args.benchmarks:
        lambada_texts = load_lambada_texts()
        if args.max_lambada_examples and args.max_lambada_examples > 0:
            lambada_texts = lambada_texts[: args.max_lambada_examples]

    started = time.time()
    models: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for model_spec in args.models:
        loaded_model = load_eval_model(
            model_spec,
            device=device,
            v2_checkpoint=resolve_path(args.v2_checkpoint, Path.cwd()),
            v2_config=resolve_path(args.v2_config, Path.cwd()),
            v2_label=args.v2_label,
            trust_remote_code=args.trust_remote_code,
        )
        label = loaded_model.label
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        metadata = loaded_model.metadata
        models[label] = metadata

        if wikitext_texts is not None:
            results.append(
                {
                    "model": label,
                    "model_loader": loaded_model.loader,
                    **run_wikitext2(
                        model,
                        tokenizer,
                        texts=wikitext_texts,
                        device=device,
                        context_length=args.context_length,
                        max_tokens=args.max_wikitext_tokens,
                        disable_progress=args.no_progress,
                        model_name=label,
                    ),
                }
            )
        if lambada_texts is not None:
            results.append(
                {
                    "model": label,
                    "model_loader": loaded_model.loader,
                    **run_lambada(
                        model,
                        tokenizer,
                        texts=lambada_texts,
                        device=device,
                        context_length=args.context_length,
                        disable_progress=args.no_progress,
                        model_name=label,
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
            "max_wikitext_tokens": args.max_wikitext_tokens,
            "max_lambada_examples": args.max_lambada_examples,
            "tokenizer_policy": "native_per_model",
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
