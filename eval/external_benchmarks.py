"""Run lightweight external base-LM comparisons for SoloGPT v2 and GPT-2."""

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
from transformers import GPT2LMHeadModel, GPT2Tokenizer


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval import load_checkpoint_state, load_json, resolve_device, resolve_path
from sologpt_v2.model import SoloGPT_v2


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
    parser.add_argument("--models", nargs="+", default=["v2", "gpt2"], choices=["v2", "gpt2"])
    parser.add_argument("--benchmarks", nargs="+", default=["wikitext2", "lambada"], choices=["wikitext2", "lambada"])
    parser.add_argument("--v2-checkpoint", default=str(DEFAULT_V2_CHECKPOINT))
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--v2-label", default="v2", help="Display label for the v2-loaded model")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--context-length", type=int, default=512, help="Shared context window for all models")
    parser.add_argument("--max-wikitext-tokens", type=int, default=250_000)
    parser.add_argument("--max-lambada-examples", type=int, default=1_000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def display_name(model_name: str, *, v2_label: str) -> str:
    return v2_label if model_name == "v2" else model_name


def load_model(
    model_name: str,
    *,
    device: torch.device,
    v2_checkpoint: Path,
    v2_config: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        return model, {
            "model": "gpt2",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "native_context_length": int(model.config.n_positions),
        }

    config = load_json(v2_config)
    model = SoloGPT_v2(config).to(device)
    state = load_checkpoint_state(v2_checkpoint, device)
    model.load_state_dict(state)
    return model, {
        "model": "v2",
        "checkpoint": str(v2_checkpoint),
        "config": str(v2_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_context_length": int(config.get("max_seq_len", config.get("seq_length", 512))),
    }


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


def tokenize_corpus(tokenizer: GPT2Tokenizer, texts: list[str], max_tokens: int | None) -> list[int]:
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    token_ids: list[int] = []
    for text in texts:
        if eos_ids and token_ids:
            token_ids.extend(eos_ids)
        token_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if max_tokens is not None and max_tokens > 0 and len(token_ids) >= max_tokens:
            return token_ids[:max_tokens]
    return token_ids


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids)
    return outputs.logits if hasattr(outputs, "logits") else outputs


@torch.no_grad()
def score_token_ids(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    desc: str,
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
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf"),
        "tokens": total_tokens,
        "chunks": len(chunks),
    }


@torch.no_grad()
def score_lambada_last_token(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
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
    tokenizer: GPT2Tokenizer,
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
    tokenizer: GPT2Tokenizer,
    *,
    texts: list[str],
    device: torch.device,
    context_length: int,
    max_tokens: int,
    disable_progress: bool,
    model_name: str,
) -> dict[str, Any]:
    token_ids = tokenize_corpus(tokenizer, texts, max_tokens)
    result = score_token_ids(
        model,
        token_ids,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} wikitext2",
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
    tokenizer: GPT2Tokenizer,
    *,
    texts: list[str],
    device: torch.device,
    context_length: int,
    disable_progress: bool,
    model_name: str,
) -> dict[str, Any]:
    token_ids = tokenize_corpus(tokenizer, texts, None)
    ppl_result = score_token_ids(
        model,
        token_ids,
        device=device,
        context_length=context_length,
        disable_progress=disable_progress,
        desc=f"{model_name} lambada-ppl",
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
        "Compact external comparison using the GPT-2 tokenizer and a shared context window.",
        "",
        f"- Context length: `{result['settings']['context_length']}`",
        f"- Device: `{result['settings']['device']}`",
        f"- WikiText token cap: `{format_cap(result['settings']['max_wikitext_tokens'])}`",
        f"- LAMBADA example cap: `{format_cap(result['settings']['max_lambada_examples'])}`",
        "",
        "| Benchmark | Model | Tokens | PPL | Loss | Last-token acc | Last-word exact |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["benchmark"],
                    row["model"],
                    format_value(row.get("tokens"), 0),
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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

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
    for model_name in args.models:
        label = display_name(model_name, v2_label=args.v2_label)
        model, metadata = load_model(
            model_name,
            device=device,
            v2_checkpoint=resolve_path(args.v2_checkpoint, Path.cwd()),
            v2_config=resolve_path(args.v2_config, Path.cwd()),
        )
        models[label] = metadata

        if wikitext_texts is not None:
            results.append(
                {
                    "model": label,
                    "model_loader": model_name,
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
                    "model_loader": model_name,
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
