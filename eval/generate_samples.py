"""Generate fixed qualitative samples for SoloLLM Phase 4."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval import load_checkpoint_state, load_json, resolve_device
from sologpt_v1.model import SoloGPT_v1
from sologpt_v2.model import SoloGPT_v2


DEFAULT_PROMPTS = HERE / "phase4_prompts.json"
DEFAULT_V1_CHECKPOINT = REPO_ROOT / "outputs" / "sologpt_v1" / "published" / "pytorch_model.safetensors"
DEFAULT_V1_CONFIG = REPO_ROOT / "outputs" / "sologpt_v1" / "published" / "config" / "soloGPT_v1_config.json"
DEFAULT_V2_CHECKPOINT = (
    REPO_ROOT / "outputs" / "sologpt_v2" / "final_3b_modern_small_from300m" / "final_model.pt"
)
DEFAULT_V2_CONFIG = REPO_ROOT / "sologpt_v2" / "config_modern_small.json"
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "outputs" / "sologpt_v2" / "final_3b_modern_small_from300m" / "phase4_generations.json"
)
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs" / "results" / "phase4_generations.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed qualitative Phase 4 samples")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--models", nargs="+", default=["v1", "v2", "gpt2"], choices=["v1", "v2", "gpt2"])
    parser.add_argument("--v1-checkpoint", default=str(DEFAULT_V1_CHECKPOINT))
    parser.add_argument("--v1-config", default=str(DEFAULT_V1_CONFIG))
    parser.add_argument("--v2-checkpoint", default=str(DEFAULT_V2_CHECKPOINT))
    parser.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list):
        raise TypeError("prompt file must contain a list")
    for prompt in prompts:
        if not isinstance(prompt, dict) or "id" not in prompt or "prompt" not in prompt:
            raise TypeError("each prompt must contain at least 'id' and 'prompt'")
    return prompts


def load_model(
    model_name: str,
    *,
    device: torch.device,
    v1_checkpoint: Path,
    v1_config: Path,
    v2_checkpoint: Path,
    v2_config: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        return model, {
            "model": model_name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "context_length": int(model.config.n_positions),
        }

    if model_name == "v1":
        config = load_json(v1_config)
        model = SoloGPT_v1(config).to(device)
        model.device = device
        checkpoint = v1_checkpoint
    else:
        config = load_json(v2_config)
        model = SoloGPT_v2(config).to(device)
        checkpoint = v2_checkpoint

    state = load_checkpoint_state(checkpoint, device)
    model.load_state_dict(state)
    context_length = int(config.get("max_seq_len", config.get("seq_length", config.get("n_positions", 512))))
    return model, {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "config": str(v1_config if model_name == "v1" else v2_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "context_length": context_length,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(values / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, sampled)

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str, int]:
    encoded = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = encoded
    eos_token_id = tokenizer.eos_token_id
    new_tokens = 0

    model.eval()
    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:]
        outputs = model(model_input)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        if next_token.item() == eos_token_id:
            break
        generated = torch.cat([generated, next_token], dim=1)
        new_tokens += 1

    return tokenizer.decode(generated[0], skip_special_tokens=True), new_tokens


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Phase 4 Generation Samples",
        "",
        "Fixed prompt suite using the GPT-2 tokenizer for all models.",
        "",
        "## Settings",
        "",
        f"- Seed: `{results['settings']['seed']}`",
        f"- Temperature: `{results['settings']['temperature']}`",
        f"- Top-k: `{results['settings']['top_k']}`",
        f"- Max new tokens: `{results['settings']['max_new_tokens']}`",
        "",
        "These are raw samples, not cherry-picked completions.",
        "",
    ]

    for sample in results["samples"]:
        lines.extend(
            [
                f"## {sample['prompt_id']} ({sample['category']})",
                "",
                f"Prompt: `{sample['prompt']}`",
                "",
            ]
        )
        for completion in sample["completions"]:
            text = completion["text"].strip()
            lines.extend(
                [
                    f"### {completion['model']}",
                    "",
                    "```text",
                    text,
                    "```",
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

    prompts = load_prompts(resolve_path(args.prompts))
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    settings = {
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "device": device.type,
    }

    models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for model_name in args.models:
        models[model_name] = load_model(
            model_name,
            device=device,
            v1_checkpoint=resolve_path(args.v1_checkpoint),
            v1_config=resolve_path(args.v1_config),
            v2_checkpoint=resolve_path(args.v2_checkpoint),
            v2_config=resolve_path(args.v2_config),
        )

    samples: list[dict[str, Any]] = []
    started = time.time()
    for prompt_index, prompt_row in enumerate(prompts):
        prompt_result: dict[str, Any] = {
            "prompt_id": prompt_row["id"],
            "category": prompt_row.get("category", ""),
            "prompt": prompt_row["prompt"],
            "completions": [],
        }
        for model_index, model_name in enumerate(args.models):
            model, metadata = models[model_name]
            seed_everything(args.seed + prompt_index * 100 + model_index)
            text, new_tokens = generate_one(
                model,
                tokenizer,
                prompt_row["prompt"],
                device=device,
                context_length=int(metadata["context_length"]),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            prompt_result["completions"].append(
                {
                    "model": model_name,
                    "text": text,
                    "new_tokens": new_tokens,
                    "parameter_count": metadata["parameter_count"],
                    "context_length": metadata["context_length"],
                }
            )
        samples.append(prompt_result)

    result = {
        "settings": settings,
        "models": {name: metadata for name, (_, metadata) in models.items()},
        "samples": samples,
        "elapsed_sec": time.time() - started,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(output_md, result)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "elapsed_sec": result["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
