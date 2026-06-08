"""Evaluate SoloGPT and GPT-2 checkpoints on tokenized shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sologpt_v1.model import SoloGPT_v1
from sologpt_v2.model import SoloGPT_v2


class HeldoutDataset(Dataset):
    def __init__(self, tensor_path: Path) -> None:
        data = torch.load(tensor_path, map_location="cpu")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"expected {tensor_path} to contain a torch.Tensor")
        if data.dim() != 2:
            raise ValueError(f"expected shard tensor with shape (rows, seq), got {tuple(data.shape)}")
        self.data = data.long()

    def __len__(self) -> int:
        return int(self.data.size(0))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate language-model perplexity on tokenized shards")
    parser.add_argument("--model", choices=["v1", "v2", "gpt2"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for v1/v2")
    parser.add_argument("--config", default=None, help="Config JSON path for v1/v2")
    parser.add_argument("--shard-dir", default="data/tokenized_chunks", help="Directory containing .pt shards")
    parser.add_argument("--shards", default=None, help="Inclusive shard range, for example 58:60")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def resolve_path(value: str | os.PathLike[str], base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' was requested but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' was requested but MPS is not available")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shard_index(path: Path) -> int | None:
    match = re.search(r"shard_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def parse_shard_range(spec: str) -> tuple[int, int]:
    if ":" not in spec:
        value = int(spec)
        return value, value
    start_text, end_text = spec.split(":", maxsplit=1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"invalid shard range {spec!r}: end must be >= start")
    return start, end


def discover_shards(shard_dir: Path) -> list[Path]:
    if not shard_dir.exists():
        raise FileNotFoundError(f"shard directory does not exist: {shard_dir}")
    shards = sorted(shard_dir.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"no .pt shards found in {shard_dir}")
    return shards


def select_shards(all_shards: list[Path], spec: str | None) -> list[Path]:
    if not spec:
        return all_shards
    start, end = parse_shard_range(spec)
    by_index = [
        path
        for path in all_shards
        if shard_index(path) is not None and start <= int(shard_index(path)) <= end
    ]
    if by_index:
        return by_index
    if start < len(all_shards):
        selected = all_shards[start : min(end + 1, len(all_shards))]
        if selected:
            return selected
    raise ValueError(f"shard range {spec!r} did not match files in shard directory")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return strip_module_prefix(load_file(str(path), device=str(device)))

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return strip_module_prefix(checkpoint)


def default_config_path(model_name: str) -> Path:
    if model_name == "v1":
        return REPO_ROOT / "sologpt_v1" / "config.json"
    if model_name == "v2":
        return REPO_ROOT / "sologpt_v2" / "config.json"
    raise ValueError("GPT-2 does not use a local config")


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.model == "gpt2":
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        return model, {
            "model_type": "gpt2",
            "context_length": int(getattr(model.config, "n_positions", 1024)),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }

    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required when --model {args.model}")

    config_path = resolve_path(args.config, Path.cwd()) if args.config else default_config_path(args.model)
    config = load_json(config_path)

    if args.model == "v1":
        model = SoloGPT_v1(config).to(device)
        model.device = device
    else:
        model = SoloGPT_v2(config).to(device)

    state = load_checkpoint_state(resolve_path(args.checkpoint, Path.cwd()), device)
    model.load_state_dict(state)
    return model, {
        "model_type": args.model,
        "config": str(config_path),
        "context_length": int(config.get("max_seq_len", config.get("seq_length", config.get("n_positions", 0)))),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def eval_shard(
    model: torch.nn.Module,
    tensor_path: Path,
    device: torch.device,
    *,
    batch_size: int,
    max_batches: int | None,
    disable_progress: bool,
) -> tuple[float, int]:
    loader = DataLoader(
        HeldoutDataset(tensor_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    sum_loss = 0.0
    sum_tokens = 0

    model.eval()
    with torch.no_grad():
        iterator = tqdm(loader, desc=tensor_path.name, dynamic_ncols=True, disable=disable_progress)
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch.to(device, non_blocking=True)
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = logits[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            tokens = int(targets.numel())
            sum_loss += float(loss.item())
            sum_tokens += tokens

    return sum_loss, sum_tokens


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    shard_dir = resolve_path(args.shard_dir, Path.cwd())
    shard_paths = select_shards(discover_shards(shard_dir), args.shards)
    model, model_info = load_model(args, device)

    started = time.time()
    total_loss = 0.0
    total_tokens = 0
    shard_results: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        shard_started = time.time()
        loss, tokens = eval_shard(
            model,
            shard_path,
            device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            disable_progress=args.no_progress,
        )
        total_loss += loss
        total_tokens += tokens
        shard_loss = loss / tokens if tokens else float("inf")
        shard_results.append(
            {
                "shard": shard_path.name,
                "loss": shard_loss,
                "perplexity": math.exp(shard_loss) if math.isfinite(shard_loss) else float("inf"),
                "tokens": tokens,
                "elapsed_sec": time.time() - shard_started,
            }
        )

    mean_loss = total_loss / total_tokens if total_tokens else float("inf")
    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "config": model_info.get("config"),
        "shard_dir": str(shard_dir),
        "shards": [path.name for path in shard_paths],
        "per_shard": shard_results,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf"),
        "tokens": total_tokens,
        "batch_size": args.batch_size,
        "max_batches_per_shard": args.max_batches,
        "parameter_count": model_info.get("parameter_count"),
        "context_length": model_info.get("context_length"),
        "device": device.type,
        "elapsed_sec": time.time() - started,
    }

    if args.output_json:
        output_path = resolve_path(args.output_json, Path.cwd())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
