"""Train SoloGPT v2.

The CLI is intentionally usable for both tiny CPU smoke tests and real shard
training:

    python -m sologpt_v2.pretrain --dry-run --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .model import SoloGPT_v2
except ImportError:  # pragma: no cover - supports direct script execution
    from model import SoloGPT_v2


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CONFIG_PATH = HERE / "config.json"


class TokenizedShardDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor) -> None:
        if data_tensor.dim() != 2:
            raise ValueError(f"expected shard tensor with shape (rows, seq), got {tuple(data_tensor.shape)}")
        self.data = data_tensor.long()

    def __len__(self) -> int:
        return int(self.data.size(0))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SoloGPT v2")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON")
    parser.add_argument("--output-dir", default=None, help="Run output directory")
    parser.add_argument("--shard-dir", default=None, help="Directory containing tokenized .pt shards")
    parser.add_argument("--train-shards", default=None, help="Inclusive shard range, for example 0:54")
    parser.add_argument("--val-shards", default=None, help="Inclusive validation shard range, for example 55:57")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum input tokens to process")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override config grad_accum_steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config learning_rate")
    parser.add_argument("--max-eval-tokens", type=int, default=None, help="Cap validation tokens per eval")
    parser.add_argument("--eval-every-tokens", type=int, default=None, help="Override validation cadence")
    parser.add_argument("--tokens-per-checkpoint", type=int, default=None, help="Override checkpoint cadence")
    parser.add_argument("--log-every-opt-steps", type=int, default=None, help="Override train metric cadence")
    parser.add_argument("--dry-run", action="store_true", help="Use tiny default limits for smoke testing")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None, help="Training device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return parser.parse_args(argv)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | os.PathLike[str], base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_device(requested: str | None, explicit: bool = False) -> torch.device:
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        if explicit:
            raise RuntimeError("device 'cuda' was requested but CUDA is not available")
        logger.warning("Config requested CUDA, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        if explicit:
            raise RuntimeError("device 'mps' was requested but MPS is not available")
        logger.warning("Config requested MPS, but MPS is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_shard(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"expected {path} to contain a torch.Tensor, got {type(data).__name__}")
    return data.long()


def make_loader(
    data: torch.Tensor,
    config: dict[str, Any],
    *,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        TokenizedShardDataset(data),
        batch_size=int(config.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)) and device.type == "cuda",
        drop_last=shuffle,
        generator=generator,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def lr_for_tokens(config: dict[str, Any], tokens_total: int, target_tokens: int) -> float:
    learning_rate = float(config["learning_rate"])
    min_learning_rate = float(config.get("min_learning_rate", learning_rate))
    warmup_tokens = int(config.get("warmup_tokens", 0))

    if warmup_tokens > 0 and tokens_total < warmup_tokens:
        warmup_scale = max(tokens_total, 1) / warmup_tokens
        return learning_rate * warmup_scale

    decay_tokens = max(target_tokens - warmup_tokens, 1)
    progress = min(max((tokens_total - warmup_tokens) / decay_tokens, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (learning_rate - min_learning_rate)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def current_gpu_mem_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.memory_allocated(device) / 1e9)


def peak_gpu_mem_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1e9)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)

    if isinstance(checkpoint, dict):
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state") and device.type == "cuda":
            scaler.load_state_dict(checkpoint["scaler_state"])
        return dict(checkpoint.get("meta", {}))
    return {}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    config: dict[str, Any],
    meta: dict[str, Any],
    save_optimizer_state: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "meta": {
            **meta,
            "config": config,
        },
    }
    if save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["scaler_state"] = scaler.state_dict() if device.type == "cuda" else None
    atomic_torch_save(payload, path)


def evaluate(
    model: nn.Module,
    shard_paths: list[Path],
    config: dict[str, Any],
    device: torch.device,
    criterion: nn.Module,
    amp_ctx: contextlib.AbstractContextManager[Any],
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    max_eval_tokens = config.get("max_eval_tokens")
    max_eval_tokens = int(max_eval_tokens) if max_eval_tokens else None

    with torch.no_grad():
        for shard_path in shard_paths:
            data = load_shard(shard_path)
            loader = make_loader(data, config, shuffle=False, device=device)
            for batch in loader:
                input_ids = batch.to(device, non_blocking=True)
                with amp_ctx:
                    logits = model(input_ids)
                    logits = logits[:, :-1, :].contiguous()
                    targets = input_ids[:, 1:].contiguous()
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                tokens = int(targets.numel())
                total_loss += float(loss.item()) * tokens
                total_tokens += tokens

                if max_eval_tokens and total_tokens >= max_eval_tokens:
                    model.train()
                    return total_loss / total_tokens, total_tokens

    model.train()
    return (total_loss / total_tokens if total_tokens else float("inf")), total_tokens


def make_run_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.output_dir:
        run_dir = resolve_path(args.output_dir, Path.cwd())
    else:
        base = resolve_path(config.get("save_path", "outputs/sologpt_v2"))
        run_dir = base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def train(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config, Path.cwd())
    config = load_config(config_path)

    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    set_seed(seed)
    config["seed"] = seed

    device_requested = args.device if args.device is not None else config.get("device")
    device = resolve_device(device_requested, explicit=args.device is not None)
    config["device"] = device.type

    if args.shard_dir:
        config["shard_dir"] = args.shard_dir
    shard_dir = resolve_path(config.get("shard_dir", "data/tokenized_chunks"), Path.cwd())

    if args.train_shards is not None:
        config["train_shards"] = args.train_shards
    if args.val_shards is not None:
        config["val_shards"] = args.val_shards
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        config["grad_accum_steps"] = args.grad_accum_steps
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.max_eval_tokens is not None:
        config["max_eval_tokens"] = args.max_eval_tokens
    if args.eval_every_tokens is not None:
        config["eval_every_tokens"] = args.eval_every_tokens
    if args.tokens_per_checkpoint is not None:
        config["tokens_per_checkpoint"] = args.tokens_per_checkpoint
    if args.log_every_opt_steps is not None:
        config["log_every_opt_steps"] = args.log_every_opt_steps

    target_tokens = int(
        args.max_tokens
        if args.max_tokens is not None
        else config.get("total_tokens")
        or (config.get("training", {}) or {}).get("total_tokens")
        or 300_000_000
    )
    if args.dry_run and args.max_steps is None and args.max_tokens is None:
        max_steps = 3
        target_tokens = min(target_tokens, int(config.get("batch_size", 1)) * int(config.get("seq_length", 1)) * 8)
    else:
        max_steps = args.max_steps

    run_dir = make_run_dir(args, config)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "metrics_summary.json"
    final_model_path = run_dir / "final_model.pt"
    latest_ckpt_path = run_dir / "checkpoints" / "latest.pt"

    all_shards = discover_shards(shard_dir)
    train_shards = select_shards(all_shards, config.get("train_shards"))
    val_shards = select_shards(all_shards, config.get("val_shards")) if config.get("val_shards") else []
    if not train_shards:
        raise FileNotFoundError("no training shards selected")

    config_resolved = {
        **config,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "shard_dir": str(shard_dir),
        "train_shard_files": [str(path) for path in train_shards],
        "val_shard_files": [str(path) for path in val_shards],
        "target_tokens": target_tokens,
        "max_steps": max_steps,
        "dry_run": bool(args.dry_run),
    }
    write_json(run_dir / "config_resolved.json", config_resolved)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    model = SoloGPT_v2(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext()

    tokens_total = 0
    optimizer_steps = 0
    micro_step = 0
    resume_meta: dict[str, Any] = {}
    if args.resume:
        resume_path = resolve_path(args.resume, Path.cwd())
        resume_meta = load_checkpoint(resume_path, model, optimizer, scaler, device)
        tokens_total = int(resume_meta.get("tokens_total", 0))
        optimizer_steps = int(resume_meta.get("optimizer_steps", resume_meta.get("global_steps", 0)))
    tokens_start = tokens_total

    model_summary = model.summary()
    run_start = {
        "type": "run_start",
        "ts": time.time(),
        "run_dir": str(run_dir),
        "device": device.type,
        "seed": seed,
        "config_path": str(config_path),
        "resume": str(args.resume) if args.resume else None,
        "tokens_start": tokens_start,
        "model": model_summary,
    }
    write_json(run_dir / "run_meta.json", run_start)
    write_jsonl(metrics_path, run_start)

    logger.info("Training SoloGPT v2 on %s", device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Train shards: %s", ", ".join(path.name for path in train_shards))
    if val_shards:
        logger.info("Validation shards: %s", ", ".join(path.name for path in val_shards))
    logger.info("Model parameters: %s", f"{model_summary['parameter_count']:,}")

    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    clip_grad_norm = config.get("clip_grad_norm", 1.0)
    clip_grad_norm = float(clip_grad_norm) if clip_grad_norm else 0.0
    log_every = max(int(config.get("log_every_opt_steps", 50)), 1)
    tokens_per_checkpoint = int(config.get("tokens_per_checkpoint", target_tokens))
    eval_every_tokens = int(config.get("eval_every_tokens", 0))
    save_optimizer_state = bool(config.get("save_optimizer_state", True))

    start_time = time.time()
    last_train_loss: float | None = None
    best_val_loss: float | None = None
    tokens_since_checkpoint = 0
    tokens_since_eval = 0
    interrupted = False
    status = "complete"
    stop_requested = False
    if max_steps is not None and optimizer_steps >= max_steps:
        stop_requested = True
    if tokens_total >= target_tokens:
        stop_requested = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        total=max(target_tokens - tokens_total, 0),
        desc="tokens",
        unit="tok",
        dynamic_ncols=True,
        disable=args.no_progress or args.dry_run,
    )

    try:
        for pass_idx in range(10**9):
            if stop_requested:
                break
            for shard_position, shard_path in enumerate(train_shards):
                if stop_requested:
                    break

                data = load_shard(shard_path)
                if data.size(1) > int(config.get("max_seq_len", config.get("seq_length", data.size(1)))):
                    raise ValueError(
                        f"{shard_path.name} sequence length {data.size(1)} exceeds max_seq_len"
                    )

                write_jsonl(
                    metrics_path,
                    {
                        "type": "shard_start",
                        "ts": time.time(),
                        "pass_idx": pass_idx,
                        "shard_position": shard_position,
                        "shard": shard_path.name,
                        "tokens_total": tokens_total,
                        "optimizer_steps": optimizer_steps,
                    },
                )

                loader = make_loader(data, config, shuffle=True, device=device, generator=generator)
                for batch_idx, batch in enumerate(loader):
                    input_ids = batch.to(device, non_blocking=True)
                    tokens_this_batch = int(input_ids.numel())
                    lr = lr_for_tokens(config, tokens_total, target_tokens)
                    set_optimizer_lr(optimizer, lr)

                    with amp_ctx:
                        logits = model(input_ids)
                        logits = logits[:, :-1, :].contiguous()
                        targets = input_ids[:, 1:].contiguous()
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                        scaled_loss = loss / grad_accum_steps

                    scaler.scale(scaled_loss).backward()
                    micro_step += 1
                    tokens_total += tokens_this_batch
                    tokens_since_checkpoint += tokens_this_batch
                    tokens_since_eval += tokens_this_batch
                    last_train_loss = float(loss.item())

                    did_step = False
                    grad_norm: float | None = None
                    if micro_step >= grad_accum_steps:
                        if clip_grad_norm > 0:
                            if use_cuda:
                                scaler.unscale_(optimizer)
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm).item()
                            )

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        micro_step = 0
                        did_step = True

                    elapsed = time.time() - start_time
                    progress.update(tokens_this_batch)
                    progress.set_postfix(loss=f"{last_train_loss:.4f}", step=optimizer_steps)

                    if did_step and (optimizer_steps % log_every == 0 or args.dry_run):
                        write_jsonl(
                            metrics_path,
                            {
                                "type": "train",
                                "ts": time.time(),
                                "elapsed_sec": elapsed,
                                "tokens_total": tokens_total,
                                "tokens_processed_this_run": tokens_total - tokens_start,
                                "tokens_this_batch": tokens_this_batch,
                                "tokens_per_sec": (tokens_total - tokens_start) / max(elapsed, 1e-9),
                                "optimizer_steps": optimizer_steps,
                                "micro_step": micro_step,
                                "train_loss": last_train_loss,
                                "lr": lr,
                                "grad_norm": grad_norm,
                                "pass_idx": pass_idx,
                                "shard_position": shard_position,
                                "shard": shard_path.name,
                                "batch_idx": batch_idx,
                                "gpu_mem_gb": current_gpu_mem_gb(device),
                                "gpu_peak_mem_gb": peak_gpu_mem_gb(device),
                            },
                        )

                    if val_shards and eval_every_tokens > 0 and tokens_since_eval >= eval_every_tokens:
                        val_loss, val_tokens = evaluate(model, val_shards, config, device, criterion, amp_ctx)
                        best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
                        write_jsonl(
                            metrics_path,
                            {
                                "type": "validation",
                                "ts": time.time(),
                                "tokens_total": tokens_total,
                                "optimizer_steps": optimizer_steps,
                                "val_loss": val_loss,
                                "val_ppl": perplexity(val_loss),
                                "val_tokens": val_tokens,
                            },
                        )
                        tokens_since_eval = 0

                    if tokens_since_checkpoint >= tokens_per_checkpoint:
                        meta = {
                            "tokens_total": tokens_total,
                            "optimizer_steps": optimizer_steps,
                            "micro_step": micro_step,
                            "pass_idx": pass_idx,
                            "shard_position": shard_position,
                            "shard": shard_path.name,
                            "batch_idx": batch_idx,
                        }
                        numbered = run_dir / "checkpoints" / f"ckpt_{tokens_total // 1_000_000}M.pt"
                        save_checkpoint(
                            numbered,
                            model,
                            optimizer,
                            scaler,
                            device=device,
                            config=config_resolved,
                            meta=meta,
                            save_optimizer_state=save_optimizer_state,
                        )
                        save_checkpoint(
                            latest_ckpt_path,
                            model,
                            optimizer,
                            scaler,
                            device=device,
                            config=config_resolved,
                            meta=meta,
                            save_optimizer_state=save_optimizer_state,
                        )
                        write_jsonl(
                            metrics_path,
                            {
                                "type": "checkpoint",
                                "ts": time.time(),
                                "checkpoint_path": str(numbered),
                                "latest_path": str(latest_ckpt_path),
                                "tokens_total": tokens_total,
                                "optimizer_steps": optimizer_steps,
                            },
                        )
                        tokens_since_checkpoint = 0

                    if max_steps is not None and optimizer_steps >= max_steps:
                        stop_requested = True
                    if tokens_total >= target_tokens:
                        stop_requested = True
                    if stop_requested:
                        break

                write_jsonl(
                    metrics_path,
                    {
                        "type": "shard_end",
                        "ts": time.time(),
                        "pass_idx": pass_idx,
                        "shard_position": shard_position,
                        "shard": shard_path.name,
                        "tokens_total": tokens_total,
                        "optimizer_steps": optimizer_steps,
                    },
                )

    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        logger.warning("Interrupted. Saving partial progress.")
    except Exception as exc:
        status = "failed"
        write_jsonl(metrics_path, {"type": "error", "ts": time.time(), "error": repr(exc)})
        raise
    finally:
        progress.close()

    if micro_step:
        if clip_grad_norm > 0:
            if use_cuda:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        micro_step = 0

    should_run_final_eval = (
        bool(val_shards)
        and status != "failed"
        and (best_val_loss is None or (stop_requested and tokens_since_eval > 0))
    )
    if should_run_final_eval:
        val_loss, val_tokens = evaluate(model, val_shards, config, device, criterion, amp_ctx)
        best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
        write_jsonl(
            metrics_path,
            {
                "type": "validation",
                "final": True,
                "ts": time.time(),
                "tokens_total": tokens_total,
                "optimizer_steps": optimizer_steps,
                "val_loss": val_loss,
                "val_ppl": perplexity(val_loss),
                "val_tokens": val_tokens,
            },
        )

    final_meta = {
        "tokens_total": tokens_total,
        "optimizer_steps": optimizer_steps,
        "micro_step": micro_step,
        "interrupted": interrupted,
        "status": status,
    }
    save_checkpoint(
        latest_ckpt_path,
        model,
        optimizer,
        scaler,
        device=device,
        config=config_resolved,
        meta=final_meta,
        save_optimizer_state=save_optimizer_state,
    )
    atomic_torch_save(model.state_dict(), final_model_path)

    elapsed = time.time() - start_time
    tokens_processed_this_run = tokens_total - tokens_start
    summary = {
        "status": status,
        "interrupted": interrupted,
        "tokens_total": tokens_total,
        "tokens_start": tokens_start,
        "tokens_processed_this_run": tokens_processed_this_run,
        "optimizer_steps": optimizer_steps,
        "final_train_loss": last_train_loss,
        "best_val_loss": best_val_loss,
        "best_val_ppl": perplexity(best_val_loss),
        "total_time_sec": elapsed,
        "tokens_per_sec_avg": tokens_processed_this_run / max(elapsed, 1e-9),
        "parameter_count": model_summary["parameter_count"],
        "checkpoint_path": str(latest_ckpt_path),
        "final_model_path": str(final_model_path),
        "metrics_jsonl": str(metrics_path),
        "config": config_resolved,
        "resume_meta": resume_meta,
        "gpu_peak_mem_gb": peak_gpu_mem_gb(device),
    }
    write_json(summary_path, summary)
    write_jsonl(
        metrics_path,
        {
            "type": "run_end",
            "ts": time.time(),
            "status": status,
            "interrupted": interrupted,
            "tokens_total": tokens_total,
            "tokens_processed_this_run": tokens_processed_this_run,
            "optimizer_steps": optimizer_steps,
            "elapsed_sec": elapsed,
            "final_train_loss": last_train_loss,
            "best_val_loss": best_val_loss,
        },
    )

    logger.info("Summary saved: %s", summary_path)
    logger.info("Final model saved: %s", final_model_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    return train(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
