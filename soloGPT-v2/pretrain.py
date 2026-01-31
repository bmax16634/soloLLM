# pretrain.py
import os
import json
import logging
import time
import contextlib
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------- Import (works for both "python pretrain.py" and "-m package") ----------
try:
    from .model import SoloGPT_v2  # package style
except ImportError:
    from model import SoloGPT_v2   # script style


# ---------- Config & Setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

output_dir = config.get("save_path", os.path.join(HERE, "outputs"))
os.makedirs(output_dir, exist_ok=True)
model_basename = "sologpt_pretrained"
final_model_path = os.path.join(output_dir, f"{model_basename}.pth")

# Your shard dir (explicit)
shard_dir = "/home/bmx/_projects/soloLLM/data/tokenized_chunks"
val_path = config.get("val_path")

device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
batch_size = int(config["batch_size"])
learning_rate = float(config["learning_rate"])
weight_decay = float(config["weight_decay"])

target_tokens = (
    config.get("total_tokens")
    or (config.get("training", {}) or {}).get("total_tokens")
    or 300_000_000
)
target_tokens = int(target_tokens)

tokens_per_checkpoint = int(config.get("tokens_per_checkpoint", 300_000_000))
num_workers = int(config.get("num_workers", 0))
pin_memory = bool(config.get("pin_memory", False))
grad_accum_steps = int(config.get("grad_accum_steps", 1))

# Logging controls (NEW)
log_every_opt_steps = int(config.get("log_every_opt_steps", 50))          # log to file every N optimizer steps
eval_every_checkpoints = int(config.get("eval_every_checkpoints", 1))     # run val every N checkpoints
save_optimizer_state = bool(config.get("save_optimizer_state", True))     # save optimizer/scaler in ckpt
clip_grad_norm = config.get("clip_grad_norm", 1.0)                        # set None/0 to disable

logger.info(f"Using device: {device}")
logger.info(f"Shard dir: {shard_dir}")
logger.info(f"Config: {CONFIG_PATH}")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_meta_path = os.path.join(output_dir, f"run_meta_{run_id}.json")
metrics_path = os.path.join(output_dir, f"metrics_{run_id}.jsonl")  # NEW: JSONL (streaming)
summary_path = os.path.join(output_dir, f"training_summary_{run_id}.json")
loss_curve_path = os.path.join(output_dir, f"loss_curve_{run_id}.json")  # optional end-of-run

# ---------- AMP (updated API to remove deprecation warnings) ----------
use_cuda = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
amp_ctx = (torch.amp.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext())


# ---------- Dataset ----------
class TokenizedShardDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader_from_shard(shard_path: str) -> DataLoader:
    logger.info(f"🔹 Loading shard: {shard_path}")
    data = torch.load(shard_path, map_location="cpu")
    logger.info(f"   shard tensor shape={tuple(data.shape)} dtype={data.dtype}")  # NEW

    dataset = TokenizedShardDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device, non_blocking=True)
            with amp_ctx:
                logits = model(input_ids)
                logits = logits[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            tokens = targets.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def write_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def current_gpu_mem_gb() -> float | None:
    if not use_cuda:
        return None
    return float(torch.cuda.memory_allocated() / 1e9)


def peak_gpu_mem_gb() -> float | None:
    if not use_cuda:
        return None
    return float(torch.cuda.max_memory_allocated() / 1e9)


# ---------- Model Setup ----------
model = SoloGPT_v2(config).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# ---------- Optional Validation ----------
val_loader = None
if val_path and os.path.isfile(val_path):
    logger.info(f"🔸 Loading validation shard: {val_path}")
    val_data = torch.load(val_path, map_location="cpu")
    logger.info(f"   val tensor shape={tuple(val_data.shape)} dtype={val_data.dtype}")  # NEW
    val_loader = DataLoader(
        TokenizedShardDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

# ---------- Training ----------
logger.info("🚀 Starting training across shards...")

shard_paths = sorted(glob(os.path.join(shard_dir, "*.pt")))
if not shard_paths:
    raise FileNotFoundError(f"No shards found in {shard_dir} (expected *.pt)")

# Save run metadata up front (NEW)
run_meta = {
    "run_id": run_id,
    "started_at": datetime.now().isoformat(),
    "device": str(device),
    "use_cuda": use_cuda,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
    "num_gpus": torch.cuda.device_count() if use_cuda else 0,
    "output_dir": output_dir,
    "config_path": CONFIG_PATH,
    "shard_dir": shard_dir,
    "num_shards": len(shard_paths),
    "metrics_jsonl": metrics_path,
    "summary_json": summary_path,
    "save_optimizer_state": save_optimizer_state,
    "grad_accum_steps": grad_accum_steps,
    "clip_grad_norm": clip_grad_norm,
    "log_every_opt_steps": log_every_opt_steps,
    "tokens_per_checkpoint": tokens_per_checkpoint,
    "target_tokens": target_tokens,
    "config": config,
}
with open(run_meta_path, "w", encoding="utf-8") as f:
    json.dump(run_meta, f, indent=2)
logger.info(f"📝 Run metadata saved: {run_meta_path}")
logger.info(f"📝 Metrics JSONL: {metrics_path}")

total_tokens_processed = 0
tokens_since_ckpt = 0
global_steps = 0          # optimizer steps (not micro-steps)
micro_step = 0            # micro-batches accumulated
checkpoint_count = 0

# We'll keep a small in-memory list for easy plotting later, but JSONL is the main log (NEW)
loss_log = []
start_time = time.time()
interrupted = False

if use_cuda:
    torch.cuda.reset_peak_memory_stats()

progress_total = tqdm(total=target_tokens, desc="Total Training", unit="tok", dynamic_ncols=True)
progress_ckpt = tqdm(total=tokens_per_checkpoint, desc="Next Checkpoint", unit="tok", dynamic_ncols=True, leave=False)

try:
    for shard_idx, shard_path in enumerate(shard_paths):
        if total_tokens_processed >= target_tokens:
            break

        dataloader = get_dataloader_from_shard(shard_path)
        model.train()

        shard_name = os.path.basename(shard_path)

        # NEW: log shard start
        write_jsonl(metrics_path, {
            "type": "shard_start",
            "ts": time.time(),
            "shard_idx": shard_idx,
            "shard": shard_name,
            "tokens_total": total_tokens_processed,
            "global_steps": global_steps,
        })

        for batch_idx, batch in enumerate(dataloader):
            if total_tokens_processed >= target_tokens:
                break

            input_ids = batch.to(device, non_blocking=True)
            tokens_this_batch = input_ids.numel()

            with amp_ctx:
                logits = model(input_ids)
                logits = logits[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            did_opt_step = False
            grad_norm = None

            if micro_step >= grad_accum_steps:
                # optional grad norm + clipping (NEW)
                if clip_grad_norm and float(clip_grad_norm) > 0:
                    if use_cuda:
                        scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm)).item())

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                micro_step = 0
                global_steps += 1
                did_opt_step = True

            # book-keeping
            total_tokens_processed += tokens_this_batch
            tokens_since_ckpt += tokens_this_batch

            elapsed = time.time() - start_time
            tok_s = total_tokens_processed / max(elapsed, 1e-9)
            it_s = global_steps / max(elapsed, 1e-9)
            lr = float(optimizer.param_groups[0]["lr"])

            progress_total.update(tokens_this_batch)
            progress_ckpt.update(tokens_this_batch)
            progress_total.set_postfix(
                loss=f"{(loss.item() * grad_accum_steps):.4f}",
                speed=f"{tok_s:,.0f} tok/s",
                iters=f"{it_s:.2f} it/s",
            )

            # In-memory curve point (small)
            loss_log.append({
                "tokens": int(total_tokens_processed),
                "train_loss": float(loss.item() * grad_accum_steps),
            })

            # NEW: write JSONL metrics periodically (optimizer-step based)
            if did_opt_step and (global_steps % log_every_opt_steps == 0):
                write_jsonl(metrics_path, {
                    "type": "train",
                    "ts": time.time(),
                    "elapsed_s": float(elapsed),
                    "tokens_total": int(total_tokens_processed),
                    "tokens_this_batch": int(tokens_this_batch),
                    "tokens_per_sec": float(tok_s),
                    "optimizer_steps": int(global_steps),
                    "micro_step": int(micro_step),
                    "train_loss": float(loss.item() * grad_accum_steps),
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "shard_idx": int(shard_idx),
                    "shard": shard_name,
                    "batch_idx": int(batch_idx),
                    "gpu_mem_gb": current_gpu_mem_gb(),
                    "gpu_peak_mem_gb": peak_gpu_mem_gb(),
                })

            # ---- checkpoint WITHOUT skipping data ----
            while tokens_since_ckpt >= tokens_per_checkpoint:
                checkpoint_count += 1
                ckpt_name = os.path.join(
                    output_dir,
                    f"{model_basename}_{total_tokens_processed // 1_000_000}M.pth"
                )

                to_save = model.module if isinstance(model, nn.DataParallel) else model

                # NEW: optionally save optimizer/scaler/counters for resuming
                if save_optimizer_state:
                    ckpt = {
                        "model_state": to_save.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if use_cuda else None,
                        "meta": {
                            "run_id": run_id,
                            "tokens_total": int(total_tokens_processed),
                            "tokens_since_ckpt": int(tokens_since_ckpt),
                            "global_steps": int(global_steps),
                            "micro_step": int(micro_step),
                            "shard_idx": int(shard_idx),
                            "shard": shard_name,
                            "batch_idx": int(batch_idx),
                            "config": config,
                        }
                    }
                    torch.save(ckpt, ckpt_name)
                else:
                    torch.save(to_save.state_dict(), ckpt_name)

                logger.info(f"✅ Checkpoint saved at {total_tokens_processed:,} tokens → {ckpt_name}")

                # NEW: checkpoint event log
                write_jsonl(metrics_path, {
                    "type": "checkpoint",
                    "ts": time.time(),
                    "checkpoint_count": int(checkpoint_count),
                    "checkpoint_path": ckpt_name,
                    "tokens_total": int(total_tokens_processed),
                    "optimizer_steps": int(global_steps),
                    "shard_idx": int(shard_idx),
                    "shard": shard_name,
                    "gpu_peak_mem_gb": peak_gpu_mem_gb(),
                })

                # Validation every N checkpoints (NEW)
                if val_loader and (checkpoint_count % eval_every_checkpoints == 0):
                    val_loss = evaluate(model, val_loader, criterion)
                    logger.info(f"📉 Validation loss: {val_loss:.4f}")

                    write_jsonl(metrics_path, {
                        "type": "val",
                        "ts": time.time(),
                        "tokens_total": int(total_tokens_processed),
                        "optimizer_steps": int(global_steps),
                        "val_loss": float(val_loss),
                        "checkpoint_count": int(checkpoint_count),
                    })

                    # also attach to last in-memory point for convenience
                    if loss_log:
                        loss_log[-1]["val_loss"] = float(val_loss)
                    model.train()

                tokens_since_ckpt -= tokens_per_checkpoint
                progress_ckpt.reset()
                progress_ckpt.update(tokens_since_ckpt)

        # NEW: log shard end
        write_jsonl(metrics_path, {
            "type": "shard_end",
            "ts": time.time(),
            "shard_idx": shard_idx,
            "shard": shard_name,
            "tokens_total": int(total_tokens_processed),
            "optimizer_steps": int(global_steps),
        })

except KeyboardInterrupt:
    interrupted = True
    logger.warning("🛑 Interrupted by user. Saving partial progress...")

finally:
    # If we exit with partially accumulated grads, optionally flush one last optimizer step (NEW)
    # (If you prefer NOT to do this, comment it out.)
    if micro_step != 0:
        try:
            if clip_grad_norm and float(clip_grad_norm) > 0:
                if use_cuda:
                    scaler.unscale_(optimizer)
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            global_steps += 1
            logger.info("🔧 Flushed final partial grad accumulation step.")
        except Exception as e:
            logger.warning(f"Could not flush partial accumulation step: {e}")

    to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(to_save.state_dict(), final_model_path)
    logger.info(f"💾 Final model saved to {final_model_path}")

    # Save loss curve (optional / end-of-run)
    with open(loss_curve_path, "w", encoding="utf-8") as f:
        json.dump(loss_log, f, indent=2)

    total_time = time.time() - start_time
    summary = {
        "run_id": run_id,
        "final_tokens": int(total_tokens_processed),
        "final_loss": (loss_log[-1].get("train_loss", None) if loss_log else None),
        "final_val_loss": (loss_log[-1].get("val_loss", None) if loss_log else None),
        "optimizer_steps": int(global_steps),
        "total_time_sec": float(total_time),
        "interrupted": bool(interrupted),
        "gpu_peak_mem_gb": peak_gpu_mem_gb(),
        "final_model_path": final_model_path,
        "loss_curve_path": loss_curve_path,
        "metrics_jsonl": metrics_path,
        "run_meta_json": run_meta_path,
        "config": config,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # NEW: end event
    write_jsonl(metrics_path, {
        "type": "run_end",
        "ts": time.time(),
        "interrupted": bool(interrupted),
        "tokens_total": int(total_tokens_processed),
        "optimizer_steps": int(global_steps),
        "elapsed_s": float(total_time),
        "gpu_peak_mem_gb": peak_gpu_mem_gb(),
    })

    logger.info(f"🧾 Summary saved: {summary_path}")
    logger.info(f"🧾 Loss curve saved: {loss_curve_path}")
    logger.info(f"🧾 Metrics saved: {metrics_path}")
