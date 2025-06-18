import os
import json
import logging
import time
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.soloGPT_v1_model import SoloGPT_v1
from tqdm import tqdm

# ---------- Config & Setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/soloGPT_v1_config.json") as f:
    config = json.load(f)

device = torch.device(config["general"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
save_path = config["general"].get("save_path", "pretrained_model.pth")
shard_dir = config["general"].get("shard_dir", "tokenized_chunks")
val_path = config["general"].get("val_path")  # Optional

batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
weight_decay = config["training"]["weight_decay"]
target_tokens = config["training"].get("total_tokens", 300_000_000_000)
tokens_per_checkpoint = config["training"].get("tokens_per_checkpoint", 300_000_000)
num_workers = config["training"].get("num_workers", 0)
pin_memory = config["training"].get("pin_memory", False)
grad_accum_steps = config["training"].get("grad_accum_steps", 1)

logger.info(f"Using device: {device}")

# ---------- Dataset ----------
class TokenizedShardDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader_from_shard(shard_path):
    logger.info(f"🔹 Loading shard: {shard_path}")
    data = torch.load(shard_path)
    dataset = TokenizedShardDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in val_loader:
            input_ids = batch.to(device)
            tokens = input_ids.numel()
            logits = model(input_ids)
            logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

# ---------- Model Setup ----------
model = SoloGPT_v1(config).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler("cuda")
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# ---------- Optional Validation Setup ----------
val_loader = None
if val_path and os.path.isfile(val_path):
    logger.info(f"🔸 Loading validation shard: {val_path}")
    val_data = torch.load(val_path)
    val_loader = DataLoader(TokenizedShardDataset(val_data), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ---------- Training ----------
logger.info("🚀 Starting training across shards...")

total_tokens_processed = 0
step_tokens = 0
step_count = 0
loss_log = []

progress_total = tqdm(total=target_tokens, desc="Total Training", unit="tok", dynamic_ncols=True)
progress_ckpt = tqdm(total=tokens_per_checkpoint, desc="Next Checkpoint", unit="tok", dynamic_ncols=True, leave=False)

shard_paths = sorted(glob(os.path.join(shard_dir, "*.pt")))
shard_idx = 0
start_time = time.time()

while total_tokens_processed < target_tokens and shard_idx < len(shard_paths):
    dataloader = get_dataloader_from_shard(shard_paths[shard_idx])
    model.train()

    for batch in dataloader:
        input_ids = batch.to(device)
        tokens_this_batch = input_ids.numel()

        with torch.amp.autocast("cuda"):
            logits = model(input_ids)
            logits = logits[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step_count + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_tokens_processed += tokens_this_batch
        step_tokens += tokens_this_batch
        step_count += 1

        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens_processed / elapsed
        iterations_per_sec = step_count / elapsed

        progress_total.update(tokens_this_batch)
        progress_ckpt.update(tokens_this_batch)
        progress_total.set_postfix(
            loss=f"{loss.item() * grad_accum_steps:.4f}",
            speed=f"{tokens_per_sec:,.0f} tok/s",
            iters=f"{iterations_per_sec:.2f} it/s"
        )

        loss_log.append({
            "tokens": total_tokens_processed,
            "train_loss": loss.item() * grad_accum_steps
        })

        if step_tokens >= tokens_per_checkpoint:
            ckpt_name = f"{os.path.splitext(save_path)[0]}_{total_tokens_processed//1_000_000}M.pth"
            torch.save(model.state_dict(), ckpt_name)
            logger.info(f"✅ Checkpoint saved at {total_tokens_processed:,} tokens → {ckpt_name}")

            if val_loader:
                val_loss = evaluate(model, val_loader)
                logger.info(f"📉 Validation loss: {val_loss:.4f}")
                loss_log[-1]["val_loss"] = val_loss

            step_tokens = 0
            progress_ckpt.reset()
            break  # move to next shard

    shard_idx += 1

# Final save
torch.save(model.state_dict(), save_path)
logger.info(f"🏁 Training complete. Total tokens: {total_tokens_processed:,}. Final model saved to {save_path}")

# Save loss log
with open("loss_curve.json", "w") as f:
    json.dump(loss_log, f, indent=2)

# Save training summary
with open("training_summary.json", "w") as f:
    json.dump({
        "final_tokens": total_tokens_processed,
        "final_loss": loss_log[-1].get("train_loss", None),
        "final_val_loss": loss_log[-1].get("val_loss", None),
        "total_time_sec": time.time() - start_time,
        "config": config
    }, f, indent=2)
