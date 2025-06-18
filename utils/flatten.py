import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
import json

# ==== Config ====
with open("config/soloGPT_v1_config.json") as f:
    config = json.load(f)

SEQ_LEN = config['seq_length']
TOKENS_PER_SHARD = 150_000_000
SAVE_DIR = "data/tokenized_chunks"
CONFIG_PATH = "config/soloGPT_v1_config.json"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load tokenized dataset
print("📂 Loading tokenized dataset from disk...")
dataset = load_from_disk("data/openwebtext_tokenized")  # saved by prepare_data.py

# Flatten tokens in a memory-efficient way
print("🧱 Flattening token stream and saving shards...")
buffer = []
shard_idx = 0
total_tokens = 0

with tqdm(total=len(dataset), desc="Examples", unit="ex") as pbar:
    for ex in dataset:
        buffer.extend(ex["input_ids"])
        while len(buffer) >= TOKENS_PER_SHARD:
            chunk = buffer[:TOKENS_PER_SHARD]
            buffer = buffer[TOKENS_PER_SHARD:]

            tensor = torch.tensor(chunk, dtype=torch.long)
            num_seqs = len(tensor) // SEQ_LEN
            tensor = tensor[:num_seqs * SEQ_LEN].view(-1, SEQ_LEN)

            save_path = os.path.join(SAVE_DIR, f"shard_{shard_idx:05d}.pt")
            torch.save(tensor, save_path)

            shard_idx += 1
            total_tokens += tensor.numel()
        pbar.update(1)

# Final save if there's leftover data
if len(buffer) >= SEQ_LEN:
    tensor = torch.tensor(buffer, dtype=torch.long)
    num_seqs = len(tensor) // SEQ_LEN
    tensor = tensor[:num_seqs * SEQ_LEN].view(-1, SEQ_LEN)
    torch.save(tensor, os.path.join(SAVE_DIR, f"shard_{shard_idx:05d}.pt"))
    total_tokens += tensor.numel()
   
print(f"\n✅ Done. Saved {shard_idx + 1} shards. Total tokens: {total_tokens:,}")   
 
# ==== Update config.json with total_tokens ====
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["total_tokens"] = total_tokens

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"📝 Updated config.json with total_tokens: {total_tokens:,}")
else:
    print("⚠️ config.json not found. Skipping config update.")






