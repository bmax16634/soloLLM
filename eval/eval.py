import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from models.soloGPT_v1_model import SoloGPT_v1
import os


batch_size:int = 1

# --- Config & Device ---
with open("config/soloGPT_v1_config.json") as f:
    config = json.load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Single‐shard Dataset & Loss‐calc helper ---
class HeldoutDataset(Dataset):
    def __init__(self, tensor_path):
        self.data = torch.load(tensor_path)
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]

def eval_shard(model, tensor_path, is_hf_model=False, batch_size=batch_size):
    """Load one shard, run through it, return (sum_loss, sum_tokens)."""
    loader = DataLoader(HeldoutDataset(tensor_path), batch_size=batch_size, num_workers=16, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    sum_loss = 0.0
    sum_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Shard {os.path.basename(tensor_path)}"):
            ids = batch.to(DEVICE)
            if is_hf_model:
                out = model(ids, labels=ids)
                # out.loss is avg per token
                tok = ids.numel()
                loss = out.loss.item() * tok
                sum_tokens += tok
            else:
                logits = model(ids)[:, :-1, :].contiguous()
                targets = ids[:, 1:].contiguous()
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                ).item()
                sum_tokens += targets.numel()
            sum_loss += loss

    return sum_loss, sum_tokens

# --- Model loaders ---
def load_solo_gpt():
    m = SoloGPT_v1(config).to(DEVICE)
    ckpt = torch.load("outputs/pytorch_model.bin", map_location=DEVICE)
    m.load_state_dict(ckpt)
    return m

models_to_eval = [
    ("SoloGPT_v1 (6000M)", load_solo_gpt, False),
    ("GPT-2 (124M)", lambda: GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE), True),
    # add more...
]

# --- Loop over shards 40–60, one at a time ---
shard_dir = "data/tokenized_chunks"
start, end = 40, 60

for name, loader_fn, is_hf in models_to_eval:
    model = loader_fn()
    total_loss = 0.0
    total_tokens = 0

    for idx in range(start, end + 1):
        path = os.path.join(shard_dir, f"shard_{idx:05d}.pt")
        loss, toks = eval_shard(model, path, is_hf_model=is_hf, batch_size=batch_size)
        total_loss += loss
        total_tokens += toks

    ppl = np.exp(total_loss / total_tokens)
    print(f"{name} → PPL = {ppl:.2f}")
