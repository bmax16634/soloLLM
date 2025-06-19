import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.soloGPT_v1_model import SoloGPT_v1

# --- Config & Device ---
with open("config/soloGPT_v1_config.json") as f:
    config = json.load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class HeldoutDataset(Dataset):
    def __init__(self, tensor_path):
        self.data = torch.load(tensor_path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(tensor_path, batch_size=1):
    return DataLoader(HeldoutDataset(tensor_path), batch_size=batch_size)

# --- Perplexity evaluation ---
def evaluate_perplexity(model, dataloader, is_hf_model=False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model.__class__.__name__}"):
            input_ids = batch.to(DEVICE)
            if is_hf_model:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss * input_ids.numel()
                total_tokens += input_ids.numel()
            else:
                logits = model(input_ids)
                logits = logits[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                total_tokens += targets.numel()

            total_loss += loss.item()

    return np.exp(total_loss / total_tokens)

# --- SoloGPT loader function ---
def load_solo_gpt():
    m = SoloGPT_v1(config).to(DEVICE)
    ckpt = torch.load(
        "/home/bmx/_projects/miniGPT2/eval/pretrained_gpt_6000M.pth",
        map_location=DEVICE
    )
    m.load_state_dict(ckpt)
    return m

# --- Define your models ---
models_to_evaluate = [
    {
        "name": "SoloGPT_v1 (6000M)",
        "loader": load_solo_gpt,
        "is_hf": False
    },
    {
        "name": "GPT-2 (124M)",
        "loader": lambda: GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE),
        "is_hf": True
    },
    # add more here...
]

# --- Run ---
shard_path = "/home/bmx/_projects/miniGPT2/eval/shard_00060.pt"
dataloader = get_dataloader(shard_path, batch_size=1)

results = {}
for spec in models_to_evaluate:
    print(f"\n--- {spec['name']} ---")
    model = spec["loader"]()
    ppl = evaluate_perplexity(model, dataloader, is_hf_model=spec["is_hf"])
    results[spec["name"]] = ppl
    print(f"{spec['name']} perplexity: {ppl:.2f}")

print("\n=== Comparison ===")
for name, ppl in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:20s} ➜  PPL = {ppl:.2f}")
