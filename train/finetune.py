# train/finetune.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from sologpt_v1.model import SoloGPT_v1

# --- Config ---
with open("sologpt_v1/config.json", encoding="utf-8") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
batch_size = 12
effective_batch_size = 24
gradient_accumulation_steps = effective_batch_size // batch_size
lr = 1e-6
epochs = 2
max_length = 1024

# --- Load pretrained model ---
model = SoloGPT_v1(config).to(device)
model.load_state_dict(torch.load("outputs/pytorch_model.pth"))
model.train()

# --- Tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

# --- Load Dolly dataset ---
def format_dolly(example):
    prompt = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    answer = example.get("response", "").strip()

    full_prompt = f"{prompt}\n{context}" if context else prompt
    full_input = f"Instruction: {full_prompt}\nAnswer: {answer}{tokenizer.eos_token}"

    if not full_prompt or not answer:
        return None

    encoded = tokenizer(full_input, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)

class DollyDataset(Dataset):
    def __init__(self, split="train"):
        raw = load_dataset("databricks/databricks-dolly-15k", split=split)
        self.data = []
        for ex in raw:
            ids = format_dolly(ex)
            if ids is not None and torch.any(ids != pad_token_id):
                self.data.append(ids)
        print(f"✅ Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return ids, ids.clone()

# --- Load datasets ---
train_dataset = DollyDataset()
val_dataset = DollyDataset()
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Training setup ---
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

def evaluate(model, loader):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    model.train()
    return total_loss / total_tokens

# --- Training loop ---
for epoch in range(epochs):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
    total_loss = 0
    optimizer.zero_grad()

    for step, (ids, labels) in enumerate(pbar):
        ids, labels = ids.to(device), labels.to(device)
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(ids)
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / gradient_accumulation_steps

        if torch.isnan(loss):
            print("⚠️ NaN detected. Skipping batch.")
            continue

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix(train_loss=f"{loss.item() * gradient_accumulation_steps:.4f}")

    avg_train_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)
    print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

# --- Save model ---
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/finetuned_sologpt_dolly15k.pth")
print("🎉 Done. Model saved to outputs/finetuned_sologpt_dolly15k.pth")
