import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset, concatenate_datasets
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
epochs = 500
max_length = 1024
patience = 2
best_val_loss = float("inf")
epochs_no_improve = 0

# --- Load model ---
model = SoloGPT_v1(config).to(device)
model.load_state_dict(torch.load("outputs/pytorch_model.pth"))
model.train()
os.makedirs("outputs", exist_ok=True)

# --- Tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

def format_sample(prompt: str, answer: str):
    if not prompt.strip() or not answer.strip():
        return None
    full_input = f"Instruction: {prompt.strip()}\nAnswer: {answer.strip()}{tokenizer.eos_token}"
    out = tokenizer(full_input, truncation=True, padding="max_length", max_length=max_length)
    return {"input_ids": out["input_ids"]}

def format_dolly(example):
    prompt = example.get("instruction", "")
    context = example.get("context", "")
    answer = example.get("response", "")
    full_prompt = f"{prompt.strip()}\n{context.strip()}" if context.strip() else prompt.strip()
    return format_sample(full_prompt, answer)

def format_oasst(example):
    prompt = msg_lookup.get(example["parent_id"], "")
    answer = example["text"]
    return format_sample(prompt, answer)

# --- Load & prepare datasets ---
print("📦 Loading Dolly...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
dolly = dolly.map(format_dolly).filter(lambda x: x is not None and isinstance(x.get("input_ids"), list))

print("📦 Loading OASST1...")
raw_oasst = load_dataset("OpenAssistant/oasst1", split="train")
msg_lookup = {m["message_id"]: m["text"] for m in raw_oasst if m["role"] == "prompter"}
oasst_filtered = raw_oasst.filter(lambda x: x["role"] == "assistant" and x.get("parent_id") in msg_lookup)
oasst = oasst_filtered.map(format_oasst).filter(lambda x: x is not None and isinstance(x.get("input_ids"), list))

print("🔗 Combining datasets...")
full_dataset = concatenate_datasets([dolly, oasst])

class CombinedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.samples = hf_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx]["input_ids"])
        return ids, ids.clone()




split = full_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = CombinedDataset(split["train"])
val_dataset = CombinedDataset(split["test"])
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Training ---
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

def evaluate(model, loader):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
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
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
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

    # --- Early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "outputs/finetuned_sologpt_combined_best.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered.")
            break

# --- Save ---
torch.save(model.state_dict(), "outputs/finetuned_sologpt_combined.pth")
print("🎉 Done. Model saved to outputs/finetuned_sologpt_combined.pth")
