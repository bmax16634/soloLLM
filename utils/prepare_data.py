import os
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer

# ==== Config ====
TOKENIZER_NAME = "gpt2"
SAVE_PATH = "data/openwebtext_tokenized"
NUM_PROC = multiprocessing.cpu_count()

# ==== Setup ====
print("🔽 Downloading & loading OpenWebText locally...")
dataset = load_dataset("Skylion007/openwebtext", split="train")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"

# ==== Tokenize ====
def tokenize_function(examples):
    return tokenizer(examples["text"], return_attention_mask=False, truncation=False)

print(f"🔤 Tokenizing with multiprocessing (num_proc={NUM_PROC})...")
tokenized = dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC)

# ==== Save ====
tokenized.save_to_disk(SAVE_PATH)
print(f"✅ Tokenized dataset saved to: {SAVE_PATH}")
