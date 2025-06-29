# inference/soloGPT_v1_generate.py
import os
import json
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
from models.soloGPT_v1_model import SoloGPT_v1
from safetensors.torch import load_file

# ==== Config ====
BASE = Path(__file__).resolve().parent.parent
cfg_file = BASE / "config" / "soloGPT_v1_config.json"

with open(cfg_file) as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_text      = config.get("prompt", "").strip() or "The universe"
max_new_tokens   = config.get("output_len", 100)
temperature      = config.get("temperature", 1.0)
top_k            = config.get("k", 40)
save_path        = BASE / "outputs" / "finetuned_sologpt_combined_best.pth"

# ==== Tokenizer ====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

# ==== Load model ====
model = SoloGPT_v1(config).to(device)
ext = os.path.splitext(save_path)[1].lower()
state_dict = load_file(str(save_path), device=device) if ext == ".safetensors" else torch.load(save_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ==== Manual Generation ====
@torch.no_grad()
def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
        probs = torch.softmax(top_k_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(-1, next_token)

        if next_token.item() == eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# ==== Run ====
if __name__ == "__main__":
    print("prompt:", prompt_text)
    output = generate(prompt_text)
    print(output)
