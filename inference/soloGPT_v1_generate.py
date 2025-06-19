import os
import json
import torch
import tiktoken
from models.soloGPT_v1_model import SoloGPT_v1
from safetensors.torch import load_file
from pathlib import Path

# ─── two levels up from this file ───
BASE = Path(__file__).resolve().parent.parent
cfg_file = BASE / "config" / "soloGPT_v1_config.json"

# ==== Load Config ====
with open(cfg_file, "r") as f:
    config = json.load(f)

# ==== Select device ====
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ==== Inference settings from config ====
base_prompt   = config.get("prompt", "").strip() or "The universe"
max_new_tokens = config.get("output_len", 100)
temperature    = config.get("temperature", 1.0)
top_k          = config.get("k", 40)
save_path      = BASE / "outputs" / "pytorch_model.safetensors"  # or .pth / .safetensors

# ==== Load tokenizer ====
tokenizer = tiktoken.get_encoding("gpt2")

# ==== Load model architecture ====
model = SoloGPT_v1(config).to(device)

# ==== Load checkpoint (.bin, .pth, or .safetensors) ====
ext = os.path.splitext(save_path)[1].lower()

if ext in (".bin", ".pth"):
    # standard PyTorch checkpoint directly to GPU/CPU
    state_dict = torch.load(save_path, map_location=device)

elif ext == ".safetensors":
    # safetensors: pass "cuda:0" if you want GPU 0,
    # or "cuda:1" for GPU 1, etc.
    # Map your torch.device into a string:
    if device.type == "cuda":
        # e.g. torch.device("cuda")  →  "cuda:0"
        device_str = f"cuda:{device.index or 0}"
    else:
        device_str = "cpu"
    state_dict = load_file(save_path, device=device_str)

else:
    raise ValueError(f"Unsupported checkpoint format: '{ext}'")

model.load_state_dict(state_dict)
model.to(device).eval()

# ==== Generation function ====
@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 40
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated)
        logits = logits[:, -1, :]  # last-token logits

        # top-k sampling
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(topk_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = topk_indices.gather(-1, next_token)

        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0].tolist())

# ==== Run ====
if __name__ == "__main__":
    output = generate(
        prompt=base_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    print(output)
