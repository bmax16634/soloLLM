import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import GPT2Tokenizer

from .model import SoloGPT_v1


BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_CHECKPOINT = BASE / "outputs" / "finetuned_sologpt_combined_best.pth"

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_text = config.get("prompt", "").strip() or "The universe"

_MODEL = None
_TOKENIZER = None
_LOADED_CHECKPOINT = None


def _device_str(torch_device: torch.device) -> str:
    if torch_device.type == "cuda":
        return f"cuda:{torch_device.index if torch_device.index is not None else 0}"
    return torch_device.type


def _load_state_dict(checkpoint_path: Path, torch_device: torch.device):
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext in (".bin", ".pth"):
        state_dict = torch.load(checkpoint_path, map_location=torch_device)
    elif ext == ".safetensors":
        state_dict = load_file(str(checkpoint_path), device=_device_str(torch_device))
    else:
        raise ValueError(f"Unsupported checkpoint format: {ext}")

    if isinstance(state_dict, dict) and "model_state" in state_dict:
        return state_dict["model_state"]
    return state_dict


def load_generator(checkpoint_path: Path | str = DEFAULT_CHECKPOINT):
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if _MODEL is not None and _TOKENIZER is not None and _LOADED_CHECKPOINT == checkpoint_path:
        return _MODEL, _TOKENIZER

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = SoloGPT_v1(config).to(device)
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _LOADED_CHECKPOINT = checkpoint_path
    return model, tokenizer


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
) -> str:
    model, tokenizer = load_generator(checkpoint_path)

    max_new_tokens = int(max_new_tokens or config.get("output_len", 100))
    temperature = float(temperature or config.get("temperature", 1.0))
    top_k = int(top_k or config.get("k", 40))
    eos_token_id = tokenizer.eos_token_id

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


if __name__ == "__main__":
    print("prompt:", prompt_text)
    output = generate(prompt_text)
    print(output)
