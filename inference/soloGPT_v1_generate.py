import torch
import json
from model import CustomDecoderGPT
import tiktoken

# ==== Load Config ====
with open("config.json", "r") as f:
    config = json.load(f)

device = torch.device(config["general"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
model_cfg = config["model"]
infer_cfg = config["inference"]
save_path = config["general"]["save_path"]

# ==== Load tokenizer ====
tokenizer = tiktoken.get_encoding("gpt2")
model_cfg["vocab_size"] = tokenizer.n_vocab  # Update vocab size in config just in case

# ==== Load model ====
model = CustomDecoderGPT(config).to(device)
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

# ==== Generation Function ====
@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=1.0, top_k=40):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated)
        logits = logits[:, -1, :]  # Only the last token's logits

        # Top-k sampling
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(top_k_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        next_token = top_k_indices.gather(-1, next_token)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0].tolist())

# ==== Run ====
if __name__ == "__main__":
    print(generate(
        prompt=infer_cfg.get("prompt", "The universe"),
        max_new_tokens=infer_cfg.get("output_len", 100),
        temperature=infer_cfg.get("temperature", 1.0),
        top_k=infer_cfg.get("k", 40)
    ))
