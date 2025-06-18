import torch
import json
from models.soloGPT_v1_model import SoloGPT_v1
import tiktoken

# ==== Load Config ====
with open("config/soloGPT_v1_config.json", "r") as f:
    config = json.load(f)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#infer_cfg = config["inference"]
prompt = config['prompt']
k = config['k']
temp = config['temperature']
out_len = config['output_len']
save_path = 'outputs/pytorch_model.bin'

# ==== Load tokenizer ====
tokenizer = tiktoken.get_encoding("gpt2")

# ==== Load model ====
model = SoloGPT_v1(config).to(device)
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
        prompt=(prompt, "The universe"),
        max_new_tokens=(out_len, 100),
        temperature=(temp, 1.0),
        top_k=(k, 40)
    ))
