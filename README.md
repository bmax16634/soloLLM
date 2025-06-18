# soloLLM

Custom GPT-style language model framework built from scratch in PyTorch and trained on consumer hardware (RTX 3090).  
Includes efficient data loading, tokenization, training, and generation workflows.

---

## 🚀 Available Models

| Model              | Description                           | Hugging Face |
|-------------------|---------------------------------------|--------------|
| `SoloGPT-base-v1` | Pretrained from scratch on 5.1B tokens | [🔗 Link](https://huggingface.co/yourname/SoloGPT-base-v1) |
| `SoloGPT-ft-v1`   | Fine-tuned for Q&A (coming soon)       | Coming soon  |

---

## Model Comparsion

| Model                | Params | Layers | Hidden Size | Context Length | Training Tokens | Hardware        | Perplexity (est.) |
| -------------------- | ------ | ------ | ----------- | -------------- | --------------- | --------------- | ----------------- |
| **SoloGPT-base-v1**  | \~13M  | 6      | 512         | 128            | 5.1B            | RTX 3090 (1x)   | **31.4**          |
| GPT-2 Small          | 117M   | 12     | 768         | 1024           | 40B             | 8x V100 (est.)  | \~25.3            |
| GPT-3 Ada (smallest) | 350M   | 24     | 1024        | 2048           | 300B+           | Massive cluster | \~20.0            |



## 🛠️ Setup

```bash
conda create -n solollm python=3.10
conda activate solollm
pip install -r requirements.txt
```

## Example: Text Generation

```python
from models.soloGPT_v1_model import SoloGPT
from config.soloGPT_v1_config import load_config
import torch

model = SoloGPT(load_config("config/soloGPT_v1_config.json"))
model.load_state_dict(torch.load("checkpoints/solo_gpt_base_v1/best.pth"))
model.eval()

output = model.generate(prompt="In a future world,", max_new_tokens=50)
print(output)
```

## 📤 Model Weights
Model weights are available on Hugging Face:
👉 SoloGPT-base-v1 on Hugging Face

## 📄 License
This project is licensed under the MIT License.




## 👤 Author
Benjamin Maxwell
📧 benjamin.davis.maxwell@gmail.com
🔗 GitHub | LinkedIn