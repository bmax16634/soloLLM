# 🧠 SoloLLM

**SoloLLM** is a custom GPT-style language model framework built entirely from scratch in PyTorch and trained on consumer-grade hardware (RTX 3090).  
It includes efficient pipelines for tokenization, training, and text generation.

---

## 🚀 Available Models

| Model              | Description                            | Hugging Face                       |
|-------------------|----------------------------------------|------------------------------------|
| `SoloGPT-base-v1` | Pretrained from scratch on 5.1B tokens | [🔗 View on HF](https://huggingface.co/bmax16634/sologpt-base-v1) |
| `SoloGPT-ft-v1`   | Fine-tuned for Q&A                     | _Coming soon_                      |

---

## 📊 Model Comparison

| Model                | Params | Layers | Hidden Size | Context Length | Training Tokens | Hardware      | Est. Perplexity |
|----------------------|--------|--------|--------------|----------------|------------------|----------------|-----------------|
| **SoloGPT-base-v1**  | ~13M   | 6      | 512          | 128            | 5.1B             | RTX 3090 (1x)  | **31.4**         |
| GPT-2 Small          | 117M   | 12     | 768          | 1024           | 40B              | 8× V100 (est.) | ~25.3            |
| GPT-3 Ada            | 350M   | 24     | 1024         | 2048           | 300B+            | Large cluster  | ~20.0            |

---

Perplexity for SoloGPT-base-v1 and GPT-2 Small was evaluated on shard 61 of the OpenWebText dataset (not seen during training).

## ⚙️ Setup

```bash
conda create -n solollm python=3.10
conda activate solollm
pip install -r requirements.txt
```

### 📁 Key Files

| File               | Description                               |
|--------------------|-------------------------------------------|
| `prepare_data.py`  | Tokenizes raw text into token ID sequences |
| `flatten.py`       | Saves preprocessed data into `.pt` chunks  |

---

## 🏋️‍♂️ Training

```bash
python -m train.soloGPT_v1_train
```

---

## ✨ Text Generation

```bash
python -m inference.soloGPT_v1_generate
```

Or use the Streamlit app:

```bash
streamlit run app.py
```

---

## 🧪 Example: Programmatic Usage

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

---

## 📦 Model Weights

Model weights are available via Hugging Face Hub:

👉 [`SoloGPT-base-v1`](https://huggingface.co/bmax16634/sologpt-base-v1)

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

## 👤 Author

**Benjamin Maxwell**  
📧 benjamin.davis.maxwell@gmail.com  
🔗 [GitHub](https://github.com/yourname) • [LinkedIn](https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/) • [Hugging Face](https://huggingface.co/bmax16634)
