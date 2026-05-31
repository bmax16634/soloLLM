# SoloLLM

SoloLLM is a from-scratch GPT-style language model project built in PyTorch. It includes the core pieces needed to train, evaluate, fine-tune, and serve a small decoder-only language model on consumer hardware.

The project is organized as a portfolio case study: `sologpt_v1` is the published baseline, and `sologpt_v2` is the next iteration focused on stronger training infrastructure and a better transformer block.

## Highlights

- Custom decoder-only transformer implementation in PyTorch.
- OpenWebText tokenization and sharded pretraining pipeline.
- Perplexity evaluation against a held-out OpenWebText shard.
- Streamlit text-generation demo.
- Hugging Face model artifact for the published v1 checkpoint.
- v2 prototype with rotary positional embeddings, checkpoint metadata, JSONL metrics, validation hooks, gradient clipping, and resumable checkpoint structure.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `sologpt_v1/` | Published baseline model, config, pretraining, and generation code. |
| `sologpt_v2/` | Work-in-progress v2 model and training loop. |
| `utils/prepare_data.py` | Downloads and tokenizes OpenWebText. |
| `utils/flatten.py` | Converts tokenized data into fixed-length `.pt` shards. |
| `train/finetune.py` | Dolly fine-tuning script for v1. |
| `eval/eval.py` | Perplexity evaluation script. |
| `app.py` | Streamlit generation demo. |
| `docs/V2_PLAN.md` | Planned second iteration for the portfolio. |

Generated datasets, checkpoints, training logs, and model weights are intentionally ignored by Git. Published weights live on Hugging Face instead of in this repository.

## Model Variants

| Variant | Status | Main Idea |
| --- | --- | --- |
| `sologpt_v1` | Published baseline | Custom GPT-style decoder with learned token embeddings, sinusoidal positional encoding, causal self-attention, and MLP blocks. |
| `sologpt_v2` | Prototype | Larger config, RoPE attention, improved training logs, checkpoint metadata, validation events, and safer config-driven paths. |

Published v1 artifact:

- Hugging Face: <https://huggingface.co/bmax16634/sologpt-base-v1>

## Setup

```bash
conda create -n solollm python=3.10
conda activate solollm
pip install -r requirements.txt
```

The requirements file uses the PyTorch CUDA 11.8 wheel index. For CPU-only or a different CUDA version, install the matching PyTorch build for your machine first, then install the remaining requirements.

## Data Pipeline

Tokenize OpenWebText:

```bash
python utils/prepare_data.py
```

Flatten tokenized examples into fixed-length training shards:

```bash
python utils/flatten.py
```

The generated files are written under `data/`, which is ignored by Git.

## Training

Train the v1 baseline:

```bash
python -m sologpt_v1.pretrain
```

Train the v2 prototype:

```bash
python -m sologpt_v2.pretrain
```

Both configs expect tokenized shards under `data/tokenized_chunks` by default. Override `shard_dir`, `save_path`, and other hyperparameters in the relevant `config.json`.

## Evaluation

Run perplexity evaluation:

```bash
python eval/eval.py
```

The current eval script compares SoloGPT against GPT-2 on local tokenized shards. It expects the local checkpoint path configured in the script to exist.

## Generation Demo

Run the Streamlit demo:

```bash
streamlit run app.py
```

By default the app looks for:

```text
outputs/finetuned_sologpt_combined_best.pth
```

Model checkpoints are not committed to Git. Place a compatible `.pth`, `.bin`, or `.safetensors` checkpoint in `outputs/`, or adapt `sologpt_v1/generate.py` to load from a Hugging Face download path.

## V2 Direction

The next portfolio iteration is tracked in [docs/V2_PLAN.md](docs/V2_PLAN.md). The practical goal is to make v2 a clear improvement over v1 in three visible ways:

1. Better architecture: RoPE, cleaner attention implementation, optional tied embeddings, and stronger defaults.
2. Better training system: config-driven paths, resumable checkpoints, structured metrics, validation, and cleaner failure recovery.
3. Better project presentation: reproducible commands, model cards, before/after evaluation, and demo examples.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Benjamin Maxwell

- GitHub: <https://github.com/bmax16634>
- LinkedIn: <https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/>
- Hugging Face: <https://huggingface.co/bmax16634>
