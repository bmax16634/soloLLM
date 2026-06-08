# SoloLLM

SoloLLM is a from-scratch GPT-style language model project built in PyTorch. It includes the core pieces needed to train, evaluate, fine-tune, and serve a small decoder-only language model on consumer hardware.

The project is organized as a portfolio case study: `sologpt_v1` is the published baseline, and `sologpt_v2` is the next iteration focused on stronger training infrastructure and a better transformer block.

## Highlights

- Custom decoder-only transformer implementation in PyTorch.
- OpenWebText tokenization and sharded pretraining pipeline.
- Perplexity evaluation against a held-out OpenWebText shard.
- Streamlit text-generation demo.
- Hugging Face model artifact for the published v1 checkpoint.
- v2 implementation with RoPE attention, tied embeddings, checkpoint metadata, JSONL metrics, validation hooks, gradient clipping, resume support, and CPU smoke tests.

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
| `docs/PROJECT_BRIEF.md` | Portfolio case-study scaffold. |
| `docs/PHASE_0_CHECKLIST.md` | Phase 0 acceptance checklist. |
| `docs/PHASE_1_PLAN.md` | Two-variant v2 architecture and 50M sanity plan. |
| `docs/PHASE_2_PLAN.md` | 300M modern-small pilot plan and completion status. |
| `docs/PHASE_4_EVAL_PLAN.md` | Final robust comparison plan beyond perplexity alone. |
| `docs/V3_PLAN.md` | V2-to-GPT-2 gap analysis and v3 plan. |
| `docs/results/phase1_50m_sanity.md` | Completed 50M sanity results and Phase 1 decision. |
| `docs/results/phase2_300m_pilot.md` | Completed 300M pilot results and Phase 2 decision. |
| `docs/results/final_3b_modern_small.md` | Final v2 3B-token training and Phase 4 evaluation results. |
| `docs/results/v2_gpt2_full_diagnostic.md` | Full v2 5.60B vs GPT-2 diagnostic across held-out, external, generation, and multiple-choice evals. |
| `docs/results/phase4_generations.md` | Fixed qualitative generation samples for v1, v2, and GPT-2. |
| `tests/` | CPU-safe smoke tests and tiny v2 config. |

Generated datasets, checkpoints, training logs, and model weights are intentionally ignored by Git. Published weights live on Hugging Face instead of in this repository.

## Model Variants

| Variant | Status | Main Idea |
| --- | --- | --- |
| `sologpt_v1` | Published baseline | Custom GPT-style decoder with learned token embeddings, sinusoidal positional encoding, causal self-attention, and MLP blocks. |
| `v2-gpt2-parity` | 50M sanity complete | 123,616,512-param GPT-2-small-scale control with RoPE, pre-norm blocks, tied embeddings, CLI training, and fair-eval path. |
| `v2-modern-small` | Final + stretch runs complete | 91,654,400-param modern-small config with RMSNorm, SwiGLU, RoPE, tied embeddings, 5.60B-checkpoint full held-out PPL 25.56, and clear improvement over v1. |

Published v1 artifact:

- Hugging Face: <https://huggingface.co/bmax16634/sologpt-base-v1>

## Setup

```bash
conda create -n solollm python=3.10
conda activate solollm
pip install -r requirements.txt
```

The requirements file uses the PyTorch CUDA 11.8 wheel index. For CPU-only or a different CUDA version, install the matching PyTorch build for your machine first, then install the remaining requirements.

## Smoke Tests

Run the CPU-safe checks:

```bash
python -m pytest
```

The tests instantiate v1 and v2, verify v2 context checks and tied embeddings, run a tiny v2 training dry run, resume from `checkpoints/latest.pt`, and run the eval CLI on a tiny generated shard. They do not require OpenWebText, model weights, a GPU, or internet access.

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
python -m sologpt_v2.pretrain \
  --config sologpt_v2/config.json \
  --shard-dir data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/final_v2 \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 4000000000
```

Both configs expect tokenized shards under `data/tokenized_chunks` by default. Override `shard_dir`, `save_path`, and other hyperparameters in the relevant `config.json`. V2 writes structured artifacts into the run directory: `config_resolved.json`, `run_meta.json`, `metrics.jsonl`, `metrics_summary.json`, `checkpoints/latest.pt`, and `final_model.pt`.

Run the v2 50M-token sanity check:

```bash
python -m sologpt_v2.pretrain \
  --config sologpt_v2/config.json \
  --shard-dir data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/sanity_50m \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 50000000 \
  --max-eval-tokens 5000000
```

Resume v2 training:

```bash
python -m sologpt_v2.pretrain \
  --config sologpt_v2/config.json \
  --shard-dir data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/final_v2_resumed \
  --resume outputs/sologpt_v2/final_v2/checkpoints/latest.pt \
  --train-shards 0:54 \
  --val-shards 55:57
```

## Evaluation

Final Phase 4 held-out perplexity on reserved shards `58:60`:

| Model | Params | Train tokens | Test PPL | Notes |
| --- | ---: | ---: | ---: | --- |
| `sologpt_v1` | 203.75M | 9.03B declared target, actual unverified | 30.48 | Published HF checkpoint |
| `v2-modern-small` | 91.65M | 3.00B confirmed | 26.26 | Single RTX 3090 from-scratch run |
| `v2-modern-small` stretch | 91.65M | 5.60B checkpoint metadata | 25.56 | Best official v2 checkpoint |
| GPT-2 small | 124.44M | external/public | 25.32 | Reference baseline |

The best v2 checkpoint is about 55% smaller than v1 and beats v1 by about 16.1% lower held-out perplexity. It is about 26% smaller than GPT-2 small and lands very close to, but does not beat, GPT-2 on this test: about 0.95% higher perplexity and +0.0094 loss.

Stretch-checkpoint full eval, completed June 8, 2026:

| Model | Params | Checkpoint / train tokens | Eval tokens | Test loss | Test PPL | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `sologpt_v1` | 203.75M | Published HF checkpoint; 9.03B declared target, actual unverified | 331,353,862 | 3.4170 | 30.48 | Historical baseline |
| `v2-modern-small` final | 91.65M | 3.00B confirmed | 331,353,862 | 3.2682 | 26.26 | Main 3B checkpoint |
| `v2-modern-small` stretch | 91.65M | `latest.pt`, metadata `5,600,206,848` tokens | 331,353,862 | 3.2408 | 25.56 | Durable 5.60B checkpoint |
| GPT-2 small | 124.44M | external/public | 331,353,862 | 3.2314 | 25.32 | Reference baseline |

The 5.60B stretch checkpoint improves over the 3B checkpoint by about `2.7%` lower perplexity on the same full held-out split. It remains behind GPT-2 small, but the full-eval gap is narrow: about `+0.0094` loss and `+0.95%` perplexity.

Additional robust comparison checks were added for the 5.60B checkpoint. On the fixed prompt suite, v2 and GPT-2 have similar diversity metrics, but GPT-2 has slightly lower repeated bigram/trigram rates. On full external corpora, GPT-2 pulls ahead more clearly: WikiText-2 PPL `49.86` vs v2 `84.81`, LAMBADA PPL `42.26` vs v2 `63.03`, LAMBADA last-token accuracy `46.67%` vs v2 `44.30%`, and LAMBADA last-word greedy exact `32.60%` vs v2 `29.09%`. GPT-2 also leads all five length-normalized multiple-choice checks. The honest conclusion is that v2 is very close on the project held-out split, but GPT-2 remains more robust across the board. See [docs/results/v2_gpt2_full_diagnostic.md](docs/results/v2_gpt2_full_diagnostic.md).

Run perplexity evaluation:

```bash
python eval/eval.py \
  --model v2 \
  --checkpoint outputs/sologpt_v2/final_v2/final_model.pt \
  --config sologpt_v2/config.json \
  --shard-dir data/tokenized_chunks \
  --shards 58:60 \
  --batch-size 8 \
  --output-json outputs/eval/v2_test.json
```

The eval CLI supports `--model v1`, `--model v2`, and `--model gpt2`. For fair portfolio results, use the same shard range and shifted-token loss path for every model.

Run the fixed qualitative generation suite:

```bash
python -m eval.generate_samples \
  --models v1 v2 gpt2 \
  --output-json outputs/sologpt_v2/final_3b_modern_small_from300m/phase4_generations.json \
  --output-md docs/results/phase4_generations.md
```

Run the automatic generation metrics:

```bash
python -m eval.generation_metrics \
  --input-json outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/phase4_generations_5p6b_v2_gpt2.json \
  --output-json outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/phase4_generation_metrics_5p6b_v2_gpt2.json \
  --output-md docs/results/phase4_generation_metrics_5p6b_v2_gpt2.md
```

Run the compact external benchmark pass:

```bash
python -m eval.external_benchmarks \
  --models v2 gpt2 \
  --benchmarks wikitext2 lambada \
  --v2-checkpoint outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/checkpoints/latest.pt \
  --v2-config sologpt_v2/config_modern_small.json \
  --context-length 512 \
  --max-wikitext-tokens 0 \
  --max-lambada-examples 0 \
  --output-json outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/external_benchmarks_5p6b_v2_gpt2_full.json \
  --output-md docs/results/external_benchmarks_5p6b_v2_gpt2_full.md
```

Run the multiple-choice base-LM benchmark pass:

```bash
python -m eval.multiple_choice_benchmarks \
  --models v2 gpt2 \
  --benchmarks hellaswag piqa arc_easy arc_challenge winogrande \
  --v2-checkpoint outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/checkpoints/latest.pt \
  --v2-config sologpt_v2/config_modern_small.json \
  --context-length 512 \
  --max-examples 0 \
  --output-json outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/multiple_choice_5p6b_v2_gpt2_full.json \
  --output-md docs/results/multiple_choice_5p6b_v2_gpt2_full.md
```

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

The current v2 default is a 12-layer, 768-width, 12-head, 512-context model with tied embeddings, measuring 123,616,512 parameters. That intentionally matches GPT-2-small scale, making the final held-out perplexity comparison easier to interpret than a vague "bigger model" claim.

Phase 1 results are recorded in [docs/results/phase1_50m_sanity.md](docs/results/phase1_50m_sanity.md). The 91,654,400-param `v2-modern-small` variant was smaller, faster, and reached lower capped validation perplexity than the 123,616,512-param parity control.

Phase 2 results are recorded in [docs/results/phase2_300m_pilot.md](docs/results/phase2_300m_pilot.md). The 300M-token pilot completed with final capped validation PPL `42.95`, peak GPU memory about `10.48GB`, and enough stability to promote `v2-modern-small` to the final long-run phase.

The final v2 run is recorded in [docs/results/final_3b_modern_small.md](docs/results/final_3b_modern_small.md). The 3B-token run completed on a single RTX 3090 with final validation PPL `26.13`, full held-out test PPL `26.26`, and average throughput about `50k tok/s`. A durable 5.60B stretch checkpoint later reached full held-out test PPL `25.56`.

The v3 direction is recorded in [docs/V3_PLAN.md](docs/V3_PLAN.md). It documents what the v2 gaps to GPT-2 show, how v3 should address them, and why the v2 eval suite should be reused but expanded before claiming v3 beats GPT-2 across the board.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Benjamin Maxwell

- GitHub: <https://github.com/bmax16634>
- LinkedIn: <https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/>
- Hugging Face: <https://huggingface.co/bmax16634>
