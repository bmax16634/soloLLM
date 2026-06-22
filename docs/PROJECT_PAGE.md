# SoloLLM: Training GPT-2-Class Language Models From Scratch

SoloLLM is an end-to-end language-modeling project built to show the full
engineering loop behind a small GPT-style base model: dataset construction,
model design, single-GPU pretraining, checkpoint recovery, evaluation, and
honest comparison against GPT-2 small.

The final version, SoloLLM v3, trains GPT-2-class decoder-only language models
from scratch on one RTX 3090 using a curated 10B-token dataset. The best 150M
model beats GPT-2 small overall on the fixed project evaluation suite. A smaller
123M ablation beats GPT-2 on most external checks, but does not beat it across
every metric.

## Why This Project

The goal was not to build a chat assistant. The goal was to build and evaluate a
real base language model pipeline with enough rigor that the results could be
audited:

- custom PyTorch decoder-only transformer,
- data curation and tokenized shard pipeline,
- resumable pretraining on consumer hardware,
- structured run metadata and metrics,
- fixed evaluation against GPT-2 small,
- documented failures, ablations, and final claims.

## Project Arc

V1 proved the basic pipeline worked: a custom GPT-style model, OpenWebText
tokenization, pretraining, fine-tuning, and a Streamlit demo.

V2 rebuilt the system into a stronger ML engineering project. It added RoPE,
RMSNorm, SwiGLU, tied embeddings, config-driven training, checkpoint metadata,
resume support, JSONL metrics, validation hooks, and CPU smoke tests. The best
91.65M-param v2 checkpoint reached held-out perplexity 25.56, close to GPT-2
small at 25.32, but external benchmarks showed GPT-2 was still much more robust.

V3 used that diagnosis directly. Instead of only training longer, it switched to
a broader 10B-token, 1024-context dataset with FineWeb-Edu, DCLM, FineWeb,
Wikipedia, and OpenWebText. It trained both a smaller-than-GPT-2 123M model and a
larger 150M model, then evaluated both against GPT-2 on the same fixed suite.

## Final Results

| Model | Params | Train tokens | Held-out PPL | WikiText-2 PPL | LAMBADA PPL | MC avg acc norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 124.44M | public | 25.32 | 45.32 | 40.62 | 41.05% |
| SoloLLM v3 123M | 123.55M | 9.80B | 25.64 | 41.87 | 36.28 | 42.46% |
| SoloLLM v3 150M | 151.87M | 10.00B | 24.90 | 41.18 | 35.35 | 42.71% |

The 150M model is the final best checkpoint. It beats GPT-2 small overall on the
fixed suite, including project held-out perplexity, WikiText-2 perplexity,
LAMBADA perplexity, LAMBADA continuation accuracy, and average multiple-choice
continuation scoring.

The 123M model is the smaller-model ablation. It is slightly smaller than GPT-2
small and wins most external checks, but loses project held-out perplexity and
some fixed-prompt generation diversity/repetition diagnostics. That means the
strict claim "smaller than GPT-2 and better across the board" is not proven.

## Technical Stack

- PyTorch transformer implementation
- GPT-2 tokenizer
- 1024-token packed training shards
- RTX 3090 training runs
- JSONL training metrics and resolved config snapshots
- Full eval suite covering held-out perplexity, WikiText-2, LAMBADA,
  multiple-choice continuation scoring, and fixed-prompt generation metrics

## What This Shows

SoloLLM is strongest as an ML systems case study. It shows the practical work
behind training a base model from scratch: building the dataset, running long
single-GPU jobs, recovering from interrupted training, designing fair evals,
reading failure signals, and changing the plan based on evidence.

The final honest claim:

> SoloLLM v3 trains GPT-2-class base LMs from scratch on one RTX 3090. The final
> 150M model beats GPT-2 small overall on a fixed evaluation suite, while a
> slightly smaller 123M model beats GPT-2 on most external benchmarks but does
> not fully clear the strict across-board smaller-than-GPT-2 bar.

## Key Artifacts

- Final 150M Hugging Face model: <https://huggingface.co/bmax16634/sologpt-v3-150m-base>
- 123M Hugging Face ablation: <https://huggingface.co/bmax16634/sologpt-v3-123m-base>
- Public completion demo: <https://huggingface.co/spaces/bmax16634/sologpt-v3-150m-demo>
- Final report: `docs/results/v3_final_gpt2_comparison.md`
- Artifact manifest: `docs/V3_ARTIFACT_MANIFEST.md`
- V3 plan and closeout: `docs/V3_PLAN.md`
- V3 dataset plan: `docs/V3_DATA_PLAN.md`
- V3 eval suite: `docs/V3_EVAL_SUITE.md`
- Final 150M local model: `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/final_model.pt`
- Final 123M local model: `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/final_model.pt`
