# SoloLLM Resume Bullets

Date: July 22, 2026

Use the wording below depending on how much room the resume has. The strongest
current project claim is v4.2 150M: a from-scratch GPT-style base LM trained on
one RTX 3090 that beats GPT-2 small across the fixed project evaluation suite.
The honest caveat is that v4.2 150M is larger than GPT-2 small.

## One-Line Version

- Built SoloLLM, a from-scratch PyTorch GPT-style language-modeling system; trained a 151.9M-parameter v4.2 base LM on 9.8B tokens with one RTX 3090, beating GPT-2 small across a fixed quantitative eval suite.

## Technical Resume Version

- Built and evaluated SoloLLM, a from-scratch GPT-style LM stack in PyTorch with RoPE attention, RMSNorm, SwiGLU, tied embeddings, resumable sharded pretraining, checkpoint recovery, JSONL metrics, and fixed eval harnesses.
- Trained SoloLLM v4.2, a 151.9M-parameter base LM, from scratch on a 9.8B-token curated blend using one RTX 3090; improved over GPT-2 small on held-out perplexity, WikiText-2, LAMBADA, normalized multiple-choice average, and fixed-prompt repetition/diversity metrics.
- Ran controlled 123M/150M ablations against GPT-2 small and modern compact baselines including Pythia and SmolLM2; documented failure modes, scaling tradeoffs, and a v5 research plan focused on local small-model quality.

## Short Project Summary

SoloLLM is an end-to-end small-LM systems project: custom decoder-only model
implementation, dataset construction, long-running single-GPU pretraining,
recovery from interrupted runs, model comparison, and public documentation.

The best research checkpoint, SoloLLM v4.2 150M, trains from scratch on 9.8B
tokens and beats GPT-2 small across the fixed project suite. The 123M v4.2
ablation is slightly smaller than GPT-2 small and wins all fixed
multiple-choice checks, but does not beat GPT-2 across every metric.

## Numbers To Mention

| Result | Value |
| --- | --- |
| Best checkpoint | `SoloLLM v4.2 150M` |
| Parameters | `151.9M` |
| Train tokens | `9.8B` |
| Hardware | `1x RTX 3090` |
| Held-out PPL | `24.30` vs GPT-2 small `25.32` |
| WikiText-2 PPL | `41.75` vs GPT-2 small `45.32` |
| LAMBADA PPL | `34.06` vs GPT-2 small `40.62` |
| MC avg norm | `43.61%` vs GPT-2 small `41.05%` |
| Smaller-model ablation | `123.55M`, wins all fixed normalized MC checks |

## Interview Framing

The project is strongest as an ML systems case study, not an assistant-model
claim. The important engineering work was building the whole loop: data
mixtures, tokenized shards, training stability, resume/recovery, fair baselines,
full evals, and documented decisions when proxy results did not transfer.
