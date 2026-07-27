# Results

This page is generated from structured leaderboard data in `leaderboard/`.
Evaluation details are described in [docs/EVALUATION_SUITE.md](EVALUATION_SUITE.md).

## Current Read

- SoloLLM v4.2 150M is the strongest completed SoloLLM checkpoint.
- It beats GPT-2 small across the fixed project suite, but it is larger than GPT-2 small.
- SoloLLM v4.2 123M is slightly smaller than GPT-2 small and wins the fixed normalized multiple-choice suite, but not every metric.
- V5 moves the project toward modern small-model baselines such as SmolLM2 rather than repeatedly chasing GPT-2.

## Base Core Leaderboard

| Model | Params | Tokens | Held-out PPL | WikiText-2 PPL | LAMBADA PPL | LAMBADA tok | LAMBADA word | MC avg norm | Read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 small | 124.44M | public | 25.32 | 45.32 | 40.62 | 46.67% | 32.60% | 41.05% | Historical GPT-2-class reference baseline. |
| SoloLLM v2 5.60B | 91.65M | 5.60B | 25.56 | 84.81 | 63.03 | - | 29.09% | 38.97% | v2 diagnostic baseline. |
| SoloLLM v3 123M | 123.55M | 9.80B | 25.64 | 41.87 | 36.28 | 46.67% | 32.80% | 42.46% | Smaller-than-GPT-2 v3 ablation. |
| SoloLLM v3 150M | 151.87M | 10.00B | 24.90 | 41.18 | 35.35 | 47.43% | 33.07% | 42.71% | Public final v3 checkpoint. |
| SoloLLM v4 reasoning 150M | 151.87M | 9.80B | 27.38 | 43.59 | 38.46 | 47.09% | 32.91% | 43.52% | Reasoning-heavy v4 ablation; MC/repetition improved, broad PPL regressed. |
| SoloLLM v4.2 123M | 123.55M | 9.80B | 25.56 | 43.75 | 35.45 | 46.30% | 31.69% | 43.13% | Smaller-than-GPT-2 ablation; wins fixed MC suite, not every metric. |
| SoloLLM v4.2 150M | 151.87M | 9.80B | 24.30 | 41.75 | 34.06 | 47.27% | 33.05% | 43.61% | Current best SoloLLM quantitative checkpoint. |

## Modern Baseline Board

| Model | Params | WikiText BPB | LAMBADA BPB | LAMBADA tok | LAMBADA word | MC avg norm | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SoloLLM v4.2 150M | 151.87M | 1.1953 | 1.3109 | 47.27% | 33.05% | 43.61% | Current best SoloLLM quantitative checkpoint. |
| SoloLLM v4.2 123M | 123.55M | 1.2103 | 1.3258 | 46.30% | 31.69% | 43.13% | Smaller-than-GPT-2 ablation; wins fixed MC suite, not every metric. |
| SoloLLM v3 150M | 151.87M | 1.1909 | 1.3247 | 47.43% | 33.07% | 42.71% | Public final v3 checkpoint. |
| GPT-2 small | 124.44M | 1.2216 | 1.3764 | 46.67% | 32.60% | 41.05% | Historical GPT-2-class reference baseline. |
| DistilGPT2 | 81.91M | 1.4104 | 1.5041 | 40.87% | 24.98% | 38.44% | Compressed GPT-2-family reference. |
| Pythia 70M | 70.43M | inf | 6.7967 | 41.88% | 19.08% | 38.47% | Controlled open baseline below SoloLLM scale. |
| Pythia 160M | 162.32M | 2.0182 | 3.0843 | 52.36% | 34.33% | 40.43% | Controlled open baseline near SoloLLM 150M scale. |
| Pythia 410M | 405.33M | 0.9791 | 1.2337 | 66.02% | 52.51% | 45.18% | Larger controlled baseline; useful direction check, not same-scale. |
| SmolLM 135M | 134.52M | 1.1849 | 1.3489 | 54.47% | 37.71% | 48.47% | Modern compact baseline. |
| SmolLM2 135M | 134.52M | 1.1071 | 1.2616 | 58.33% | 42.67% | 49.78% | Primary modern same-scale target. |
| SmolLM2 360M | 361.82M | 1.0251 | 1.2004 | 66.49% | 53.25% | 56.77% | Modern compact upper-bound target. |
| TinyLlama 1.1B | 1.10B | 0.9630 | 1.1828 | 80.07% | 58.80% | 51.61% | Stretch local baseline; quality target, not same-size. |

## Multiple-Choice Detail

| Model | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande | Average |
| --- | --- | --- | --- | --- | --- | --- |
| SoloLLM v4.2 150M | 32.99% | 65.50% | 45.09% | 23.08% | 51.38% | 43.61% |
| SoloLLM v4.2 123M | 31.92% | 65.00% | 44.21% | 24.08% | 50.43% | 43.13% |
| SoloLLM v3 150M | 30.08% | 62.70% | 44.56% | 25.08% | 51.14% | 42.71% |
| SoloLLM v3 123M | 29.85% | 63.40% | 44.04% | 24.08% | 50.91% | 42.46% |
| GPT-2 small | 29.53% | 63.60% | 40.35% | 22.07% | 49.72% | 41.05% |
| DistilGPT2 | 26.33% | 60.20% | 35.79% | 20.40% | 49.49% | 38.44% |
| Pythia 70M | 26.39% | 59.60% | 34.39% | 22.41% | 49.57% | 38.47% |
| Pythia 160M | 29.72% | 62.00% | 38.25% | 22.41% | 49.80% | 40.43% |
| Pythia 410M | 39.36% | 68.40% | 44.21% | 23.08% | 50.83% | 45.18% |
| SmolLM 135M | 40.32% | 65.60% | 54.91% | 30.43% | 51.07% | 48.47% |
| SmolLM2 135M | 41.09% | 67.10% | 56.67% | 32.44% | 51.62% | 49.78% |
| SmolLM2 360M | 53.42% | 70.60% | 68.42% | 38.13% | 53.28% | 56.77% |
| TinyLlama 1.1B | 56.83% | 72.60% | 46.84% | 29.43% | 52.33% | 51.61% |

## Generation Diagnostics

| Model | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |
| --- | --- | --- | --- | --- | --- |
| GPT-2 small | 0.3869 | 0.7836 | 0.1715 | 0.1120 | 2 |
| SoloLLM v3 123M | 0.3918 | 0.7734 | 0.1805 | 0.0906 | 1 |
| SoloLLM v3 150M | 0.4308 | 0.8053 | 0.1723 | 0.1006 | 2 |
| SoloLLM v4.2 123M | 0.4535 | 0.8453 | 0.1347 | 0.0618 | 2 |
| SoloLLM v4.2 150M | 0.4000 | 0.8109 | 0.1534 | 0.0836 | 2 |

## Local Inference

Batch size `1`, greedy generation, FP16, three fixed prompts.

| Model | Params | Generation path | Tok/s | Peak VRAM |
| --- | --- | --- | --- | --- |
| SoloLLM v4.2 150M | 151.87M | manual_preallocated_kv_cache | 99.44 | 0.32GB |
| SoloLLM v4.2 123M | 123.55M | manual_preallocated_kv_cache | 142.01 | 0.26GB |
| GPT-2 small | 124.44M | transformers_generate_use_cache | 224.94 | 0.27GB |
| Pythia 160M | 162.32M | transformers_generate_use_cache | 210.66 | 0.34GB |
| Pythia 410M | 405.33M | transformers_generate_use_cache | 112.19 | 0.83GB |
| SmolLM2 135M | 134.52M | transformers_generate_use_cache | 62.74 | 0.28GB |
| SmolLM2 360M | 361.82M | transformers_generate_use_cache | 56.05 | 0.75GB |
| TinyLlama 1.1B | 1.10B | transformers_generate_use_cache | 83.32 | 2.21GB |

## V5 Proxy Gate

| Proxy | Params | Tokens | Val PPL | Train loss | WikiText BPB | LAMBADA BPB | LAMBADA tok | LAMBADA word | MC avg norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SoloLLM v4.2 57M/300M proxy | 57.12M | 300M | 54.98 | 3.9285 | 1.5864 | 1.5484 | 24.20% | 6.04% | 37.05% |
| SoloLLM v5 smol-quality 57M/1B proxy | 57.12M | 1.00B | 32.26 | 3.4911 | 1.4286 | 1.4865 | 33.13% | 16.40% | 38.66% |

## Claim Standard

A leaderboard win means the model is better across the fixed suite shown here.
It does not prove the model is better at every possible language task.

The strongest current SoloLLM claim is that v4.2 150M beats GPT-2 small
across the fixed project suite, while the smaller v4.2 123M checkpoint
wins the normalized multiple-choice suite but not every metric.
