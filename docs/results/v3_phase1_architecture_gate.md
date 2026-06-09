# V3 Phase 1 Architecture Gate

Phase 1 tests whether a larger 1024-context v3 model is practical on the RTX 3090 before committing to longer training.

Date: June 9, 2026

Dataset:

- `/home/bmx/_projects/soloLLM/data/v3_pilot_1b_1024/chunks`
- train shards `0:116`
- val shards `117:118`
- sequence length `1024`

## Configs

| Config | Params | Layers | Width | Heads | MLP hidden | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v3-gpt2-scale-1024` | 123,551,232 | 12 | 768 | 12 | 2048 | 1024 |
| `v3-plus-150m-1024` | 151,868,928 | 16 | 768 | 12 | 2048 | 1024 |
| `v3-plus-180m-1024` | 180,186,624 | 20 | 768 | 12 | 2048 | 1024 |

Both use the existing `SoloGPT_v2` model class: RMSNorm, SwiGLU, RoPE, tied embeddings, no attention/MLP biases.

## Smoke Method

Each config ran:

- 20 optimizer steps,
- batch size 8,
- gradient accumulation 8,
- 1,310,720 training tokens,
- 1,048,576-token validation cap,
- CUDA fp16 autocast.

## Results

| Config | Peak VRAM | Last-10 train tok/s | Summary tok/s | Final train loss | Val loss | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v3-gpt2-scale-1024` | 11.75GB | 39.96k | 29.94k | 10.2780 | 10.2002 | 43.8s |
| `v3-plus-150m-1024` | 13.76GB | 32.36k | 24.26k | 10.3199 | 10.1832 | 54.0s |
| `v3-plus-180m-1024` | 15.78GB | 27.20k | 20.35k | 10.3600 | 10.2267 | 64.4s |

## Read

The 150M and 180M configs are both viable on the RTX 3090:

- 150M uses about 2.0GB more peak VRAM than 123M and is about 19% slower by training-step throughput.
- 180M uses about 4.0GB more peak VRAM than 123M and is about 32% slower by training-step throughput.
- 180M still stays well below the 24GB card limit and remains above the practical speed threshold for a pilot.

The 20-step losses are not meaningful for model quality, but the fit/speed result is meaningful. The 180M model is feasible, but it should earn the final run through a 50M-token sanity comparison because it costs meaningful throughput.

## Recommendation

Proceed to Phase 2 with a short 50M-token architecture sanity comparison:

1. train `v3-plus-150m-1024` for 50M tokens,
2. train `v3-plus-180m-1024` for 50M tokens,
3. compare loss curve, validation PPL, tokens/sec, and peak VRAM,
4. choose the 180M model only if it gives a clearly better learning curve or validation result.

Keep `v3-gpt2-scale-1024` as the safe fallback, but the real v3 decision is now 150M versus 180M. If 180M does not show a clear advantage at 50M, use 150M for the 300M v3 pilot and spend the saved compute on more tokens.
