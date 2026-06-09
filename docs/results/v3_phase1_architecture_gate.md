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

## Read

The 150M config is viable on the RTX 3090:

- it uses only about 2.0GB more peak VRAM than the 123M config,
- it stays far below the 24GB card limit,
- it is about 19% slower by training-step throughput,
- it has enough memory headroom for a real pilot.

The 20-step losses are not meaningful for model quality, but the fit/speed result is meaningful. The 150M model should be treated as the main v3 candidate unless the 50M sanity run shows worse learning behavior than the 123M baseline.

## Recommendation

Proceed to Phase 2 with a short 50M-token architecture sanity comparison:

1. train `v3-gpt2-scale-1024` for 50M tokens,
2. train `v3-plus-150m-1024` for 50M tokens,
3. compare loss curve, validation PPL, tokens/sec, and peak VRAM,
4. choose the 150M model if it is not clearly worse per unit compute.

If the 150M model remains healthy at 50M, use it for the 300M v3 pilot.
