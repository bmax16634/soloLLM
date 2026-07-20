# V5 Baseline Board

Date: July 20, 2026

## Status

This is the first v5 baseline board. It reuses completed eval artifacts
from v3/v4 instead of launching new training.
It includes quality metrics plus the first local inference benchmark.
Pythia 410M and TinyLlama 1.1B are now included.
Quantization is blocked locally by a bitsandbytes CUDA library mismatch.

The cross-tokenizer language-model metric to read is BPB
(bits per byte), not raw token perplexity. PPL is still shown for
continuity, but it is tokenizer-dependent.

## Headline Board

| Model | Params | WikiText BPB | LAMBADA BPB | LAMBADA tok acc | LAMBADA word acc | MC avg norm | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
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
| TinyLlama 1.1B | 1100.05M | 0.9630 | 1.1828 | 80.07% | 58.80% | 51.61% | Stretch local baseline; quality target, not same-size. |

## Multiple-Choice Detail

| Model | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SoloLLM v4.2 150M | 32.99% | 65.50% | 45.09% | 23.08% | 51.38% | 43.61% |
| SoloLLM v4.2 123M | 31.92% | 65.00% | 44.21% | 24.08% | 50.43% | 43.13% |
| SoloLLM v3 150M | 30.08% | 62.70% | 44.56% | 25.08% | 51.14% | 42.71% |
| GPT-2 small | 29.53% | 63.60% | 40.35% | 22.07% | 49.72% | 41.05% |
| DistilGPT2 | 26.33% | 60.20% | 35.79% | 20.40% | 49.49% | 38.44% |
| Pythia 70M | 26.39% | 59.60% | 34.39% | 22.41% | 49.57% | 38.47% |
| Pythia 160M | 29.72% | 62.00% | 38.25% | 22.41% | 49.80% | 40.43% |
| Pythia 410M | 39.36% | 68.40% | 44.21% | 23.08% | 50.83% | 45.18% |
| SmolLM 135M | 40.32% | 65.60% | 54.91% | 30.43% | 51.07% | 48.47% |
| SmolLM2 135M | 41.09% | 67.10% | 56.67% | 32.44% | 51.62% | 49.78% |
| SmolLM2 360M | 53.42% | 70.60% | 68.42% | 38.13% | 53.28% | 56.77% |
| TinyLlama 1.1B | 56.83% | 72.60% | 46.84% | 29.43% | 52.33% | 51.61% |

## Interpretation

- SoloLLM v4.2 150M remains the strongest SoloLLM checkpoint.
- SoloLLM v4.2 123M is competitive with GPT-2 and Pythia 160M, but
  SmolLM2 135M is the main same-scale modern target.
- The largest same-scale gap is still modern-small-model quality:
  SmolLM2 135M and especially SmolLM2 360M are well ahead on MC and
  LAMBADA exact-word accuracy.
- Pythia 410M and TinyLlama 1.1B confirm the expected scaling curve:
  substantially larger modern baselines are far ahead on BPB and
  continuation accuracy.
- The board is complete enough to choose v5 direction. The remaining
  engineering gap is SoloLLM KV-cache inference; quantized benchmarking
  needs the local bitsandbytes CUDA issue fixed first.

## Local Inference

Batch size `1`, greedy generation, FP16, three fixed prompts. SoloLLM
uses the current manual generation path without KV cache; Hugging Face
baselines use `transformers.generate` with cache enabled.

| Model | Params | Generation path | Tok/s | Peak VRAM |
| --- | ---: | --- | ---: | ---: |
| SoloLLM v4.2 150M | 151.87M | manual_no_kv_cache | 122.21 | 0.34GB |
| SoloLLM v4.2 123M | 123.55M | manual_no_kv_cache | 162.12 | 0.28GB |
| GPT-2 small | 124.44M | transformers_generate_use_cache | 231.20 | 0.27GB |
| Pythia 160M | 162.32M | transformers_generate_use_cache | 211.71 | 0.34GB |
| Pythia 410M | 405.33M | transformers_generate_use_cache | 105.00 | 0.83GB |
| SmolLM2 135M | 134.52M | transformers_generate_use_cache | 62.20 | 0.28GB |
| SmolLM2 360M | 361.82M | transformers_generate_use_cache | 55.51 | 0.75GB |
| TinyLlama 1.1B | 1100.05M | transformers_generate_use_cache | 80.18 | 2.21GB |

Inference report: `docs/results/v5_inference_benchmark.md`

## Pending Board Items

| Item | Status | Reason |
| --- | --- | --- |
| Pythia 410M | complete | Included in external quality and multiple-choice board. |
| TinyLlama 1.1B | complete | Included as stretch local quality baseline. |
| Inference tokens/sec | v0 complete | Covered for the selected practical baseline set. |
| Peak inference VRAM | v0 complete | Covered for the selected practical baseline set. |
| Quantized quality/speed | blocked | `bitsandbytes` fails to load CUDA backend locally: missing `libnvJitLink.so.13`. |
| SoloLLM KV-cache inference | pending | Requires implementation work; current SoloLLM speed uses manual no-cache generation. |

## Source Artifacts

- `modern_baselines_v3_150m`
  - `outputs/eval_suites/modern_baselines_v3_150m/external_modern_baselines.json`
  - `outputs/eval_suites/modern_baselines_v3_150m/multiple_choice_modern_baselines.json`
- `v4_2_150m_gpt2_suite`
  - `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_150m_gpt2_full_suite/external_full_candidate_gpt2.json`
  - `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_150m_gpt2_full_suite/multiple_choice_full_candidate_gpt2.json`
- `v4_2_123m_gpt2_suite`
  - `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_123m_gpt2_full_suite/external_full_candidate_gpt2.json`
  - `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_123m_gpt2_full_suite/multiple_choice_full_candidate_gpt2.json`
- `v5_missing_baselines`
  - `outputs/eval_suites/v5_missing_baselines/external_missing_baselines.json`
  - `outputs/eval_suites/v5_missing_baselines/multiple_choice_missing_baselines.json`

Machine-readable board: `outputs/eval_suites/v5_baseline_board/baseline_board.json`
