# V5 Inference Benchmark

Date: July 21, 2026

## Method

Batch size `1`, greedy generation, three fixed prompts, and `80` measured
new tokens per prompt by default. SoloLLM checkpoints use the local
manual generation loop with KV cache. Hugging Face baselines use
`transformers.generate(..., use_cache=True)`. This measures practical
local usability of the current implementations, not pure architecture
throughput under identical kernels.

## Results

| Model | Params | Path | DType | Tok/s | Peak VRAM | Load sec |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| SoloLLM v4.2 150M | 151.87M | manual_preallocated_kv_cache | float16 | 99.44 | 0.32GB | 2.50 |
| SoloLLM v4.2 123M | 123.55M | manual_preallocated_kv_cache | float16 | 142.01 | 0.26GB | 1.90 |
| GPT-2 small | 124.44M | transformers_generate_use_cache | float16 | 224.94 | 0.27GB | 1.19 |
| Pythia 160M | 162.32M | transformers_generate_use_cache | float16 | 210.66 | 0.34GB | 0.98 |
| Pythia 410M | 405.33M | transformers_generate_use_cache | float16 | 112.19 | 0.83GB | 1.03 |
| SmolLM2 135M | 134.52M | transformers_generate_use_cache | float16 | 62.74 | 0.28GB | 0.92 |
| SmolLM2 360M | 361.82M | transformers_generate_use_cache | float16 | 56.05 | 0.75GB | 0.94 |
| TinyLlama 1.1B | 1100.05M | transformers_generate_use_cache | float16 | 83.32 | 2.21GB | 1.39 |

## Read

- SoloLLM numbers reflect the current local preallocated KV-cache generation path.
- The SoloLLM path is correctness-oriented and still uses a manual Python
  token loop, so it is not directly equivalent to optimized HF generation kernels.
- Hugging Face model numbers reflect the optimized local path available
  through `transformers.generate`.
- Use this board for practical local planning: model size, memory headroom,
  and whether optimizing SoloLLM generation should be a v5 priority.
- Quantized benchmarking is not included yet because local bitsandbytes
  CUDA backend loading currently fails with missing `libnvJitLink.so.13`.

Raw JSON: `outputs/eval_suites/v5_inference_benchmark/inference_benchmark.json`
