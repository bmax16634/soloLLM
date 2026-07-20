# V5 Inference Benchmark

Date: July 20, 2026

## Method

Batch size `1`, greedy generation, three fixed prompts, and `80` measured
new tokens per prompt by default. SoloLLM checkpoints use the current
manual generation loop without KV cache. Hugging Face baselines use
`transformers.generate(..., use_cache=True)`. This measures practical
local usability of the current implementations, not pure architecture
throughput under identical kernels.

## Results

| Model | Params | Path | DType | Tok/s | Peak VRAM | Load sec |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| SoloLLM v4.2 150M | 151.87M | manual_no_kv_cache | float16 | 122.21 | 0.34GB | 2.35 |
| SoloLLM v4.2 123M | 123.55M | manual_no_kv_cache | float16 | 162.12 | 0.28GB | 1.90 |
| GPT-2 small | 124.44M | transformers_generate_use_cache | float16 | 231.20 | 0.27GB | 1.10 |
| Pythia 160M | 162.32M | transformers_generate_use_cache | float16 | 211.71 | 0.34GB | 1.12 |
| Pythia 410M | 405.33M | transformers_generate_use_cache | float16 | 105.00 | 0.83GB | 1.10 |
| SmolLM2 135M | 134.52M | transformers_generate_use_cache | float16 | 62.20 | 0.28GB | 0.92 |
| SmolLM2 360M | 361.82M | transformers_generate_use_cache | float16 | 55.51 | 0.75GB | 1.03 |
| TinyLlama 1.1B | 1100.05M | transformers_generate_use_cache | float16 | 80.18 | 2.21GB | 1.56 |

## Read

- SoloLLM numbers reflect the current no-KV-cache generation path and are
  expected to understate what the architecture could do with a cached
  inference implementation.
- Hugging Face model numbers reflect the optimized local path available
  through `transformers.generate`.
- Use this board for practical local planning: model size, memory headroom,
  and whether implementing a SoloLLM KV cache should be a v5 priority.
- Quantized benchmarking is not included yet because local bitsandbytes
  CUDA backend loading currently fails with missing `libnvJitLink.so.13`.

Raw JSON: `outputs/eval_suites/v5_inference_benchmark/inference_benchmark.json`
