# SoloLLM

## Overview

SoloLLM started out as a challenge to see if I could train a base model from scratch on consumer hardware that could reach GPT-2-level performance. After a few model iterations, it had already reached GPT-2 small level performance. At that point the project shifted to a more open small model research project.

> **The refined goal for this project is to use SoloLLM as a research platform for understanding how to build stronger small language models from scratch with hardware constraints**

| Model | Params | Tokens | Status | Hugging Face |
| --- | --- | --- | --- | --- |
| `SoloLLM-1.0-152M-Base` | 151.9M | 9.8B | Strongest completed SoloLLM 1.0 checkpoint | [model](https://huggingface.co/bmax16634/solollm-1.0-152m-base) |
| `SoloLLM-1.0-123M-Base` | 123.6M | 9.8B | Smaller-than-GPT-2 comparison checkpoint | [model](https://huggingface.co/bmax16634/solollm-1.0-123m-base) |

## Current Findings

- `SoloLLM-1.0-152M-Base` is the strongest checkpoint so far. It beats GPT-2 small across the fixed base-model evaluation suite, but it is slightly larger than GPT-2 small.
- `SoloLLM-1.0-123M-Base` is slightly smaller than GPT-2 small. It wins the normalized multiple-choice suite and several external checks, but it does not beat GPT-2 small across every metric.
-  Data mix seemed to have the largest impact on model performance. A reasoning-heavy mix improved some multiple-choice and repetition metrics, but hurt broad language-modeling perplexity. A broader blend with dropout `0.0` produced the best overall result.


## Documents

- [Evaluation Suite](docs/EVALUATION_SUITE.md)
- [Results Summary](docs/RESULTS.md)
---
Benjamin Maxwell

- [GitHub](https://github.com/bmax16634)
- [LinkedIn](https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/)
- [Hugging Face](https://huggingface.co/bmax16634)
---
[MIT License](LICENSE)
