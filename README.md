# soloLLM

## Overview

`soloLLM` started out as a challenge to see I could train a base model from scratch on consumer hardware that could reach GPT-2-level performance. After a few model iterations I had already reached `GPT-2 small` level performance. At that point the project shifted to a more open small model research project. 

> **The refined goal for this project is to research how to build the best models from scratch and find what methods have the biggest impact on performance.**

| Model | Params | Tokens | Status | Hugging Face |
| --- | ---: | ---: | --- | --- |
| `SoloLLM-1.0-152M-Base` | 151.9M | 9.8B | Strongest completed SoloLLM 1.0 checkpoint | [model](https://huggingface.co/bmax16634/solollm-1.0-152m-base) |
| `SoloLLM-1.0-123M-Base` | 123.6M | 9.8B | Smaller-than-GPT-2 comparison checkpoint | [model](https://huggingface.co/bmax16634/solollm-1.0-123m-base) |
## Current Findings

`SoloLLM-1.0-152M-Base` is our strongest baseline model so far. This model has 151.9M parameters and was trained on 9.8B tokens. This makes `SoloLLM-1.0-152M-Base` slightly larger than `GPT-2 small`. However it beats `GPT-2 small` across our fixed base model [evaluation suite](docs/EVALUATION_SUITE.md). We also trained a slightly smaller model, `SoloLLM-1.0-123M-Base`. This model is smaller than `GPT-2 small`. It beat GPT-2 small on most tests in the suite but did not beat it on all of them.


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