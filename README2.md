# soloLLM

## Overview

`soloLLM` is a research project where I research ways to improve and build the best possible LLM from scratch. The project started with the goal to see if I could beat and an older larger GPT-2 model with a more modern architecture and less parameters from a model I build from scratch . After a few iterations I had already reached GPT-2 level performance. And the project has sense shifted from trying to beat a specific model to more of an open research project. The refined goal for this project is to build the best models from scratch and experiment with research.

## Current Findings

`soloLLM v4.2 152M` is the strongest checkpoint baseline model so far. This model has 151.9M parameters and was trained on 9.8B tokens. This makes `soloLLM v4.2 152M` slightly larger than `GPT-2 small`. However it beats `GPT-2 small` across our whole [evaluation suite](docs/EVALUATION_SUITE.md). We also trained a slightly smaller model `soloLLM v4.2 123M`. This model is smaller than `GPT-2 small` This model beat GPT on most tests in the suite but did not beat it on all of them.

## Documents

- [Results Summary](docs/RESULTS.md)
- [Evaluation Suite](docs/EVALUATION_SUITE.md)

## License

[MIT Licence](/home/bmx/ariya-data/projects/soloLLM/LICENSE)

## Author

Benjamin Maxwell

- [GitHub](https://github.com/bmax16634)
- [LinkedIn](https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/)
- [Hugging Face](https://huggingface.co/bmax16634)