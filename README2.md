# soloLLM

`soloLLM` is a research project where I research ways to improve and build the best possible LLM from scratch. The project started with the goal to beat and an older larger GPT-2 model with a more modern architecture and less parameters from a model we build from scratch . After a few iterations we had already reach GPT-2 level performance. And the project has shifted from trying to beat a specific model to more of an open research project. The refined goal for this project is to build the best models from scratch and experiment with research.

## Current Findings

`soloLLM v4.2` is the strongest quantitative checkpoint baseline model so far. This model has 151.9M parameters and was trained on 9.8B tokens. This makes `soloLLM v4.2` slight larger than GPT-2 small. However `soloLLM v4.2` beats GPT-2 small across our whole [evaluation suite](docs/EVALUATION_SUITE.md). We also trained a slightly smaller 123M parameter version. This model beat GPT on most tests in the suite but did not beat it on all of them.

## Other Findings

## License

MIT License. See [LICENSE](LICENSE).

## Author

Benjamin Maxwell

- GitHub: <https://github.com/bmax16634>
- LinkedIn: <https://www.linkedin.com/in/benjamin-maxwell-95a9342b0/>
- Hugging Face: <https://huggingface.co/bmax16634>