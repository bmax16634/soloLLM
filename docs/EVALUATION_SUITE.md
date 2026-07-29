# Evaluation Suites

The goal of our evaluation suites is to use a wide variety of tests to understand how a model is performing. Each of suite is split in to smaller groups for faster run times. All full suite test isnt always  required for each run.
| Suite | Model Types |
| --- | --- |
| Base Model Suite | Base models, pretrained models |
## Base Model Suite

### Core Base Tests
| Area | Test | What it checks | Direction |
| --- | --- | --- | --- |
| Project held-out perplexity | Held-out OpenWebText-style validation split | Whether the model predicts text from the project's own training distribution. | Lower is better |
| External perplexity | WikiText-2 | General next-token prediction on a standard outside corpus. | Lower is better |
| Completion perplexity | LAMBADA perplexity | Whether the model handles longer context and story-like continuation. | Lower is better |
| Completion accuracy | LAMBADA last-token / last-word accuracy | Whether the model can choose the expected final continuation, not just assign reasonable probability. | Higher is better |
| Multiple-choice reasoning | HellaSwag | Commonsense continuation selection. | Higher is better |
| Multiple-choice reasoning | PIQA | Physical commonsense selection. | Higher is better |
| Multiple-choice reasoning | ARC Easy | Basic science question answering. | Higher is better |
| Multiple-choice reasoning | ARC Challenge | Harder science question answering. | Higher is better |
| Multiple-choice reasoning | WinoGrande | Pronoun/coreference-style commonsense. | Higher is better |
| Generation diagnostics | Fixed prompts, repetition, diversity | Whether generations are usable or collapse into repeated text. | Lower repetition, higher diversity |
### Extended Base Tests
| Area | Test | What it checks | Direction |
| --- | --- | --- | --- |
| External perplexity | C4-style heldout | General web-text prediction outside the project split. | Lower is better |
| External perplexity | FineWeb-Edu heldout | Educational/web-document prediction outside the training holdout. | Lower is better |
| External perplexity | Penn Treebank or another small standard corpus | Whether gains transfer to a compact older benchmark. | Lower is better |
| Knowledge QA | MMLU or MMLU-lite | Broad academic and factual multiple-choice knowledge. | Higher is better |
| Knowledge QA | SciQ | Science question answering. | Higher is better |
| Knowledge QA | OpenBookQA | Elementary science with commonsense and fact use. | Higher is better |
| Commonsense QA | CommonsenseQA | General commonsense question answering. | Higher is better |
| Commonsense QA | COPA | Cause/effect commonsense reasoning. | Higher is better |
| Reading / boolean QA | BoolQ | Short passage/question yes-no reasoning. | Higher is better |
| Math / symbolic reasoning | GSM8K small subset | Basic grade-school math reasoning. | Higher is better |
| Calibration | Multiple-choice confidence / margin metrics | Whether the model is confidently wrong or appropriately uncertain. | Better calibration is better |
| Robustness | Few-shot and prompt-format variants | Whether results survive small prompt changes. | Smaller score swings are better |
### Stress / Research Tests
| Area | Test | What it checks | Direction |
| --- | --- | --- | --- |
| Hard reasoning | BBH-lite | Harder symbolic, logic, and reasoning-style tasks. | Higher is better |
| Hard math | MATH or arithmetic stress tests | More difficult math behavior. | Higher is better |
| Code, if code data is added | HumanEval / MBPP-style prompts | Whether the model can produce simple working code. | Higher pass rate is better |
| Contamination | Eval/train n-gram overlap checks | Whether benchmark text may be present in training data. | Lower overlap is better |
| Memorization | Canary or exact-regurgitation checks | Whether the model memorizes inserted or sensitive sequences. | Lower memorization is better |
| Toxicity / safety diagnostics | Toxic prompt continuation checks | Whether the base model tends to continue unsafe or toxic text. | Lower toxicity is better |
| Long-context behavior | Length-binned prompt tests | Whether quality degrades as prompts approach context length. | Less degradation is better |
