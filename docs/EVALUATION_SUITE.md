# Evaluation Suite

The goal is to use a wide variety of tests to understand how a model is performing. I have split the evaluation suite into sepertate

The main suite is split into smaller suites for different model types.

A model should not be judged from one number. A small language model can look good on perplexity and still produce repetitive text, or look good on multiple-choice tasks while being worse at broad next-token prediction. The suite combines several checks so each checkpoint has to improve across more than one kind of behavior.

| Suite | Model type | Status | Purpose |
| --- | --- | --- | --- |
| Base Model Suite | Raw pretrained/base LMs | Active | Measures language modeling, continuation, multiple-choice behavior, and generation quality before instruction tuning. |
| Instruction / Chat Suite | Instruction-tuned or chat models | Future | Measures whether a tuned model follows instructions, handles dialogue, and gives useful answers. |
| Local Deployment Suite | Base or chat models | Active / expanding | Measures speed, VRAM, quantized behavior, and whether the model is practical on local hardware. |

## Base Model Suite

This is the active suite for current SoloLLM checkpoints. It is split into three levels:

stronger modern baselines.

- Core tests: always run for major checkpoints.
- Extended tests: run for serious comparison checkpoints.
- Stress/research tests: run when diagnosing weaknesses or comparing against

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

These tests add broader coverage beyond the original GPT-2 comparison suite.

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

These tests are useful for diagnosis, but they should not be the only source of headline claims.

| Area | Test | What it checks | Direction |
| --- | --- | --- | --- |
| Hard reasoning | BBH-lite | Harder symbolic, logic, and reasoning-style tasks. | Higher is better |
| Hard math | MATH or arithmetic stress tests | More difficult math behavior. | Higher is better |
| Code, if code data is added | HumanEval / MBPP-style prompts | Whether the model can produce simple working code. | Higher pass rate is better |
| Contamination | Eval/train n-gram overlap checks | Whether benchmark text may be present in training data. | Lower overlap is better |
| Memorization | Canary or exact-regurgitation checks | Whether the model memorizes inserted or sensitive sequences. | Lower memorization is better |
| Toxicity / safety diagnostics | Toxic prompt continuation checks | Whether the base model tends to continue unsafe or toxic text. | Lower toxicity is better |
| Long-context behavior | Length-binned prompt tests | Whether quality degrades as prompts approach context length. | Less degradation is better |

## Instruction / Chat Suite

This suite is for future instruction-tuned or chat versions of SoloLLM. It should not replace the base-model suite because chat models and base models are trying to do different jobs.

| Area | Test type | What it checks | Status |
| --- | --- | --- | --- |
| Instruction following | Single-turn instruction prompts | Whether the model follows direct user requests. | Future |
| Multi-turn dialogue | Conversation prompts | Whether the model keeps context and answers coherently across turns. | Future |
| Helpfulness | Preference or rubric scoring | Whether answers are useful, complete, and on task. | Future |
| Formatting compliance | Structured-output prompts | Whether the model follows requested formats like lists, JSON, or short answers. | Future |
| Factual QA | Closed-book question answering | Whether answers are factually plausible and not hallucinated. | Future |
| Safety/refusal behavior | Unsafe or sensitive prompts | Whether the model refuses or redirects when appropriate. | Future |
| Coding prompts | Small programming tasks | Whether the model can produce correct simple code, if code training is added. | Optional future |
| Pairwise comparison | Compare against small chat baselines | Whether users or judges prefer SoloLLM outputs over baseline outputs. | Future |

## Local Deployment Suite

This suite measures whether a model is practical to run locally.

| Area | Test | What it checks | Direction |
| --- | --- | --- | --- |
| Inference speed | Tokens per second | How fast the model generates on local hardware. | Higher is better |
| Memory use | Peak VRAM / RAM | Whether the model fits comfortably on available hardware. | Lower is better |
| Quantization | 8-bit / 4-bit behavior | Whether quality holds after compression. | Lower quality loss is better |
| Context use | Prompt length stress tests | Whether longer prompts remain stable and usable. | Higher usable length is better |
| Generation stability | Fixed local prompts | Whether the model avoids loops, collapse, or broken formatting. | Fewer failures is better |

## Why These Tests

The active suite is meant for base language models, not instruction-tuned assistants. It focuses on whether a model trained from scratch is learning broad language modeling behavior and whether it can compete with known small-model baselines.

The fixed GPT-2 comparison uses the same tests across runs so the target does not move after seeing results. Later v5 work adds modern small-model baselines such as Pythia, SmolLM, and SmolLM2-style comparisons.

## What The Suite Can Support

A checkpoint can make a strong quantitative claim when it improves across:

claim.

- held-out project perplexity,
- external perplexity,
- LAMBADA continuation metrics,
- most or all length-normalized multiple-choice checks,
- repetition and diversity diagnostics,
- local inference and memory targets when deployment quality is part of the

For example, SoloLLM v4.2 150M beats GPT-2 small across the fixed project suite. That supports the claim that this checkpoint is a stronger base LM than GPT-2 small on these measurements.

The strongest honest claim is:

> This model is better than the baseline across our fixed base-model evaluation

> suite.

The suite should not be used to claim that a model is 100% better at everything. No finite benchmark suite can prove that.

## What The Suite Cannot Prove

This suite does not prove that a model is:

- a better chatbot,
- a better coding model,
- more factual,
- safer,
- better at long-context use,
- better than modern instruction-tuned models,
- better than much larger models such as GPT-3-class systems.

It also does not remove all risk of benchmark overfitting. It reduces that risk by using multiple fixed tests, outside corpora, and baselines, but the final answer still needs broader testing as the project moves into v5.

## Current Interpretation

SoloLLM v4.2 150M is the strongest completed checkpoint so far. It beats GPT-2 small across the fixed project suite, but it is larger than GPT-2 small.

The SoloLLM v4.2 123M ablation is slightly smaller than GPT-2 small and wins the fixed normalized multiple-choice checks, but it does not beat GPT-2 small across every metric. That means the strict smaller-than-GPT-2 claim remains unresolved.

The v5 direction is broader: compare against modern small-model baselines and study which architecture, data, and training choices produce the best model that can still run locally.

## Implementation

The suite is implemented by:

- `eval/v3_eval_suite.py`
- `eval/eval.py`
- `eval/external_benchmarks.py`
- `eval/multiple_choice_benchmarks.py`
- `eval/generate_samples.py`
- `eval/generation_metrics.py`

Detailed result reports are kept locally until they are promoted into the public repo.