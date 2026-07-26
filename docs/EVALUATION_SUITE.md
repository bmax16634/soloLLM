# Evaluation Suite

This document explains what SoloLLM is tested on, why those tests were chosen,
and what the results can and cannot prove.

The goal is to avoid judging the model from one number. A small language model
can look good on perplexity and still produce repetitive text, or look good on
multiple-choice tasks while being worse at broad next-token prediction. The
suite combines several checks so each checkpoint has to improve across more
than one kind of behavior.

## What We Test

| Area | Test | What it checks |
| --- | --- | --- |
| Project held-out perplexity | Held-out OpenWebText-style validation split | Whether the model predicts text from the project's own training distribution. |
| External perplexity | WikiText-2 | General next-token prediction on a standard outside corpus. |
| Completion perplexity | LAMBADA perplexity | Whether the model handles longer context and story-like continuation. |
| Completion accuracy | LAMBADA last-token / last-word accuracy | Whether the model can choose the expected final continuation, not just assign reasonable probability. |
| Multiple-choice reasoning | HellaSwag | Commonsense continuation selection. |
| Multiple-choice reasoning | PIQA | Physical commonsense selection. |
| Multiple-choice reasoning | ARC Easy | Basic science question answering. |
| Multiple-choice reasoning | ARC Challenge | Harder science question answering. |
| Multiple-choice reasoning | WinoGrande | Pronoun/coreference-style commonsense. |
| Generation diagnostics | Fixed prompts, repetition, diversity | Whether generations are usable or collapse into repeated text. |

## Why These Tests

The suite is meant for base language models, not instruction-tuned assistants.
It focuses on whether a model trained from scratch is learning broad language
modeling behavior and whether it can compete with known small-model baselines.

The fixed GPT-2 comparison uses the same tests across runs so the target does
not move after seeing results. Later v5 work adds modern small-model baselines
such as Pythia, SmolLM, and SmolLM2-style comparisons.

## What The Suite Can Support

A checkpoint can make a strong quantitative claim when it improves across:

- held-out project perplexity,
- external perplexity,
- LAMBADA continuation metrics,
- most or all length-normalized multiple-choice checks,
- repetition and diversity diagnostics.

For example, SoloLLM v4.2 150M beats GPT-2 small across the fixed project suite.
That supports the claim that this checkpoint is a stronger base LM than GPT-2
small on these measurements.

## What The Suite Cannot Prove

This suite does not prove that a model is:

- a better chatbot,
- a better coding model,
- more factual,
- safer,
- better at long-context use,
- better than modern instruction-tuned models,
- better than much larger models such as GPT-3-class systems.

It also does not remove all risk of benchmark overfitting. It reduces that risk
by using multiple fixed tests, outside corpora, and baselines, but the final
answer still needs broader testing as the project moves into v5.

## Current Interpretation

SoloLLM v4.2 150M is the strongest completed checkpoint so far. It beats GPT-2
small across the fixed project suite, but it is larger than GPT-2 small.

The SoloLLM v4.2 123M ablation is slightly smaller than GPT-2 small and wins the
fixed normalized multiple-choice checks, but it does not beat GPT-2 small across
every metric. That means the strict smaller-than-GPT-2 claim remains unresolved.

The v5 direction is broader: compare against modern small-model baselines and
study which architecture, data, and training choices produce the best model that
can still run locally.

## Implementation

The suite is implemented by:

- `eval/v3_eval_suite.py`
- `eval/eval.py`
- `eval/external_benchmarks.py`
- `eval/multiple_choice_benchmarks.py`
- `eval/generate_samples.py`
- `eval/generation_metrics.py`

Detailed result reports are kept locally until they are promoted into the public
repo.
