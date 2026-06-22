# V3 Final GPT-2 Comparison

Date: June 19, 2026

This report closes the v3 experiment series. It compares the final v3 models
against GPT-2 small using the fixed v3 evaluation suite:

- project held-out perplexity on shards `58:60`,
- WikiText-2 perplexity,
- LAMBADA perplexity and continuation accuracy,
- fixed multiple-choice continuation scoring,
- fixed-prompt generation metrics.

The final v3 work produced two important checkpoints:

1. `v3-plus-150m-1024`: the best overall model, but larger than GPT-2 small.
2. `v3-gpt2-scale-1024`: the smaller-than-GPT-2 test model.

## Bottom Line

V3 succeeded at building a GPT-2-class base LM from scratch on one RTX 3090
that beats GPT-2 small overall. The strongest checkpoint is the 151.9M-param
v3 150M model trained on the curated 10B-token dataset.

The stricter smaller-than-GPT-2 hypothesis was not fully proven. The 123.6M-param
model is about 0.7% smaller than GPT-2 small and beats GPT-2 on most external
benchmarks, but it loses project held-out perplexity and some generation
diversity/repetition metrics.

The honest project conclusion is:

> Modern architecture plus a curated 10B-token dataset can beat GPT-2 small
> overall on a single RTX 3090. A slightly smaller model gets very close and
> wins many external evaluations, but strict smaller-and-better superiority
> across every metric remains unproven.

## Final Models

| Model | Params | Context | Train tokens | Role |
| --- | ---: | ---: | ---: | --- |
| GPT-2 small | 124,439,808 | 1024 | external baseline | baseline |
| v2 5.60B modern small | 91,654,400 | 512 | 5.60B | v2 diagnostic baseline |
| v3 123M 10B | 123,551,232 | 1024 | 9.80B | smaller-than-GPT-2 ablation |
| v3 150M 10B | 151,868,928 | 1024 | 10.00B | final best model |

The v3 123M run used one clean pass over the train split from
`data/v3_10b_1024`, which contains 9,800,728,576 train tokens. The v3 150M run
continued the fresh 5B run to about 10B total tokens.

## Training Summary

| Model | Tokens | Runtime | Avg tok/s | Peak VRAM | Best train-val PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 123M 10B | 9.80B | 68.60h | 39.7k | 11.8GB | 24.07 |
| v3 150M 10B | 10.00B | 43.34h continuation | 32.0k | 13.8GB | 23.53 |

The 150M runtime above is for the 5B-to-10B continuation run. The full 150M
training path was a fresh 5B run followed by this continuation.

## Core Results Against GPT-2

Lower perplexity is better. Higher accuracy is better.

| Benchmark | GPT-2 small | v3 123M | v3 150M | Read |
| --- | ---: | ---: | ---: | --- |
| Project held-out PPL | 25.32 | 25.64 | 24.90 | 150M wins; 123M narrowly loses |
| WikiText-2 PPL | 45.32 | 41.87 | 41.18 | both v3 models win |
| LAMBADA PPL | 40.62 | 36.28 | 35.35 | both v3 models win |
| LAMBADA last-token acc | 46.67% | 46.67% | 47.43% | 123M ties; 150M wins |
| LAMBADA last-word acc | 32.60% | 32.80% | 33.07% | both v3 models win slightly |
| MC avg acc norm | 41.05% | 42.46% | 42.71% | both v3 models win |

## Held-Out Perplexity Detail

| Shard | GPT-2 PPL | v3 123M PPL | v3 150M PPL |
| --- | ---: | ---: | ---: |
| `shard_00058.pt` | 25.22 | 25.56 | 24.83 |
| `shard_00059.pt` | 25.42 | 25.73 | 24.99 |
| `shard_00060.pt` | 25.26 | 25.54 | 24.81 |
| Aggregate | 25.32 | 25.64 | 24.90 |

The 123M model is consistently close but behind GPT-2 on the project held-out
split. This is the main reason it cannot be claimed as a strict across-board
GPT-2 win.

## Multiple-Choice Continuation Scoring

`Accuracy norm` is the primary metric because it normalizes by continuation
length.

| Benchmark | GPT-2 acc norm | v3 123M acc norm | v3 150M acc norm | Read |
| --- | ---: | ---: | ---: | --- |
| HellaSwag | 29.53% | 29.85% | 30.08% | both v3 models win |
| PIQA | 63.60% | 63.40% | 62.70% | GPT-2 wins |
| ARC-Easy | 40.35% | 44.04% | 44.56% | both v3 models win |
| ARC-Challenge | 22.07% | 24.08% | 25.08% | both v3 models win |
| WinoGrande | 49.72% | 50.91% | 51.14% | both v3 models win |
| Average | 41.05% | 42.46% | 42.71% | both v3 models win |

The multiple-choice result is one of the strongest signs that the v3 data mix
improved general language competence beyond the project held-out distribution.

## Generation Metrics

These metrics use the fixed prompt suite, so they should be read as a small
diagnostic rather than a complete human preference evaluation.

| Metric | GPT-2 small | v3 123M | v3 150M | Read |
| --- | ---: | ---: | ---: | --- |
| Corpus distinct-1 | 0.3869 | 0.3918 | 0.4308 | both v3 models win |
| Corpus distinct-2 | 0.7836 | 0.7734 | 0.8053 | 150M wins; 123M loses |
| Mean unique token ratio | 0.5905 | 0.5312 | 0.5759 | GPT-2 wins |
| Repeated bigram fraction | 0.1715 | 0.1805 | 0.1723 | GPT-2 wins slightly |
| Repeated trigram fraction | 0.1120 | 0.0906 | 0.1006 | both v3 models win |
| Bad-loop count | 2 / 8 | 1 / 8 | 2 / 8 | 123M wins; 150M ties |

The generation metrics are mixed. V3 improves some repetition behavior, but the
smaller 123M model has weaker unique-token diversity than GPT-2. The 150M model
is better balanced and is the stronger qualitative candidate.

## Historical V2 Read

V2 was useful because it exposed the main failure mode. The 91.65M-param v2
5.60B checkpoint was close to GPT-2 on project held-out perplexity, but it lost
badly on external perplexity and lost every multiple-choice benchmark.

| Benchmark | v2 5.60B | GPT-2 baseline from v2 suite | Read |
| --- | ---: | ---: | --- |
| Project held-out PPL | 25.56 | 25.32 | close |
| WikiText-2 PPL | 84.81 | 49.86 | major v2 weakness |
| LAMBADA PPL | 63.03 | 42.26 | major v2 weakness |
| LAMBADA last-word acc | 29.09% | 32.60% | GPT-2 wins |
| MC avg acc norm | 38.97% | 41.05% | GPT-2 wins |

The v2 result led directly to the v3 strategy: improve data breadth, move to
1024 context, keep the modern architecture, and test GPT-2-scale and modestly
larger models on the same fixed suite.

## Conclusions

### What succeeded

- Built a reproducible single-GPU pretraining and eval workflow.
- Curated a 10B-token dataset from FineWeb-Edu, DCLM, FineWeb, Wikipedia, and
  OpenWebText.
- Trained a 151.9M-param base LM from scratch that beats GPT-2 small overall.
- Trained a 123.6M-param model that is smaller than GPT-2 and beats GPT-2 on
  WikiText-2, LAMBADA perplexity, LAMBADA last-word accuracy, and multiple-choice
  average.
- Used the smaller 123M run as a real ablation instead of overclaiming from the
  larger model.

### What did not succeed

- The 123M smaller-than-GPT-2 model did not beat GPT-2 across every metric.
- It lost project held-out perplexity.
- It also lost unique-token ratio, corpus distinct-2, and repeated-bigram rate
  in the fixed generation diagnostic.

### Final project claim

The project should claim:

> SoloLLM v3 trains GPT-2-class base LMs from scratch on one RTX 3090. The final
> 150M model beats GPT-2 small overall on a fixed evaluation suite, while a
> slightly smaller 123M model beats GPT-2 on most external benchmarks but does
> not fully clear the strict across-board smaller-than-GPT-2 bar.

It should not claim:

> A smaller-than-GPT-2 model beats GPT-2 across the board.

## Artifacts

| Artifact | Path |
| --- | --- |
| SoloLLM Hugging Face collection | <https://huggingface.co/collections/bmax16634/solollm> |
| v3 150M Hugging Face model | <https://huggingface.co/bmax16634/sologpt-v3-150m-base> |
| v3 123M Hugging Face ablation | <https://huggingface.co/bmax16634/sologpt-v3-123m-base> |
| Public completion demo | <https://huggingface.co/spaces/bmax16634/sologpt-v3-150m-demo> |
| v3 123M config | `sologpt_v3/config_gpt2_scale_1024.json` |
| v3 150M config | `sologpt_v3/config_plus_150m_1024.json` |
| v3 data manifest | `sologpt_v3/data_sources.yaml` |
| v3 10B dataset stats | `/home/bmx/_projects/soloLLM/data/v3_10b_1024/build_stats.json` |
| v3 123M training summary | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/metrics_summary.json` |
| v3 123M final model | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/final_model.pt` |
| v3 123M full eval | `outputs/eval_suites/v3_fresh_10b_123m_10bdata_gpt2_full_suite/` |
| v3 150M final model | `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/final_model.pt` |
| v3 150M full eval | `outputs/eval_suites/v3_fresh_10b_150m_10bdata_gpt2_full_suite/` |
| v2 diagnostic report | `docs/results/v2_gpt2_full_diagnostic.md` |
