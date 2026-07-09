# Phase 4 Generation Metrics

Automatic metrics over the fixed prompt generations. These are support metrics, not a replacement for reading the raw samples.

| Model | Samples | Avg tokens | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 8 | 74.6 | 0.3869 | 0.7836 | 0.1715 | 0.1120 | 2 |
| v4_2_dropout0_v4_1_blend_proxy_57m | 8 | 77.6 | 0.3398 | 0.6919 | 0.2567 | 0.1569 | 3 |

Lower repeated-ngram fractions and fewer bad-loop detections are better. Higher distinct scores are usually better, but very high diversity can also reflect incoherence on tiny sample sets.
