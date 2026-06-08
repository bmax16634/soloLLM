# Phase 4 Generation Metrics

Automatic metrics over the fixed prompt generations. These are support metrics, not a replacement for reading the raw samples.

| Model | Samples | Avg tokens | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 8 | 74.6 | 0.3869 | 0.7836 | 0.1715 | 0.1120 | 2 |
| v2 | 8 | 75.5 | 0.3907 | 0.7512 | 0.2015 | 0.1282 | 2 |

Lower repeated-ngram fractions and fewer bad-loop detections are better. Higher distinct scores are usually better, but very high diversity can also reflect incoherence on tiny sample sets.
