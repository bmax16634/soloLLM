# V4.1 Blend Proxy Modern Eval Summary

The 57M `v4.1_blend` proxy was evaluated on the same modern external suite used
for the previous `smollm_reasoning` proxy.

## Result

The blended mix improves broad external perplexity versus the reasoning-heavy
proxy, but gives back some of the multiple-choice gains.

| Model | WikiText BPB ↓ | WikiText PPL ↓ | LAMBADA BPB ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| v4 proxy `smollm_reasoning` 57M | 1.6170 | 155.71 | 1.5919 | 72.56 | 0.2271 | 0.3686 |
| v4.1 proxy `blend` 57M | 1.5937 | 144.80 | 1.5619 | 66.93 | 0.2408 | 0.3601 |

## Multiple Choice

| Benchmark | `smollm_reasoning` norm | `v4.1_blend` norm | Read |
|---|---:|---:|---|
| HellaSwag | 0.2514 | 0.2476 | slight regression |
| PIQA | 0.5680 | 0.5610 | slight regression |
| ARC-Easy | 0.3123 | 0.2860 | regression |
| ARC-Challenge | 0.1973 | 0.2040 | slight improvement |
| WinoGrande | 0.5138 | 0.5020 | regression |

## Interpretation

This is the first signal that the v4.1 blend corrects the broad-LM weakness of
the reasoning-heavy proxy. It has better WikiText, LAMBADA, and LAMBADA
last-token accuracy than `smollm_reasoning`.

It is not strong enough to scale directly. The MC average moved down from
`0.3686` to `0.3601`, and the internal proxy validation PPL was also weaker
than `smollm_reasoning` (`58.45` vs `47.09`).

## Decision

Do not start a full 150M/10B train from this exact mix. Use it as evidence for a
second v4.1 mix: keep the broad-web recovery, but restore some of the reasoning
signal that helped MC.
