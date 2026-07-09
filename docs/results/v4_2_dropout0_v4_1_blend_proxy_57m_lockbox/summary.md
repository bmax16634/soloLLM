# V4.2 Dropout-0 V4.1 Blend Lockbox Eval

Lockbox-style verification for the 57M proxy scale candidate.

- Held-out shards: `shard_00058.pt` to `shard_00060.pt`
- Max batches per shard: `512`

## Held-Out PPL

| Model | Tokens | Loss | PPL |
|---|---:|---:|---:|
| v4_2_dropout0_v4_1_blend_proxy_57m | 12,558,336 | 4.2131 | 67.57 |
| gpt2 | 12,558,336 | 3.2411 | 25.56 |

## Generation Metrics

| Model | Samples | Avg tokens | Distinct-1 | Distinct-2 | Repeated bigrams | Bad loops |
|---|---:|---:|---:|---:|---:|---:|
| gpt2 | 8 | 74.6 | 0.3869 | 0.7836 | 0.1715 | 2 |
| v4_2_dropout0_v4_1_blend_proxy_57m | 8 | 77.6 | 0.3398 | 0.6919 | 0.2567 | 3 |

## Decision

This does not show a model win. The 57M/300M proxy is still much worse than
GPT-2 on sampled project held-out PPL and has worse generation repetition
metrics.

That is expected for a small proxy, but it matters for the research read: the
proxy result only says this recipe is the best scale candidate so far. It does
not prove the final 150M model will beat GPT-2.

Next decision: either scale this recipe to a fresh 150M/10B run, or run the same
lockbox check on older proxies if we want a stricter proxy-vs-proxy lockbox
comparison before spending the full training budget.
