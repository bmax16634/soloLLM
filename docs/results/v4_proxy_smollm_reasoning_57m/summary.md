# V4 Proxy Modern Eval Summary

The winning 57M `smollm_reasoning` proxy was evaluated on the same modern external suite used for the v3 baseline comparison.

## Result

The proxy confirms that the SmolLM-style data mix is promising, but it does not yet transfer into broad external strength at this size and training budget.

| Model | WikiText BPB ↓ | LAMBADA BPB ↓ | LAMBADA tok acc ↑ | HellaSwag norm ↑ | PIQA norm ↑ | ARC-E norm ↑ | ARC-C norm ↑ | WinoGrande norm ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 proxy `smollm_reasoning` 57M | 1.6170 | 1.5919 | 0.2271 | 0.2514 | 0.5680 | 0.3123 | 0.1973 | 0.5138 |
| v3 150M | 1.1909 | 1.3247 | 0.4743 | 0.3008 | 0.6270 | 0.4456 | 0.2508 | 0.5114 |
| GPT-2 small | 1.2216 | 1.3764 | 0.4667 | 0.2953 | 0.6360 | 0.4035 | 0.2207 | 0.4972 |
| SmolLM2 135M | 1.1071 | 1.2616 | 0.5833 | 0.4109 | 0.6710 | 0.5667 | 0.3244 | 0.5162 |

## Interpretation

- The internal proxy validation improved strongly: v3 pilot proxy PPL `63.63` to `47.09`.
- The external suite did not improve enough at 57M/300M tokens to beat v3 150M.
- This does not invalidate the data mix. It means the next serious test should scale the winning data mix into a full v4 150M run rather than treating the proxy as a production model.

## Next Decision

Build a full 10B-token v4 dataset using the `smollm_reasoning` mix and train a fresh 150M model. Keep the architecture close to v3 because the architecture proxy did not justify a redesign.
