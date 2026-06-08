# SoloLLM V3 Plan: Beat GPT-2 Broadly

## Starting Point

SoloLLM v2 is complete as an engineering iteration. The best v2 checkpoint is the 91.65M-param `v2-modern-small` 5.60B-token stretch checkpoint.

V2 succeeded at:

- beating the v1 baseline clearly,
- getting within about `0.95%` perplexity of GPT-2 small on the project held-out split,
- proving the v2 architecture/training/eval system is reproducible,
- documenting generation and external benchmark gaps honestly.

V2 did not beat GPT-2 small across the board. V3 should target that directly.

## What The V2 Gaps Show

| Check | V2 5.60B | GPT-2 small | Gap | What it suggests |
| --- | ---: | ---: | ---: | --- |
| Project held-out PPL, shards `58:60` | 25.56 | 25.32 | v2 `+0.95%` | V2 is close in-domain. The architecture and training loop are not the main blocker. |
| WikiText-2 PPL, full test | 84.81 | 49.86 | v2 `+70.1%` | V2 generalizes worse to a different text distribution. Data mix/quality is likely the biggest v3 lever. |
| LAMBADA PPL, full test | 63.03 | 42.26 | v2 `+49.2%` | V2 assigns weaker probabilities to longer natural continuations. This points to data, context length, and calibration/generalization. |
| LAMBADA last-token accuracy | 44.30% | 46.67% | v2 `-2.37` points | The model can often identify plausible next tokens; the gap is not hopeless. |
| LAMBADA last-word greedy exact | 29.09% | 32.60% | v2 `-3.51` points | GPT-2 is still better at exact continuation, but the accuracy gap is smaller than the perplexity gap. |
| HellaSwag acc norm | 26.99% | 29.53% | v2 `-2.54` points | GPT-2 is better at commonsense continuation. |
| PIQA acc norm | 61.00% | 63.60% | v2 `-2.60` points | GPT-2 is better at physical commonsense. |
| ARC-Easy acc norm | 37.89% | 40.35% | v2 `-2.46` points | GPT-2 is better on simple science QA scoring. |
| ARC-Challenge acc norm | 19.73% | 22.07% | v2 `-2.34` points | GPT-2 is better on harder science QA scoring. |
| WinoGrande acc norm | 49.25% | 49.72% | v2 `-0.47` points | This is the closest broad-eval task. |
| Fixed prompts distinct-2 | 0.7512 | 0.7836 | v2 `-0.0323` | GPT-2 has slightly healthier n-gram diversity under the same sampling settings. |
| Fixed prompts repeated bigram fraction | 0.2015 | 0.1715 | v2 `+0.0300` | V2 is slightly more repetitive in sampled continuations. |

The important read is not "v2 failed." The important read is:

> V2 is close on its own held-out OpenWebText-style split, but GPT-2 is still more robust when the distribution shifts.

That means v3 should not only continue v2 for more tokens. It should improve the data mixture, context length, and eval-driven training loop.

## Likely Root Causes

### 1. Data Distribution Gap

The project held-out split is close, but WikiText-2 and LAMBADA are much worse. That points to a distribution/generalization gap more than an architecture failure.

V3 response:

- use a broader, cleaner training mix,
- improve deduplication and document quality filtering,
- include more long-form encyclopedic/prose text,
- keep a small external validation set visible during training.

### 2. Context-Length Gap

V2 trains and evaluates at 512 tokens. GPT-2 small has a native 1024-token context. The external benchmark script uses a shared 512-token context for fair scoring, but the training history still reflects a shorter-context model.

V3 response:

- move the main model to 1024 context,
- regenerate or reshape shards for 1024-token sequences,
- preserve a 512-token eval view only for backward-compatible v2 comparison.

### 3. Capacity Gap

The best v2 checkpoint is 91.65M params, while GPT-2 small is 124.44M params. Getting close with fewer params is a good v2 result, but beating GPT-2 broadly may require either GPT-2-small parity scale or a slightly larger modern config.

V3 response:

- start with a GPT-2-small-scale modern model as the main candidate,
- consider a 140M-180M modern-small-plus variant if 3090 memory and throughput remain acceptable,
- keep parameter counts explicit so the result is not just "bigger wins."

### 4. Generation Stability Gap

The fixed prompt metrics show similar diversity, but GPT-2 has slightly lower repeated n-gram rates. The raw samples also show that v2 still drifts factually and weakens on code/structured prompts.

V3 response:

- track generation metrics during milestone evals,
- add prompt categories for factual, code-ish, structured list, and repetition stress,
- keep raw non-cherry-picked samples in the result docs.

### 5. Eval-Driven Training Gap

V2 used the full held-out eval at the end. The external checks were added after the final checkpoint. That is fine for v2, but v3 should use those signals earlier.

V3 response:

- run capped WikiText/LAMBADA validation at checkpoints,
- watch for external validation improvement, not only project validation PPL,
- stop or adjust when project PPL improves but external PPL does not.

## Full V3 Evaluation Suite

The v2 eval suite should become the minimum frozen v3 comparison suite, and v3 should add multiple-choice base-LM benchmarks before any "beats GPT-2 across the board" claim.

The executable runner and commands are documented in `docs/V3_EVAL_SUITE.md`.

### Required Comparability Suite

1. full project held-out PPL on shards `58:60`,
2. per-shard stability table,
3. fixed prompt generation suite,
4. automatic generation metrics,
5. WikiText-2 perplexity,
6. LAMBADA perplexity,
7. LAMBADA last-token accuracy,
8. LAMBADA last-word greedy exact accuracy.

This suite answers whether v3 improved over v2 on the exact tests used to close v2.

### Required Broad-Eval Suite

For v3, the final public comparison should also include multiple-choice continuation scoring:

| Benchmark | Why include it | Primary metric |
| --- | --- | --- |
| HellaSwag | commonsense continuation under distribution shift | length-normalized accuracy |
| PIQA | physical commonsense | length-normalized accuracy |
| ARC-Easy | grade-school science QA | length-normalized accuracy |
| ARC-Challenge | harder science QA | length-normalized accuracy |
| WinoGrande | pronoun/coreference-style commonsense | length-normalized accuracy |

These are not instruction-following tests. They are base-LM scoring tests: each answer choice is scored by conditional log-likelihood, and the highest-scoring choice wins.

For v3 final eval:

- remove the current 1,000-example LAMBADA cap for final eval,
- run WikiText-2 without a token cap, or document the exact cap if kept for repeatability,
- run the multiple-choice benchmarks on full validation splits,
- keep GPT-2 small and v2 5.60B as fixed baselines,
- report all failures and do not cherry-pick prompts.

## Is The Current Eval The Final Eval For V3?

It should be part of the final eval, but not the whole final eval.

Use the current v2 suite as the frozen comparability suite. That lets v3 answer:

> Did v3 beat v2 and GPT-2 on the same tests used to close v2?

But if the claim is:

> V3 truly beats GPT-2 across the board.

Then v3 needs the expanded final eval above. The current suite is strong enough to guide v3, but the final public claim should include full external checks and at least one additional base-LM benchmark family.

## V3 Success Criteria

V3 should only claim it beats GPT-2 small if it beats GPT-2 small on:

- project held-out PPL,
- WikiText-2 PPL,
- LAMBADA PPL,
- LAMBADA last-token or last-word accuracy,
- most full validation multiple-choice benchmarks by length-normalized accuracy,
- generation repetition metrics without worse qualitative samples.

If v3 beats GPT-2 on the project held-out split but not on external checks, the honest claim should be:

> V3 beats GPT-2 on the project distribution, but GPT-2 remains more robust out-of-domain.

## Practical Next Step

Start v3 with a data/eval pilot, not a long training run:

1. define the v3 data mixture,
2. reshape shards for 1024-token context,
3. train a short 300M-500M token pilot,
4. evaluate project validation plus capped WikiText/LAMBADA,
5. choose the final v3 architecture and token budget based on those curves.
