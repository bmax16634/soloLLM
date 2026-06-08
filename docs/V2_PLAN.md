# SoloLLM V2 Full Detailed Plan: Small, Honest, Real LM

> Supersedes the earlier "10/10 scorecard" V2 plan (preserved in git history). This is the
> honest "small but real" strategy. See the linked Ariya wiki hub `[[solollm]]` for context and status.

## Reviewer notes (2026-05-31)

Technical corrections to fold in during implementation — the plan is sound; these tighten it:

1. **Parameter count is ~124M, not 160M.** 12 layers × 768 embd × 12 heads × 50257 vocab is
   exactly GPT-2-small's architecture (~124M params). The param-count helper will confirm.
   Treat this as a *feature*: matching GPT-2-small's size makes the perplexity comparison
   perfectly apples-to-apples — state that explicitly in the brief. Update the "about 160M" and
   "about 204M v1" estimates once the counter reports real numbers.
2. **Re-measure throughput after the resize.** The observed 14,277 tok/s was almost certainly the
   483M experimental model. A 124M model on a 3090 should train meaningfully faster, so the
   "1.23B tokens/day" estimate is likely conservative. Re-benchmark after the 50M sanity run.
3. **Use the VRAM headroom.** 124M uses a fraction of 24GB; raise `batch_size` above 16 (or cut
   `grad_accum_steps`) to train faster. Re-tune after the sanity run.
4. **Make the LR schedule explicit:** cosine decay from `learning_rate` to `min_learning_rate`
   after `warmup_tokens`. Name it in config/docs.
5. **Eval stride consistency:** compute perplexity for v2 *and* GPT-2 with the same windowing
   (non-overlapping vs sliding) or the comparison skews. One shared code path in `eval/eval.py`.

---

## 2026-06-06 plan update: two-variant v2 strategy

The clarified v2 goal is consumer-hardware training on a single RTX 3090, with an explicit experiment:

> Can a smaller modern from-scratch LM trained on a 3090 approach or beat GPT-2-small on a fair held-out perplexity test?

V2 should now have two tracked variants:

| Variant | Target Size | Purpose |
| --- | ---: | --- |
| `v2-gpt2-parity` | 123,616,512 params | Control model at GPT-2-small scale for an apples-to-apples comparison. |
| `v2-modern-small` | 91,654,400 params | Smaller modern architecture intended to test the original goal: beat or approach GPT-2-small with fewer parameters on consumer hardware. |

The parity model should remain because it makes the evaluation interpretable. The modern-small model is the headline experiment if it performs well.

Phase 1 is no longer just "run the 50M sanity test." Phase 1 is:

1. Verify both variants with CPU smoke tests. Done.
2. Run 50M-token sanity checks for both variants on the 3090. Done.
3. Compare loss, tokens/sec, GPU memory, and checkpoint health. Done.
4. Decide which variant earns the 300M pilot. Done: `v2-modern-small`.

See `docs/PHASE_1_PLAN.md` for the immediate execution plan.

## 2026-06-08 result update

The main v2 experiment is complete enough for portfolio documentation:

| Result | Status |
| --- | --- |
| 50M two-variant sanity | Complete; `v2-modern-small` beat the GPT-2-parity control on capped validation PPL, speed, and size. |
| 300M pilot | Complete; validation PPL reached `42.95` and justified the long run. |
| 3B final run | Complete; full held-out test PPL `26.26` on shards `58:60`. |
| v1 comparison | Complete; best v2 beats v1 by about `16.1%` lower PPL on the full held-out split. |
| GPT-2 comparison | Complete; best v2 is close but does not beat GPT-2-small on the full held-out split. |
| 5.60B stretch checkpoint | Full 331M-token held-out eval complete; PPL `25.56`, about `0.95%` above GPT-2 on the same split. |
| Robust GPT-2 comparison | Complete; fixed generation metrics are close, but GPT-2 leads on WikiText-2 PPL, LAMBADA PPL, and LAMBADA token/word accuracy. |

The official v2 result should now use the full-evaluated 5.60B stretch checkpoint as the best perplexity checkpoint, while keeping the 3B checkpoint documented as the main planned run. V2 should be closed honestly: it approaches GPT-2 on the project held-out split, but GPT-2 remains stronger across broader external checks. The v3 response is documented in `docs/V3_PLAN.md`.

## 1. Summary

SoloLLM v2 should be a deliberately designed from-scratch language model, not just a larger version of v1. The honest story is:

> v1 was a rough prototype that proved the pipeline could work. V2 is the serious engineering iteration: smaller than the current experimental v2, cleaner architecturally, reproducible, measurable, and evaluated fairly.

The target is not chatbot quality. The target is a small GPT-style model that:

- trains cleanly on one RTX 3090,
- shows real validation loss reduction,
- has reproducible smoke tests,
- has fair perplexity evaluation,
- produces limited but honest completions,
- and makes the portfolio look like a serious ML systems project.

Default v2 direction:

- Control model: `v2-gpt2-parity`, 123,616,512 params, 12 layers, 768 embedding size, 12 heads, 512 context length.
- Headline experiment: `v2-modern-small`, 91,654,400 params, modern block design, same 512-token context.
- Hardware target: single RTX 3090.
- Training result: staged runs completed through a 3B-token final checkpoint, plus a durable 5.60B stretch checkpoint with full held-out eval.
- Portfolio rating target: move from current 7/10 to 10/10 through reproducibility, architecture, training, evaluation, and presentation.

## 2. Current State

Phase 0 foundation work is implemented in the current working tree.

Existing useful pieces:

- sologpt_v1/ contains the cleaned v1 baseline.
- sologpt_v2/model.py has the GPT-2-parity v2 architecture: RoPE, pre-norm blocks, SDPA attention, tied embeddings, context validation, and parameter summary.
- sologpt_v2/pretrain.py has CLI-driven training, dry-run support, JSONL metrics, checkpoint metadata, latest checkpoint, resume support, validation caps, and summary output.
- eval/eval.py has a CLI path for v1, v2, and GPT-2.
- tests/ contains CPU-safe model, training, resume, eval, and CLI smoke tests.
- docs/PHASE_0_CHECKLIST.md records Phase 0 acceptance criteria.
- docs/PHASE_1_PLAN.md records the two-variant Phase 1 plan.
- docs/PHASE_2_PLAN.md records the fresh 300M modern-small pilot plan.
- Local tokenized OpenWebText shards exist:
    - 61 shards total.
    - Most shards are shaped (292968, 512).
    - Each full shard is about 150M tokens.
    - Total local token budget is about 9B tokens.

- Existing experimental v2 was too large for the new goal:
    - current v2 config is roughly 483M params.
    - it hit about 24GB peak GPU memory.
    - it is useful as evidence/history but should not be the main final v2 direction.

Existing problems:

- v1 should not be presented as strong; it barely functions and should be described as a rough baseline.
- v2-modern-small is implemented and completed the 50M-token GPU sanity run.
- Phase 1 50M-token sanity results are recorded in docs/results/phase1_50m_sanity.md.
- v2-modern-small completed the 300M-token Phase 2 pilot.
- Phase 2 300M-token pilot results are recorded in docs/results/phase2_300m_pilot.md.
- v2-modern-small completed the final 3B-token run and full Phase 4 held-out eval.
- The 3B final run and 5.60B stretch-checkpoint full eval are recorded in docs/results/final_3b_modern_small.md.
- Fixed generation samples are recorded in docs/results/phase4_generations.md.
- Model publication is still pending.

## 3. V2 Product Goal

V2 should prove four things:

1. Architecture maturity
    - The model is no longer "whatever worked first."
    - It uses a recognizable GPT-style decoder design.

2. Training maturity
    - Runs are reproducible.
    - Runs can resume.
    - Metrics are structured and easy to plot.
    - The training loop can be tested without a GPU.

3. Evaluation maturity
    - v2 is evaluated on fixed held-out shards.
    - v1, v2, and GPT-2 are compared honestly.
    - Results are not oversold.

4. Portfolio maturity
    - The README and project brief explain what was built, what failed, what improved, and what the limits are.
    - A reviewer can run a smoke test quickly.
    - A reviewer can understand the technical decisions without reading every source file.

## 4. V2 Success Criteria

V2 is successful if:

- python -m pytest passes from a clean clone without model weights or OpenWebText.
- A toy training run completes on CPU.
- A 50M-token sanity run shows decreasing training loss.
- A 300M-token pilot run shows sane validation behavior.
- A final 3B-token run completes with a clear documented result.
- Final v2 has a fixed test perplexity on held-out shards.
- Final v2 is compared with v1 and GPT-2-small using the broader Phase 4 evaluation plan: held-out perplexity, per-shard results, efficiency metrics, fixed prompt samples, and documented failures.
- The repo has a result table comparing:
    - v1,
    - v2,
    - GPT-2 small.

- README includes:
    - exact setup,
    - smoke test command,
    - training command,
    - eval command,
    - result table,
    - limitations.

- Portfolio brief explains that v1 was rough and v2 was the serious rebuild.

V2 does not need to:

- be a useful assistant,
- guarantee beating GPT-2,
- sound polished on long generations,
- be bigger than v1,
- hide failure cases.

V2 should, however, make a serious attempt to test whether `v2-modern-small` can beat or approach GPT-2-small with fewer parameters under the single-3090 constraint.

## 5. Model Design

Use two model configs: a GPT-2-small-scale control and a smaller modern experimental model.

### `v2-gpt2-parity` config

```json
{
  "model_type": "sologpt",
  "vocab_size": 50257,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "seq_length": 512,
  "max_seq_len": 512,
  "dropout": 0.1,
  "rope_base": 10000.0,
  "tie_weights": true,
  "qkv_bias": false,
  "learning_rate": 3e-4,
  "min_learning_rate": 3e-5,
  "weight_decay": 0.1,
  "batch_size": 16,
  "grad_accum_steps": 4,
  "clip_grad_norm": 1.0,
  "warmup_tokens": 50000000,
  "tokens_per_checkpoint": 300000000,
  "eval_every_tokens": 100000000,
  "log_every_opt_steps": 50,
  "seed": 1337,
  "device": "cuda",
  "save_path": "./outputs/sologpt_v2",
  "shard_dir": "./data/tokenized_chunks"
}
```

Expected size:

- 123,616,512 parameters with the current tied-embedding implementation.
- GPT-2-small-scale control.
- Much smaller than current experimental v2 estimate of about 483M.

### `v2-modern-small` target config

```json
{
  "model_type": "sologpt",
  "variant": "modern_small",
  "vocab_size": 50257,
  "n_embd": 640,
  "n_layer": 12,
  "n_head": 10,
  "seq_length": 512,
  "max_seq_len": 512,
  "dropout": 0.1,
  "rope_base": 10000.0,
  "tie_weights": true,
  "qkv_bias": false,
  "norm_type": "rmsnorm",
  "mlp_type": "swiglu",
  "use_bias": false,
  "init_std": 0.02,
  "scale_residual_init": true
}
```

Expected size:

- 91,654,400 parameters with the current tied-embedding implementation.
- Smaller than GPT-2-small.
- Designed as the real "can a smaller modern model compete?" experiment.

Architecture changes:

- Use pre-norm transformer blocks.
- Use RoPE instead of absolute positional encoding.
- Use torch.nn.functional.scaled_dot_product_attention for attention.
- Use causal attention with is_causal=True when supported.
- Use GELU MLP for `v2-gpt2-parity`.
- Add RMSNorm and SwiGLU support for `v2-modern-small`.
- Tie input embeddings and output LM head if tie_weights=true.
- Remove device movement inside forward.
- Add explicit sequence length validation:
    - if input length exceeds max_seq_len, raise a clear ValueError.
- Add GPT-style weight initialization and residual projection scaling.

- Add parameter-count helper:
    - total params,
    - trainable params,
    - non-embedding params if easy.

Important implementation detail:

- Keep the public class name SoloGPT_v2.
- Keep forward(input_ids) -> logits.
- Do not break simple import:

  from sologpt_v2 import SoloGPT_v2

## 6. Data Split

Use the existing local shards and document a fixed split.

Default split:

- Train:
    - shard_00000.pt through shard_00054.pt

- Validation:
    - shard_00055.pt through shard_00057.pt

- Test:
    - shard_00058.pt through shard_00060.pt

Rationale:

- Keeps final test shards untouched.
- Gives about 8.25B train-token availability.
- Gives about 450M validation-token availability.
- Gives about 332M test-token availability because final shard is smaller.

Important caveat:

- If v1 may have trained on shards now used for validation/test, document this.
- The v1 comparison can be labeled "historical baseline, split contamination possible."
- The v2 and GPT-2 comparison should use the fixed test split cleanly.

## 7. Training Strategy

Training should happen in stages.

### Stage 0: CPU Toy Test

Purpose:

- Confirm model, loss, optimizer, logging, and checkpoint writing work.

Inputs:

- Tiny fixture shard under tests/fixtures/tiny_shard.pt.
- Shape should be small, for example (16, 32) or (32, 64).
- Token ids should be random integers within a tiny vocab for test configs.

Command:

    python -m sologpt_v2.pretrain --config tests/fixtures/v2_tiny_config.json --dry-run --max-steps 3

Success:

- Completes on CPU.
- Loss is finite.
- Metrics file is written.
- Checkpoint or summary file is written depending on dry-run behavior.

### Stage 1: 50M Token Sanity Run

Purpose:

- Catch real-data bugs before spending days.

Command shape:

    python -m sologpt_v2.pretrain \
      --config sologpt_v2/config.json \
      --output-dir outputs/sologpt_v2/sanity_50m \
      --train-shards 0:54 \
      --val-shards 55:57 \
      --max-tokens 50000000

Success:

- Loss decreases from initial range.
- No NaNs.
- GPU memory is stable.
- Validation runs at least once.
- Output includes:
    - config copy,
    - run metadata,
    - metrics JSONL,
    - summary JSON.

### Stage 2: 300M Token Pilot

Purpose:

- Decide if architecture and hyperparameters are healthy.

Command shape:

    python -m sologpt_v2.pretrain \
      --config sologpt_v2/config.json \
      --output-dir outputs/sologpt_v2/pilot_300m \
      --train-shards 0:54 \
      --val-shards 55:57 \
      --max-tokens 300000000

Success:

- Training loss decreases smoothly.
- Validation loss decreases or stabilizes reasonably.
- No severe overfitting.
- Sample completions are not necessarily good, but they should show learned token structure better than random text.

Decision after pilot:

- If loss curve is broken, fix training/model before final run.
- If validation is healthy, start final run.

### Stage 3: Final 3B-4B Token Run

Purpose:

- Produce the v2 portfolio checkpoint.

Command shape:

    python -m sologpt_v2.pretrain \
      --config sologpt_v2/config.json \
      --output-dir outputs/sologpt_v2/final_v2 \
      --train-shards 0:54 \
      --val-shards 55:57 \
      --max-tokens 4000000000

Expected runtime (reviewer note 2: re-benchmark — likely faster at 124M):

- Observed prior run speed was about 14,277 tok/s.
- Approximate token throughput:
    - 1 day: 1.23B tokens.
    - 2 days: 2.47B tokens.
    - 3 days: 3.70B tokens.
    - 4 days: 4.93B tokens.

- Final run target should be 3B-4B tokens unless the curve plateaus earlier.

> DLPC note: the final run is multi-day shared-GPU compute. Before launching, check GPU
> load/VRAM/temps and other active jobs, and flag the tradeoff (per the Ariya house rules).

Success:

- Checkpoint saved.
- Final validation loss recorded.
- Metrics summary generated.
- Training curve plot generated.
- Final test evaluation can run.

## 8. Training CLI Requirements

Refactor sologpt_v2/pretrain.py to support CLI arguments.

Required args:

    --config PATH
    --output-dir PATH
    --shard-dir PATH
    --train-shards START:END
    --val-shards START:END
    --resume PATH
    --max-steps N
    --max-tokens N
    --dry-run
    --device cpu|cuda|mps
    --seed N

Defaults:

- --config: sologpt_v2/config.json
- --output-dir: config save_path
- --shard-dir: config shard_dir
- --train-shards: all shards except validation/test if not specified
- --val-shards: optional; no validation if missing
- --resume: none
- --device: config device, falling back to available hardware
- --seed: config seed or 1337

Behavior:

- Create a unique run directory under output dir unless output dir points to an explicit run path.
- Copy resolved config into run directory.
- Write all run artifacts into that run directory.
- Fail early with clear errors for:
    - no shards found,
    - invalid shard range,
    - sequence length mismatch,
    - missing resume checkpoint,
    - unsupported device.

- Do not write generated artifacts into tracked source directories.

## 9. Checkpointing And Resume

Checkpoint format:

```json
{
    "model_state": "...",
    "optimizer_state": "...",
    "scaler_state": "...",
    "scheduler_state": "...",
    "meta": {
        "run_id": "str",
        "tokens_total": "int",
        "global_steps": "int",
        "micro_step": "int",
        "epoch_or_pass": "int",
        "shard_idx": "int",
        "batch_idx": "int",
        "config": "dict",
        "git_commit": "str | None"
    }
}
```

Required checkpoint files:

- periodic checkpoints:
    - checkpoints/ckpt_<tokens>M.pt

- latest checkpoint:
    - checkpoints/latest.pt

- final model:
    - final_model.pt

Resume behavior:

- --resume outputs/sologpt_v2/final_v2/checkpoints/latest.pt
- Loads model, optimizer, scaler, scheduler, counters.
- Continues tokens and global steps from metadata.
- Appends to metrics JSONL or starts a new resumed metrics file with linkage to parent run.

Acceptance:

- A smoke test should train, save, resume, and train one more step.

## 10. Metrics And Artifacts

Each run directory should contain:

    outputs/sologpt_v2/<run_id>/
      config_resolved.json
      run_meta.json
      metrics.jsonl
      metrics_summary.json
      checkpoints/
        latest.pt
        ckpt_300M.pt
      final_model.pt
      plots/
        loss_curve.png
      samples/
        prompts.json
        generations.json

Metrics JSONL event types:

- run_start
- train
- validation
- checkpoint
- resume
- run_end
- error

Train event fields:

```json
{
  "type": "train",
  "tokens_total": 123456,
  "optimizer_steps": 123,
  "train_loss": 4.2,
  "lr": 0.00012,
  "grad_norm": 0.9,
  "tokens_per_sec": 14000,
  "gpu_mem_gb": 8.1,
  "gpu_peak_mem_gb": 18.5
}
```

Validation event fields:

```json
{
  "type": "validation",
  "tokens_total": 300000000,
  "optimizer_steps": 10000,
  "val_loss": 3.8,
  "val_ppl": 44.7,
  "val_tokens": 10000000
}
```

Summary fields:

```json
{
  "run_id": "...",
  "status": "complete|interrupted|failed",
  "final_train_loss": 0.0,
  "best_val_loss": 0.0,
  "best_val_ppl": 0.0,
  "tokens_total": 0,
  "optimizer_steps": 0,
  "total_time_sec": 0,
  "tokens_per_sec_avg": 0,
  "parameter_count": 0,
  "checkpoint_path": "...",
  "config": {}
}
```

## 11. Evaluation Plan

Refactor eval/eval.py into a proper CLI.

Required args:

    --model v1|v2|gpt2
    --checkpoint PATH
    --config PATH
    --shard-dir PATH
    --shards START:END
    --batch-size N
    --output-json PATH
    --device cpu|cuda|mps

Example commands:

    python eval/eval.py \
      --model v2 \
      --checkpoint outputs/sologpt_v2/final_v2/final_model.pt \
      --config sologpt_v2/config.json \
      --shard-dir data/tokenized_chunks \
      --shards 58:60 \
      --batch-size 8 \
      --output-json outputs/eval/v2_test.json

    python eval/eval.py \
      --model gpt2 \
      --shard-dir data/tokenized_chunks \
      --shards 58:60 \
      --batch-size 4 \
      --output-json outputs/eval/gpt2_test.json

Evaluation output JSON:

```json
{
  "model": "v2",
  "checkpoint": "...",
  "config": "...",
  "shards": [58, 59, 60],
  "loss": 0.0,
  "perplexity": 0.0,
  "tokens": 0,
  "parameter_count": 0,
  "context_length": 512,
  "device": "cuda",
  "elapsed_sec": 0
}
```

Portfolio result table columns:

- Model
- Params
- Context
- Training Tokens
- Eval Split
- Loss
- Perplexity
- Hardware
- Notes

Important result policy:

- Do not claim v2 beats GPT-2 unless it actually does.
- If v2 only beats v1, say that.
- If v2 does not beat v1 due to contamination or checkpoint issues, say v2 is more reproducible and better engineered, not necessarily better at text.
- Reviewer note 5: use the same perplexity windowing for v2 and GPT-2.

## 12. Generation And Samples

Add a v2 generation path after training.

Requirements:

- Support loading final_model.pt.
- Support prompts from JSON.
- Save outputs to samples/generations.json.

Prompt set:

```json
[
  "The future of artificial intelligence is",
  "In a small desert town,",
  "The reason the sky appears blue is",
  "Python is useful because",
  "Once upon a time"
]
```

Generation settings:

- max_new_tokens=80
- temperature=0.8
- top_k=40
- optionally top_p=0.95 if implemented.

Portfolio samples:

- Include 3 decent examples.
- Include 1 failure case.
- Explain that this is a small from-scratch LM, not an instruction-tuned assistant.

## 13. Tests

Add pytest.

Test files:

    tests/
      fixtures/
        tiny_shard.pt
        v2_tiny_config.json
      test_models.py
      test_training_smoke.py
      test_eval_smoke.py

test_models.py:

- v1 forward pass returns shape (batch, seq, vocab).
- v2 forward pass returns shape (batch, seq, vocab).
- v2 raises ValueError when sequence length exceeds max context.
- v2 parameter count is positive.
- tied embeddings share storage if tie_weights=true.

test_training_smoke.py:

- run v2 training for 2-3 optimizer steps on tiny shard.
- assert metrics file exists.
- assert summary file exists.
- assert checkpoint or final model exists.
- assert loss is finite.
- test resume from latest checkpoint for one additional step.

test_eval_smoke.py:

- run eval on tiny model/tiny shard.
- assert output JSON exists.
- assert loss and perplexity are finite.
- assert token count is positive.

Test command:

    python -m pytest

All tests must pass without:

- GPU,
- OpenWebText,
- real checkpoints,
- internet,
- private local paths.

## 14. Documentation Plan

Update README with:

- Honest v1 framing.
- V2 goal.
- Setup.
- Test command.
- Toy training command.
- Real training command.
- Eval command.
- Result table placeholder initially, real results after run.
- Link to project brief.

Add docs/PROJECT_BRIEF.md.

Project brief structure:

1. Problem
    - Build a from-scratch small LM and make it reproducible.

2. V1 baseline
    - Rough first prototype.
    - What worked: pipeline, data, model, demo.
    - What failed: weak quality, ad hoc design, weak reproducibility.

3. V2 design
    - Smaller, cleaner GPT-style architecture.
    - Why not bigger.

4. Training system
    - Shards, split, logging, checkpoints, resume.

5. Evaluation
    - Fixed test split.
    - v1/v2/GPT-2 comparison.

6. Results
    - Loss curve.
    - Perplexity table.
    - Sample generations.

7. Limitations
    - Small model.
    - Not instruction tuned.
    - Dataset limitations.
    - Compute constraints.

8. Lessons learned
    - From-scratch LMs need careful design and measurement.
    - Reproducibility matters more than a flashy demo.

Add docs/results/v1_vs_v2.md after final evaluation.

## 15. Portfolio Rating Path

Current project: 7/10

- Cleaned repo.
- Good story.
- V1 exists.
- V2 plan exists.
- But no formal tests, no reproducible final v2 result.

After Milestone 1: 8/10

- Tests.
- Toy run.
- Clean commands.
- README becomes more credible.

After Milestone 2-4: 9/10

- Real v2 architecture.
- Serious training loop.
- Fixed evaluation.
- Results table.

After Milestone 5-6: 10/10

- Project brief.
- Plots.
- Samples.
- Limitations.
- Model card or clear checkpoint story.
- Demo or documented generation path.

## 16. Implementation Order

Do the work in this exact order:

1. Update plan/docs to reflect "small but real" v2.
2. Add tests and fixtures.
3. Add tiny CPU training mode.
4. Refactor v2 model to final 160M design.
5. Refactor v2 training CLI.
6. Add checkpoint resume.
7. Add scheduler and seed handling.
8. Add eval CLI.
9. Add plotting utility.
10. Run toy test.
11. Run 50M sanity training.
12. Inspect curve and samples.
13. Run 300M pilot training.
14. Inspect validation curve.
15. Run final 3B-4B training.
16. Run final eval on shards 58-60.
17. Generate sample outputs.
18. Write project brief and final README results.
19. Optionally publish checkpoint to Hugging Face.
20. Tag final portfolio state.

## 17. Branch And Commit Plan

Use main as the public clean branch.

Suggested branches:

- v2-dev
    - implementation work: tests, model, training CLI, eval CLI.

- v2-training
    - long-running training outputs stay untracked.
    - merge only code/docs/results summaries back.

- main
    - final polished portfolio state.

Suggested commits:

1. Revise v2 plan for small real LM
2. Add v2 smoke tests and fixtures
3. Refactor v2 model architecture
4. Add v2 training CLI and dry run
5. Add checkpoint resume and scheduler
6. Refactor evaluation CLI
7. Add metrics plotting and result docs
8. Document v2 final results

## 18. Assumptions And Defaults

- Main v2 should stay from-scratch.
- Do not use a pretrained model for the primary result.
- RTX 3090 is the target hardware.
- Use 512 context because current shards are already shaped that way.
- Use fixed train/val/test split:
    - train 0-54
    - validation 55-57
    - test 58-60

- Main v2 config is 12x768, not the current 18x1280.
- Final 3B-token run is complete; the 5.60B stretch checkpoint has also been evaluated on the same full held-out split and is the best v2 perplexity result.
- Robust comparison checks are complete for v2 5.60B versus GPT-2 small: fixed generation metrics, WikiText-2 perplexity, and capped LAMBADA-style scoring.
- Success is honest measurable learning, not impressive chatbot output.
- v1 should be described as a rough prototype.
- If metrics are weak, document them honestly and emphasize engineering quality.
- V3 should target beating GPT-2 more broadly, not only narrowing the in-domain held-out perplexity gap.
