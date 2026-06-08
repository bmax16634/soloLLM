# Phase 1 50M Sanity Results

## v2-gpt2-parity

Status: complete

Run directory:

```text
outputs/sologpt_v2/sanity_50m_parity
```

Command:

```bash
nice -n 10 ionice -c2 -n7 python -m sologpt_v2.pretrain \
  --config sologpt_v2/config.json \
  --shard-dir /home/bmx/_projects/soloLLM/data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/sanity_50m_parity \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 50000000 \
  --max-eval-tokens 5000000 \
  --no-progress
```

Summary:

| Metric | Value |
| --- | ---: |
| Parameters | 123,616,512 |
| Tokens | 50,003,968 |
| Optimizer steps | 1,526 |
| Final train loss | 5.086976528167725 |
| Capped validation loss | 4.987855458571241 |
| Capped validation perplexity | 146.6216498912708 |
| Validation tokens | 5,003,712 |
| Total time | 1,075.78 sec |
| Average throughput | 46,481 tok/s |
| Peak GPU memory | 10.60GB |
| Output size | about 1.9GB |

Notes:

- Training used only `shard_00000.pt` before reaching the 50M token target.
- GPU utilization stayed near 100%.
- CPU stayed near one core with low process priority.
- Loss dropped from the initial logged 9.526 to about 5.087 final train loss.
- Completion notification was sent through Ariya reminder id `a3f2efeb-7402-4072-b1d6-aefa2b7147f0`.

## v2-modern-small

Status: complete

Run directory:

```text
outputs/sologpt_v2/sanity_50m_modern_small
```

Command:

```bash
nice -n 10 ionice -c2 -n7 python -m sologpt_v2.pretrain \
  --config sologpt_v2/config_modern_small.json \
  --shard-dir /home/bmx/_projects/soloLLM/data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/sanity_50m_modern_small \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 50000000 \
  --max-eval-tokens 5000000 \
  --no-progress
```

Summary:

| Metric | Value |
| --- | ---: |
| Parameters | 91,654,400 |
| Tokens | 50,003,968 |
| Optimizer steps | 1,526 |
| Final train loss | 4.9111480712890625 |
| Capped validation loss | 4.854336679371354 |
| Capped validation perplexity | 128.29556189105713 |
| Validation tokens | 5,003,712 |
| Total time | 981.93 sec |
| Average throughput | 50,924 tok/s |
| Peak GPU memory | 10.47GB |
| Output size | about 1.4GB |

Notes:

- Training used only `shard_00000.pt` before reaching the 50M token target.
- GPU utilization stayed near 100%.
- CPU stayed near one core with low process priority.
- Loss dropped from the initial logged 9.579 to about 4.911 final train loss.
- Completion notification was sent through Ariya reminder id `abf337e7-e269-4262-9b46-83a8c60bca7e`.

## Initial Comparison

| Metric | v2-gpt2-parity | v2-modern-small | Better |
| --- | ---: | ---: | --- |
| Parameters | 123,616,512 | 91,654,400 | modern-small |
| Final train loss | 5.086976528167725 | 4.9111480712890625 | modern-small |
| Capped validation loss | 4.987855458571241 | 4.854336679371354 | modern-small |
| Capped validation PPL | 146.6216498912708 | 128.29556189105713 | modern-small |
| Total time | 1,075.78 sec | 981.93 sec | modern-small |
| Average throughput | 46,481 tok/s | 50,924 tok/s | modern-small |
| Peak GPU memory | 10.60GB | 10.47GB | modern-small |
| Output size | about 1.9GB | about 1.4GB | modern-small |

Early decision:

- `v2-modern-small` is the stronger 300M pilot candidate based on the 50M sanity runs.
- This is not final proof that it will beat GPT-2-small; it is enough evidence to prioritize it for the 300M pilot.
