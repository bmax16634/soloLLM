# Model Naming

SoloLLM model names use this format:

```text
<ModelFamily>-<SeriesVersion>-<Size>-<Variant>
```

Example:

```text
SoloLLM-1.0-152M-Base
```

## Fields

| Field | Meaning |
| --- | --- |
| `ModelFamily` | Public model family. Use `SoloLLM`. |
| `SeriesVersion` | Public release/research line. Same architecture with better data/training increments minor versions. Major architecture changes increment the major version. |
| `Size` | Rounded parameter count, such as `57M`, `123M`, or `152M`. |
| `Variant` | Model type, such as `Base`, `Instruct`, `Chat`, or `BaseProxy`. |

## Version Rules

| Change | Version impact |
| --- | --- |
| Better data mix, longer training, new hyperparameters, same architecture | Minor version, for example `1.0 -> 1.1` |
| Same release line but different parameter size | Same version with different size, for example `SoloLLM-1.0-123M-Base` and `SoloLLM-1.0-152M-Base` |
| Small proxy run used to test a recipe | Use `BaseProxy`, not `Base` |
| Instruction tuning or chat tuning | Use `Instruct` or `Chat` |
| Major architecture/tokenizer/checkpoint-format change | Major version, for example `1.x -> 2.0` |

## Current Naming Map

| Historical name | Proposed standard name | Status | Notes |
| --- | --- | --- | --- |
| SoloLLM v2 92M / 5.6B-token diagnostic | `SoloLLM-0.2-92M-Base` | Historical | Early diagnostic base model. |
| SoloLLM v3 123M | `SoloLLM-0.3-123M-Base` | Historical | Smaller GPT-2-scale base run. |
| SoloLLM v3 150M | `SoloLLM-0.3-152M-Base` | Historical | Earlier 152M base run. |
| SoloLLM v4 reasoning 150M | `SoloLLM-0.4-152M-Base` | Historical | Base model trained with a reasoning-heavy data mix. |
| SoloLLM v4.2 123M | `SoloLLM-1.0-123M-Base` | Current full model | Smaller GPT-2-scale comparison model. |
| SoloLLM v4.2 150M / 151.9M | `SoloLLM-1.0-152M-Base` | Current best full model | Strongest completed base checkpoint. |
| SoloLLM v4.2 57M proxy | `SoloLLM-0.4.2-57M-BaseProxy` | Historical proxy | Small proxy for recipe validation. |
| SoloLLM v5 smol-quality 57M proxy | `SoloLLM-1.1-57M-BaseProxy` | Current research proxy | First 1.1 proxy aimed at SmolLM2-style quality. |
| SoloLLM v5.1 high-quality-web proxy | `SoloLLM-1.1-57M-BaseProxy-HighQualityWeb` | Planned/current research proxy | Data-mix proxy emphasizing high-quality web and education text. |
| SoloLLM v5.2 math/edu proxy | `SoloLLM-1.1-57M-BaseProxy-MathEdu` | Planned/current research proxy | Data-mix proxy emphasizing math and education text. |

## Public Names

Use these names in public-facing docs for completed models:

```text
SoloLLM-1.0-152M-Base
SoloLLM-1.0-123M-Base
```

Use these names for current proxy research:

```text
SoloLLM-1.1-57M-BaseProxy
SoloLLM-1.1-57M-BaseProxy-HighQualityWeb
SoloLLM-1.1-57M-BaseProxy-MathEdu
```

## Historical Artifact Paths

Old output directories and run names should not be renamed until a separate
artifact manifest exists. Paths such as `outputs/sologpt_v4/...` and old
`v4_2...` run names refer to completed local artifacts and are kept for
traceability.

## Hugging Face Follow-Up

Create clean v1.0 Hugging Face repos for the two completed public base models:

```text
bmax16634/solollm-1.0-152m-base
bmax16634/solollm-1.0-123m-base
```

If a public demo is kept, use:

```text
bmax16634/solollm-1.0-demo
```

Each model repo should be slim and contain only files needed to load, inspect,
and cite the model:

```text
README.md
config.json
configuration_solollm.py
modeling_solollm.py
model.safetensors
tokenizer.json
tokenizer_config.json
requirements.txt
metrics_summary.json
```

Do not publish training scripts, local outputs, notebooks, personal notes, or
research scratch files in the Hugging Face model repos.

The old `sologpt-v3-*` Hugging Face repos should be treated as historical
artifacts. Do not rename them in place until the clean `solollm-1.0-*` repos
are published and verified, because renaming old repos can break existing
links.
