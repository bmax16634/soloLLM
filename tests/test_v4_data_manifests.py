from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    (REPO_ROOT / "sologpt_v4" / "data_mix_quality_web_300m.yaml", 300_000_000),
    (REPO_ROOT / "sologpt_v4" / "data_mix_smollm_reasoning_300m.yaml", 300_000_000),
    (REPO_ROOT / "sologpt_v4" / "data_mix_smollm_reasoning_10b.yaml", 10_000_000_000),
]


def test_v4_data_mix_weights_and_text_fields():
    for path, target_tokens in MANIFESTS:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        enabled = [source for source in manifest["sources"] if source.get("enabled", True)]
        disabled = [source for source in manifest["sources"] if not source.get("enabled", True)]

        assert manifest["target_tokens"] == target_tokens
        assert manifest["seq_length"] == 1024
        assert abs(sum(float(source["weight"]) for source in enabled) - 1.0) < 1e-9
        assert all(source.get("text_field") == "text" for source in enabled)
        assert all(float(source.get("weight", 0.0)) == 0.0 for source in disabled)
