import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from sologpt_v2.model import SoloGPT_v2
from eval.model_registry import parse_model_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "v2_tiny_config.json"


def run_command(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_eval_cli_writes_finite_v2_result(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True)
    torch.manual_seed(123)
    torch.save(torch.randint(0, 64, (4, 16), dtype=torch.long), shard_dir / "shard_00000.pt")

    with open(FIXTURE_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = SoloGPT_v2(config)
    checkpoint_path = tmp_path / "tiny_v2.pt"
    torch.save(model.state_dict(), checkpoint_path)

    output_path = tmp_path / "eval.json"
    run_command(
        [
            sys.executable,
            "eval/eval.py",
            "--model",
            "v2",
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(FIXTURE_CONFIG),
            "--shard-dir",
            str(shard_dir),
            "--batch-size",
            "2",
            "--output-json",
            str(output_path),
            "--device",
            "cpu",
            "--no-progress",
        ]
    )

    with open(output_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    assert result["model"] == "v2"
    assert result["tokens"] > 0
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["perplexity"])


def test_model_spec_parser_supports_hf_labels():
    assert parse_model_spec("v2", v2_label="candidate") == ("candidate", "v2", None)
    assert parse_model_spec("gpt2", v2_label="candidate") == ("gpt2", "gpt2", "gpt2")
    assert parse_model_spec("hf:EleutherAI/pythia-160m", v2_label="candidate") == (
        "pythia-160m",
        "hf",
        "EleutherAI/pythia-160m",
    )
    assert parse_model_spec("smollm2=hf:HuggingFaceTB/SmolLM2-135M", v2_label="candidate") == (
        "smollm2",
        "hf",
        "HuggingFaceTB/SmolLM2-135M",
    )
