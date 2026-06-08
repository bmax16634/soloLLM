import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generation_metrics_cli_writes_summary(tmp_path):
    input_path = tmp_path / "generations.json"
    output_json = tmp_path / "metrics.json"
    output_md = tmp_path / "metrics.md"
    input_path.write_text(
        json.dumps(
            {
                "settings": {"seed": 1},
                "models": {},
                "samples": [
                    {
                        "prompt_id": "repeat",
                        "category": "stability",
                        "prompt": "The answer is",
                        "completions": [
                            {
                                "model": "v2",
                                "text": "The answer is yes yes yes yes yes yes",
                                "new_tokens": 6,
                            },
                            {
                                "model": "gpt2",
                                "text": "The answer is clear and concise.",
                                "new_tokens": 4,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.generation_metrics",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))

    assert "v2" in result["summary"]
    assert "gpt2" in result["summary"]
    assert result["summary"]["v2"]["bad_loop_count"] == 1
    assert output_md.exists()
