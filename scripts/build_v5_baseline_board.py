#!/usr/bin/env python3
"""Build the v5 baseline board from completed eval-suite JSON artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "eval_suites" / "v5_baseline_board"
REPORT_PATH = ROOT / "docs" / "results" / "v5_baseline_board.md"
JSON_PATH = OUT_DIR / "baseline_board.json"
INFERENCE_JSON = ROOT / "outputs" / "eval_suites" / "v5_inference_benchmark" / "inference_benchmark.json"


MODERN_DIR = ROOT / "outputs" / "eval_suites" / "modern_baselines_v3_150m"
V4_150_DIR = ROOT / "outputs" / "eval_suites" / "v4_2_dropout0_v4_1_blend_10b_150m_gpt2_full_suite"
V4_123_DIR = ROOT / "outputs" / "eval_suites" / "v4_2_dropout0_v4_1_blend_10b_123m_gpt2_full_suite"
MISSING_BASELINES_DIR = ROOT / "outputs" / "eval_suites" / "v5_missing_baselines"


@dataclass(frozen=True)
class Source:
    label: str
    external_json: Path
    mc_json: Path


SOURCES = [
    Source(
        label="modern_baselines_v3_150m",
        external_json=MODERN_DIR / "external_modern_baselines.json",
        mc_json=MODERN_DIR / "multiple_choice_modern_baselines.json",
    ),
    Source(
        label="v4_2_150m_gpt2_suite",
        external_json=V4_150_DIR / "external_full_candidate_gpt2.json",
        mc_json=V4_150_DIR / "multiple_choice_full_candidate_gpt2.json",
    ),
    Source(
        label="v4_2_123m_gpt2_suite",
        external_json=V4_123_DIR / "external_full_candidate_gpt2.json",
        mc_json=V4_123_DIR / "multiple_choice_full_candidate_gpt2.json",
    ),
    Source(
        label="v5_missing_baselines",
        external_json=MISSING_BASELINES_DIR / "external_missing_baselines.json",
        mc_json=MISSING_BASELINES_DIR / "multiple_choice_missing_baselines.json",
    ),
]


DISPLAY = {
    "v4_2_dropout0_v4_1_blend_10b_150m": "SoloLLM v4.2 150M",
    "v4_2_dropout0_v4_1_blend_10b_123m": "SoloLLM v4.2 123M",
    "solollm-v3-150m": "SoloLLM v3 150M",
    "gpt2": "GPT-2 small",
    "distilgpt2": "DistilGPT2",
    "pythia-70m": "Pythia 70M",
    "pythia-160m": "Pythia 160M",
    "pythia-410m": "Pythia 410M",
    "smollm-135m": "SmolLM 135M",
    "smollm2-135m": "SmolLM2 135M",
    "smollm2-360m": "SmolLM2 360M",
    "tinyllama-1.1b": "TinyLlama 1.1B",
}

ORDER = [
    "v4_2_dropout0_v4_1_blend_10b_150m",
    "v4_2_dropout0_v4_1_blend_10b_123m",
    "solollm-v3-150m",
    "gpt2",
    "distilgpt2",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "smollm-135m",
    "smollm2-135m",
    "smollm2-360m",
    "tinyllama-1.1b",
]

NOTES = {
    "v4_2_dropout0_v4_1_blend_10b_150m": "Current best SoloLLM quantitative checkpoint.",
    "v4_2_dropout0_v4_1_blend_10b_123m": "Smaller-than-GPT-2 ablation; wins fixed MC suite, not every metric.",
    "solollm-v3-150m": "Public final v3 checkpoint.",
    "gpt2": "Historical GPT-2-class reference baseline.",
    "distilgpt2": "Compressed GPT-2-family reference.",
    "pythia-70m": "Controlled open baseline below SoloLLM scale.",
    "pythia-160m": "Controlled open baseline near SoloLLM 150M scale.",
    "pythia-410m": "Larger controlled baseline; useful direction check, not same-scale.",
    "smollm-135m": "Modern compact baseline.",
    "smollm2-135m": "Primary modern same-scale target.",
    "smollm2-360m": "Modern compact upper-bound target.",
    "tinyllama-1.1b": "Stretch local baseline; quality target, not same-size.",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value == float("inf") or value > 1e100:
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def fmt_params(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / 1_000_000:.2f}M"


def collect() -> dict[str, Any]:
    board: dict[str, dict[str, Any]] = {}
    source_files: dict[str, list[str]] = {}

    for source in SOURCES:
        external = load_json(source.external_json)
        mc = load_json(source.mc_json)
        source_files[source.label] = [str(source.external_json.relative_to(ROOT)), str(source.mc_json.relative_to(ROOT))]

        for model, metadata in external.get("models", {}).items():
            entry = board.setdefault(model, {"model": model})
            entry["display"] = DISPLAY.get(model, model)
            entry["params"] = metadata.get("parameter_count")
            entry.setdefault("sources", []).append(source.label)

        for result in external.get("results", []):
            model = result["model"]
            entry = board.setdefault(model, {"model": model, "display": DISPLAY.get(model, model)})
            benchmark = result.get("benchmark")
            if benchmark == "wikitext2":
                entry["wikitext_bpb"] = result.get("bits_per_byte")
                entry["wikitext_ppl"] = result.get("perplexity")
            elif benchmark == "lambada":
                entry["lambada_bpb"] = result.get("bits_per_byte")
                entry["lambada_ppl"] = result.get("perplexity")
                entry["lambada_token_acc"] = result.get("last_token_accuracy")
                entry["lambada_word_acc"] = result.get("last_word_greedy_exact_accuracy")

        for model, metadata in mc.get("models", {}).items():
            entry = board.setdefault(model, {"model": model})
            entry["display"] = DISPLAY.get(model, model)
            entry["params"] = metadata.get("parameter_count", entry.get("params"))
            entry.setdefault("sources", []).append(source.label)

        mc_rows_by_model: dict[str, list[dict[str, Any]]] = {}
        for result in mc.get("results", []):
            mc_rows_by_model.setdefault(result["model"], []).append(result)
        for model, rows in mc_rows_by_model.items():
            entry = board.setdefault(model, {"model": model, "display": DISPLAY.get(model, model)})
            entry["mc_avg_acc"] = sum(float(row["accuracy"]) for row in rows) / len(rows)
            entry["mc_avg_norm"] = sum(float(row["accuracy_norm"]) for row in rows) / len(rows)
            entry["mc_tasks"] = {row["benchmark"]: row for row in rows}

    ordered = [board[model] for model in ORDER if model in board]
    inference = load_json(INFERENCE_JSON) if INFERENCE_JSON.exists() else None
    return {"models": ordered, "source_files": source_files, "inference": inference}


def markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# V5 Baseline Board")
    lines.append("")
    lines.append("Date: July 20, 2026")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("This is the first v5 baseline board. It reuses completed eval artifacts")
    lines.append("from v3/v4 instead of launching new training.")
    if data.get("inference"):
        lines.append("It includes quality metrics plus the first local inference benchmark.")
        lines.append("Pythia 410M and TinyLlama 1.1B are now included.")
        lines.append("Quantization is blocked locally by a bitsandbytes CUDA library mismatch.")
    else:
        lines.append("It is quality-focused; inference, VRAM, quantization, Pythia 410M,")
        lines.append("and TinyLlama remain pending.")
    lines.append("")
    lines.append("The cross-tokenizer language-model metric to read is BPB")
    lines.append("(bits per byte), not raw token perplexity. PPL is still shown for")
    lines.append("continuity, but it is tokenizer-dependent.")
    lines.append("")
    lines.append("## Headline Board")
    lines.append("")
    lines.append(
        "| Model | Params | WikiText BPB | LAMBADA BPB | LAMBADA tok acc | "
        "LAMBADA word acc | MC avg norm | Read |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in data["models"]:
        model = row["model"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display"],
                    fmt_params(row.get("params")),
                    fmt_float(row.get("wikitext_bpb")),
                    fmt_float(row.get("lambada_bpb")),
                    fmt_pct(row.get("lambada_token_acc")),
                    fmt_pct(row.get("lambada_word_acc")),
                    fmt_pct(row.get("mc_avg_norm")),
                    NOTES.get(model, ""),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Multiple-Choice Detail")
    lines.append("")
    lines.append("| Model | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande | Average |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in data["models"]:
        tasks = row.get("mc_tasks", {})
        values = [
            fmt_pct(tasks.get("hellaswag", {}).get("accuracy_norm")),
            fmt_pct(tasks.get("piqa", {}).get("accuracy_norm")),
            fmt_pct(tasks.get("arc_easy", {}).get("accuracy_norm")),
            fmt_pct(tasks.get("arc_challenge", {}).get("accuracy_norm")),
            fmt_pct(tasks.get("winogrande", {}).get("accuracy_norm")),
            fmt_pct(row.get("mc_avg_norm")),
        ]
        lines.append(f"| {row['display']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- SoloLLM v4.2 150M remains the strongest SoloLLM checkpoint.")
    lines.append("- SoloLLM v4.2 123M is competitive with GPT-2 and Pythia 160M, but")
    lines.append("  SmolLM2 135M is the main same-scale modern target.")
    lines.append("- The largest same-scale gap is still modern-small-model quality:")
    lines.append("  SmolLM2 135M and especially SmolLM2 360M are well ahead on MC and")
    lines.append("  LAMBADA exact-word accuracy.")
    lines.append("- Pythia 410M and TinyLlama 1.1B confirm the expected scaling curve:")
    lines.append("  substantially larger modern baselines are far ahead on BPB and")
    lines.append("  continuation accuracy.")
    if data.get("inference"):
        lines.append("- The board is complete enough to choose v5 direction. The remaining")
        lines.append("  engineering gap is SoloLLM KV-cache inference; quantized benchmarking")
        lines.append("  needs the local bitsandbytes CUDA issue fixed first.")
    else:
        lines.append("- The next engineering work is not training. It is completing v5 eval")
        lines.append("  coverage: missing baselines, inference speed, VRAM, and quantization.")
    lines.append("")
    if data.get("inference"):
        lines.append("## Local Inference")
        lines.append("")
        lines.append("Batch size `1`, greedy generation, FP16, three fixed prompts. SoloLLM")
        lines.append("uses the current manual generation path without KV cache; Hugging Face")
        lines.append("baselines use `transformers.generate` with cache enabled.")
        lines.append("")
        lines.append("| Model | Params | Generation path | Tok/s | Peak VRAM |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for row in data["inference"]["results"]:
            metadata = row["metadata"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["label"],
                        fmt_params(metadata.get("parameter_count")),
                        metadata.get("generation_path", "n/a"),
                        f"{row['tokens_per_sec']:.2f}",
                        f"{row['generation_peak_memory']['peak_allocated_gb']:.2f}GB",
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("Inference report: `docs/results/v5_inference_benchmark.md`")
        lines.append("")
    lines.append("## Pending Board Items")
    lines.append("")
    lines.append("| Item | Status | Reason |")
    lines.append("| --- | --- | --- |")
    lines.append("| Pythia 410M | complete | Included in external quality and multiple-choice board. |")
    lines.append("| TinyLlama 1.1B | complete | Included as stretch local quality baseline. |")
    if data.get("inference"):
        lines.append("| Inference tokens/sec | v0 complete | Covered for the selected practical baseline set. |")
        lines.append("| Peak inference VRAM | v0 complete | Covered for the selected practical baseline set. |")
    else:
        lines.append("| Inference tokens/sec | pending | Needed for local-usefulness comparison. |")
        lines.append("| Peak inference VRAM | pending | Needed before choosing 300M vs 400M scale. |")
    lines.append("| Quantized quality/speed | blocked | `bitsandbytes` fails to load CUDA backend locally: missing `libnvJitLink.so.13`. |")
    lines.append("| SoloLLM KV-cache inference | pending | Requires implementation work; current SoloLLM speed uses manual no-cache generation. |")
    lines.append("")
    lines.append("## Source Artifacts")
    lines.append("")
    for label, paths in data["source_files"].items():
        lines.append(f"- `{label}`")
        for path in paths:
            lines.append(f"  - `{path}`")
    lines.append("")
    lines.append(f"Machine-readable board: `{JSON_PATH.relative_to(ROOT)}`")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = collect()
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n")
    REPORT_PATH.write_text(markdown(data) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
