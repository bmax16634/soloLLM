#!/usr/bin/env python3
"""Benchmark practical local generation speed and VRAM for v5 baseline models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval import load_checkpoint_state, load_json, resolve_device
from sologpt_v2.model import SoloGPT_v2


OUT_DIR = ROOT / "outputs" / "eval_suites" / "v5_inference_benchmark"
JSON_PATH = OUT_DIR / "inference_benchmark.json"
REPORT_PATH = ROOT / "docs" / "results" / "v5_inference_benchmark.md"


PROMPTS = [
    "In simple terms, photosynthesis is",
    "The Industrial Revolution changed society by",
    "Three practical ways to save energy at home are:",
]


@dataclass(frozen=True)
class BenchSpec:
    label: str
    kind: str
    checkpoint: Path | None = None
    config: Path | None = None
    hf_id: str | None = None


DEFAULT_SPECS = [
    BenchSpec(
        label="SoloLLM v4.2 150M",
        kind="local_v2",
        checkpoint=ROOT / "outputs/sologpt_v4/v4_2_dropout0_v4_1_blend_10b_150m_1024/final_model.pt",
        config=ROOT / "sologpt_v4/config_v4_2_dropout0_v4_1_blend_150m_1024.json",
    ),
    BenchSpec(
        label="SoloLLM v4.2 123M",
        kind="local_v2",
        checkpoint=ROOT / "outputs/sologpt_v4/v4_2_dropout0_v4_1_blend_10b_123m_1024/final_model.pt",
        config=ROOT / "sologpt_v4/config_v4_2_dropout0_v4_1_blend_123m_1024.json",
    ),
    BenchSpec(label="GPT-2 small", kind="hf", hf_id="gpt2"),
    BenchSpec(label="Pythia 160M", kind="hf", hf_id="EleutherAI/pythia-160m-deduped"),
    BenchSpec(label="Pythia 410M", kind="hf", hf_id="EleutherAI/pythia-410m-deduped"),
    BenchSpec(label="SmolLM2 135M", kind="hf", hf_id="HuggingFaceTB/SmolLM2-135M"),
    BenchSpec(label="SmolLM2 360M", kind="hf", hf_id="HuggingFaceTB/SmolLM2-360M"),
    BenchSpec(label="TinyLlama 1.1B", kind="hf", hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--warmup-new-tokens", type=int, default=16)
    parser.add_argument("--output-json", default=str(JSON_PATH))
    parser.add_argument("--output-md", default=str(REPORT_PATH))
    parser.add_argument("--models", nargs="*", default=None, help="Optional labels to benchmark")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def cuda_memory_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def prepare_tokenizer(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_spec(
    spec: BenchSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    if spec.kind == "local_v2":
        if spec.checkpoint is None or spec.config is None:
            raise ValueError(f"{spec.label} missing checkpoint/config")
        config = load_json(spec.config)
        model = SoloGPT_v2(config)
        state = load_checkpoint_state(spec.checkpoint, torch.device("cpu"))
        model.load_state_dict(state)
        model = model.to(device=device, dtype=dtype).eval()
        tokenizer = prepare_tokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
        context_length = int(config.get("max_seq_len", config.get("seq_length", 1024)))
        metadata = {
            "kind": spec.kind,
            "checkpoint": str(spec.checkpoint),
            "config": str(spec.config),
            "parameter_count": parameter_count(model),
            "context_length": context_length,
            "generation_path": "manual_no_kv_cache",
        }
        return model, tokenizer, metadata

    if spec.kind == "hf":
        if spec.hf_id is None:
            raise ValueError(f"{spec.label} missing hf_id")
        tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=trust_remote_code))
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()
        context_length = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if not isinstance(context_length, int):
            context_length = getattr(getattr(model, "config", None), "n_positions", None)
        metadata = {
            "kind": spec.kind,
            "hf_id": spec.hf_id,
            "parameter_count": parameter_count(model),
            "context_length": context_length,
            "generation_path": "transformers_generate_use_cache",
        }
        return model, tokenizer, metadata

    raise ValueError(f"unknown benchmark kind: {spec.kind}")


@torch.no_grad()
def generate_local_no_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
) -> int:
    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        window = generated[:, -context_length:]
        outputs = model(window)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return max_new_tokens


@torch.no_grad()
def generate_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> int:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = int(encoded.input_ids.numel())
    output = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return int(output.numel()) - input_tokens


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_spec(
    spec: BenchSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    warmup_new_tokens: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model, tokenizer, metadata = load_spec(spec, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
    synchronize(device)
    load_elapsed = time.perf_counter() - load_start
    model_memory = cuda_memory_gb()

    context_length = int(metadata.get("context_length") or 1024)
    warmup_prompt = PROMPTS[0]
    if warmup_new_tokens > 0:
        if metadata["generation_path"] == "manual_no_kv_cache":
            generate_local_no_cache(
                model,
                tokenizer,
                warmup_prompt,
                device=device,
                context_length=context_length,
                max_new_tokens=warmup_new_tokens,
            )
        else:
            generate_hf(model, tokenizer, warmup_prompt, device=device, max_new_tokens=warmup_new_tokens)
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_new_tokens = 0
    start = time.perf_counter()
    for prompt in PROMPTS:
        if metadata["generation_path"] == "manual_no_kv_cache":
            total_new_tokens += generate_local_no_cache(
                model,
                tokenizer,
                prompt,
                device=device,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )
        else:
            total_new_tokens += generate_hf(model, tokenizer, prompt, device=device, max_new_tokens=max_new_tokens)
        synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory = cuda_memory_gb()

    result = {
        "label": spec.label,
        "metadata": metadata,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "prompts": len(PROMPTS),
        "max_new_tokens_per_prompt": max_new_tokens,
        "total_new_tokens": total_new_tokens,
        "load_elapsed_sec": load_elapsed,
        "generation_elapsed_sec": elapsed,
        "tokens_per_sec": total_new_tokens / elapsed if elapsed > 0 else None,
        "model_memory": model_memory,
        "generation_peak_memory": peak_memory,
    }

    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fmt_params(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1_000_000:.2f}M"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V5 Inference Benchmark",
        "",
        "Date: July 20, 2026",
        "",
        "## Method",
        "",
        "Batch size `1`, greedy generation, three fixed prompts, and `80` measured",
        "new tokens per prompt by default. SoloLLM checkpoints use the current",
        "manual generation loop without KV cache. Hugging Face baselines use",
        "`transformers.generate(..., use_cache=True)`. This measures practical",
        "local usability of the current implementations, not pure architecture",
        "throughput under identical kernels.",
        "",
        "## Results",
        "",
        "| Model | Params | Path | DType | Tok/s | Peak VRAM | Load sec |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        metadata = row["metadata"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    fmt_params(metadata.get("parameter_count")),
                    metadata.get("generation_path", "n/a"),
                    row["dtype"],
                    f"{row['tokens_per_sec']:.2f}",
                    f"{row['generation_peak_memory']['peak_allocated_gb']:.2f}GB",
                    f"{row['load_elapsed_sec']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- SoloLLM numbers reflect the current no-KV-cache generation path and are",
            "  expected to understate what the architecture could do with a cached",
            "  inference implementation.",
            "- Hugging Face model numbers reflect the optimized local path available",
            "  through `transformers.generate`.",
            "- Use this board for practical local planning: model size, memory headroom,",
            "  and whether implementing a SoloLLM KV cache should be a v5 priority.",
            "- Quantized benchmarking is not included yet because local bitsandbytes",
            "  CUDA backend loading currently fails with missing `libnvJitLink.so.13`.",
            "",
            f"Raw JSON: `{Path(payload['output_json']).relative_to(ROOT)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    selected = set(args.models or [])
    specs = [spec for spec in DEFAULT_SPECS if not selected or spec.label in selected]
    if not specs:
        raise SystemExit("no models selected")

    results = []
    for spec in specs:
        print(f"benchmarking {spec.label}")
        results.append(
            benchmark_spec(
                spec,
                device=device,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
                warmup_new_tokens=args.warmup_new_tokens,
                trust_remote_code=args.trust_remote_code,
            )
        )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    payload = {
        "settings": {
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "prompts": PROMPTS,
            "max_new_tokens": args.max_new_tokens,
            "warmup_new_tokens": args.warmup_new_tokens,
        },
        "results": results,
        "output_json": str(output_json),
        "output_md": str(output_md),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(output_md, payload)
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
