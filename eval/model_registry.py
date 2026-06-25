"""Shared model/tokenizer loading helpers for evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

from eval.eval import load_checkpoint_state, load_json
from sologpt_v2.model import SoloGPT_v2


@dataclass
class LoadedEvalModel:
    label: str
    loader: str
    model: torch.nn.Module
    tokenizer: Any
    metadata: dict[str, Any]


def parse_model_spec(spec: str, *, v2_label: str) -> tuple[str, str, str | None]:
    """Return label, loader, and HF model id if present.

    Supported forms:
    - v2
    - gpt2
    - distilgpt2
    - hf:EleutherAI/pythia-160m
    - pythia-160m=hf:EleutherAI/pythia-160m
    """

    label_override: str | None = None
    raw_spec = spec
    if "=" in spec:
        label_override, raw_spec = spec.split("=", maxsplit=1)
        label_override = label_override.strip()
        raw_spec = raw_spec.strip()
        if not label_override:
            raise ValueError(f"invalid model spec {spec!r}: empty label before '='")
        if not raw_spec:
            raise ValueError(f"invalid model spec {spec!r}: empty loader after '='")

    if raw_spec == "v2":
        return label_override or v2_label, "v2", None
    if raw_spec == "gpt2":
        return label_override or "gpt2", "gpt2", "gpt2"
    if raw_spec.startswith("hf:"):
        model_id = raw_spec.removeprefix("hf:")
        if not model_id:
            raise ValueError(f"invalid model spec {spec!r}: empty HF model id")
        return label_override or model_id.rsplit("/", maxsplit=1)[-1], "hf", model_id

    # Convenience: treat bare non-local names as Hugging Face model ids.
    return label_override or raw_spec.rsplit("/", maxsplit=1)[-1], "hf", raw_spec


def infer_context_length(model: torch.nn.Module, tokenizer: Any) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("n_positions", "max_position_embeddings", "seq_length", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 1_000_000:
        return value
    return None


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def prepare_tokenizer(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_eval_model(
    spec: str,
    *,
    device: torch.device,
    v2_checkpoint: Path,
    v2_config: Path,
    v2_label: str,
    trust_remote_code: bool = False,
) -> LoadedEvalModel:
    label, loader, hf_model_id = parse_model_spec(spec, v2_label=v2_label)

    if loader == "v2":
        config = load_json(v2_config)
        model = SoloGPT_v2(config).to(device)
        state = load_checkpoint_state(v2_checkpoint, device)
        model.load_state_dict(state)
        tokenizer = prepare_tokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
        context_length = int(config.get("max_seq_len", config.get("seq_length", config.get("n_positions", 0))))
        return LoadedEvalModel(
            label=label,
            loader="v2",
            model=model,
            tokenizer=tokenizer,
            metadata={
                "model": "v2",
                "checkpoint": str(v2_checkpoint),
                "config": str(v2_config),
                "tokenizer": "gpt2",
                "parameter_count": parameter_count(model),
                "native_context_length": context_length,
            },
        )

    if hf_model_id is None:
        raise ValueError(f"model spec {spec!r} did not resolve to a loadable model")

    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code))
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code).to(device)
    context_length = infer_context_length(model, tokenizer)
    return LoadedEvalModel(
        label=label,
        loader="gpt2" if hf_model_id == "gpt2" else "hf",
        model=model,
        tokenizer=tokenizer,
        metadata={
            "model": hf_model_id,
            "tokenizer": getattr(tokenizer, "name_or_path", hf_model_id),
            "parameter_count": parameter_count(model),
            "native_context_length": context_length,
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
        },
    )


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids)
    return outputs.logits if hasattr(outputs, "logits") else outputs
