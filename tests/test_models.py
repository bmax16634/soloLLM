import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from solollm.model import SoloLLM


FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "v2_tiny_config.json"
MODERN_FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "v2_tiny_modern_config.json"
GQA_FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "v2_tiny_gqa_config.json"
MODERN_SMALL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "solollm-0.2-92m-base.json"
)
PROXY_V3_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "solollm-0.4-57m-v3-style-baseproxy.json"
)
PROXY_SMOLLM2_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "solollm-0.4-57m-smollm2-style-baseproxy.json"
)


def load_tiny_config():
    with open(FIXTURE_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def load_modern_tiny_config():
    with open(MODERN_FIXTURE_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gqa_tiny_config():
    with open(GQA_FIXTURE_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def test_v2_forward_shape():
    config = load_tiny_config()
    model = SoloLLM(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, config["seq_length"]))

    logits = model(input_ids)

    assert logits.shape == (2, config["seq_length"], config["vocab_size"])


def test_v2_cached_forward_matches_full_forward():
    config = load_tiny_config()
    model = SoloLLM(config).eval()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    with torch.no_grad():
        full_logits = model(input_ids)
        past_key_values = None
        cached_logits = []
        for position in range(input_ids.size(1)):
            logits, past_key_values = model.forward_with_cache(
                input_ids[:, position : position + 1],
                past_key_values=past_key_values,
            )
            cached_logits.append(logits)

    assert torch.allclose(torch.cat(cached_logits, dim=1), full_logits, atol=1e-5, rtol=1e-5)


def test_v2_cached_forward_matches_full_forward_with_gqa():
    config = load_gqa_tiny_config()
    model = SoloLLM(config).eval()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    with torch.no_grad():
        full_logits = model(input_ids)
        past_key_values = None
        cached_logits = []
        for position in range(input_ids.size(1)):
            logits, past_key_values = model.forward_with_cache(
                input_ids[:, position : position + 1],
                past_key_values=past_key_values,
            )
            cached_logits.append(logits)

    assert torch.allclose(torch.cat(cached_logits, dim=1), full_logits, atol=1e-5, rtol=1e-5)


def test_v2_preallocated_cached_forward_matches_full_forward():
    config = load_tiny_config()
    model = SoloLLM(config).eval()
    input_ids = torch.randint(0, config["vocab_size"], (2, 8))

    with torch.no_grad():
        full_logits = model(input_ids)
        past_key_values = model.allocate_kv_cache(
            batch_size=input_ids.size(0),
            max_length=input_ids.size(1),
            device=input_ids.device,
            dtype=model.input_embed.weight.dtype,
        )
        cached_logits = []
        for position in range(input_ids.size(1)):
            logits, past_key_values = model.forward_with_cache(
                input_ids[:, position : position + 1],
                past_key_values=past_key_values,
                cache_position=position,
            )
            cached_logits.append(logits)

    assert torch.allclose(torch.cat(cached_logits, dim=1), full_logits, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 cache smoke test")
def test_v2_cached_forward_supports_cuda_fp16():
    config = load_tiny_config()
    model = SoloLLM(config).cuda().half().eval()
    input_ids = torch.randint(0, config["vocab_size"], (1, 8), device="cuda")

    with torch.no_grad():
        logits, past_key_values = model.forward_with_cache(input_ids[:, :4])
        logits, past_key_values = model.forward_with_cache(input_ids[:, 4:5], past_key_values=past_key_values)

    assert logits.dtype == torch.float16
    assert len(past_key_values) == config["n_layer"]


def test_v2_rejects_too_long_sequence():
    config = load_tiny_config()
    model = SoloLLM(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, config["max_seq_len"] + 1))

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(input_ids)


def test_v2_tied_embeddings_share_storage():
    config = load_tiny_config()
    config["tie_weights"] = True
    model = SoloLLM(config)

    assert model.input_embed.weight.data_ptr() == model.decoder.lm_head.weight.data_ptr()


def test_v2_parameter_count_positive():
    model = SoloLLM(load_tiny_config())

    assert model.num_parameters() > 0
    assert model.summary()["parameter_count"] == model.num_parameters()


def test_v2_modern_path_forward_shape_and_bias_policy():
    config = load_modern_tiny_config()
    model = SoloLLM(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, config["seq_length"]))

    logits = model(input_ids)
    summary = model.summary()

    assert logits.shape == (2, config["seq_length"], config["vocab_size"])
    assert summary["norm_type"] == "rmsnorm"
    assert summary["mlp_type"] == "swiglu"
    assert summary["use_bias"] is False
    assert all(module.bias is None for module in model.modules() if isinstance(module, nn.Linear))


def test_v2_gqa_path_forward_shape_and_summary():
    config = load_gqa_tiny_config()
    model = SoloLLM(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, config["seq_length"]))

    logits = model(input_ids)
    summary = model.summary()
    first_attn = model.decoder.layers[0].attn

    assert logits.shape == (2, config["seq_length"], config["vocab_size"])
    assert summary["n_head"] == 4
    assert summary["n_kv_head"] == 2
    assert first_attn.qkv is None
    assert first_attn.q_proj is not None
    assert first_attn.k_proj is not None
    assert first_attn.v_proj is not None


def test_v2_rejects_invalid_gqa_ratio():
    config = load_gqa_tiny_config()
    config["n_kv_head"] = 3

    with pytest.raises(ValueError, match="must be divisible by n_kv_head"):
        SoloLLM(config)


def test_v2_modern_small_config_parameter_count_target():
    with open(MODERN_SMALL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = SoloLLM(config)
    parameter_count = model.num_parameters()

    assert 85_000_000 <= parameter_count <= 100_000_000
    assert parameter_count < 123_616_512


def test_phase1_proxy_configs_are_size_matched():
    with open(PROXY_V3_CONFIG, "r", encoding="utf-8") as f:
        v3_config = json.load(f)
    with open(PROXY_SMOLLM2_CONFIG, "r", encoding="utf-8") as f:
        smollm2_config = json.load(f)

    v3_model = SoloLLM(v3_config)
    smollm2_model = SoloLLM(smollm2_config)
    v3_params = v3_model.num_parameters()
    smollm2_params = smollm2_model.num_parameters()

    assert 45_000_000 <= v3_params <= 70_000_000
    assert 45_000_000 <= smollm2_params <= 70_000_000
    assert abs(v3_params - smollm2_params) / max(v3_params, smollm2_params) <= 0.15
    assert smollm2_model.summary()["n_kv_head"] < smollm2_model.summary()["n_head"]
