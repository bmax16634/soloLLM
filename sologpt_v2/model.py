"""SoloGPT v2 model.

This is a small GPT-style decoder-only transformer intended to be trained
from scratch. The public contract stays simple:

    model = SoloGPT_v2(config)
    logits = model(input_ids)

where input_ids has shape (batch, sequence).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of unique parameters in a module."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to the last dimension of q/k tensors."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """RoPE cache for attention query/key tensors shaped (B, H, T, D)."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        inv_freq = 1.0 / (float(base) ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))
        self.cos_cached = freqs.cos().to(dtype=dtype)
        self.sin_cached = freqs.sin().to(dtype=dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cache_missing = self.cos_cached.numel() == 0
        cache_too_short = cache_missing or self.cos_cached.size(0) < seq_len
        cache_wrong_device = cache_missing or self.cos_cached.device != q.device
        cache_wrong_dtype = cache_missing or self.cos_cached.dtype != q.dtype

        if cache_too_short or cache_wrong_device or cache_wrong_dtype:
            self._build_cache(seq_len, q.device, q.dtype)

        cos = self.cos_cached[:seq_len][None, None, :, :]
        sin = self.sin_cached[:seq_len][None, None, :, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class CausalSelfAttentionRoPE(nn.Module):
    """Multi-head causal self-attention with RoPE on query/key states."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        rope_base: float = 10000.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"n_embd ({embed_dim}) must be divisible by n_head ({n_heads})")

        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.embed_dim // self.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head dimension, got {self.head_dim} "
                f"from n_embd={embed_dim}, n_head={n_heads}"
            )

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bool(qkv_bias))
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bool(proj_bias))
        self.proj._is_residual_projection = True
        self.dropout = float(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        return probs @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn = self._manual_attention(q, k, v)

        y = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_drop(self.proj(y))


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by modern decoder-only LMs."""

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def make_norm(norm_type: str, embed_dim: int) -> nn.Module:
    normalized = norm_type.lower()
    if normalized == "layernorm":
        return nn.LayerNorm(embed_dim)
    if normalized == "rmsnorm":
        return RMSNorm(embed_dim)
    raise ValueError(f"unsupported norm_type={norm_type!r}; expected 'layernorm' or 'rmsnorm'")


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        *,
        mlp_type: str = "gelu",
        use_bias: bool | None = None,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mlp_type = mlp_type.lower()
        linear_bias = True if use_bias is None else bool(use_bias)

        if self.mlp_type == "gelu":
            hidden_dim = int(mlp_hidden_dim or 4 * embed_dim)
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=linear_bias),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim, bias=linear_bias),
                nn.Dropout(dropout),
            )
            self.net[2]._is_residual_projection = True
        elif self.mlp_type == "swiglu":
            hidden_dim = int(mlp_hidden_dim or (8 * embed_dim / 3))
            self.w1 = nn.Linear(embed_dim, hidden_dim, bias=linear_bias)
            self.w3 = nn.Linear(embed_dim, hidden_dim, bias=linear_bias)
            self.w2 = nn.Linear(hidden_dim, embed_dim, bias=linear_bias)
            self.w2._is_residual_projection = True
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"unsupported mlp_type={mlp_type!r}; expected 'gelu' or 'swiglu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            return self.net(x)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        rope_base: float,
        qkv_bias: bool,
        norm_type: str,
        mlp_type: str,
        use_bias: bool | None,
        mlp_hidden_dim: int | None,
    ) -> None:
        super().__init__()
        proj_bias = False if use_bias is None else bool(use_bias)
        self.norm1 = make_norm(norm_type, embed_dim)
        self.attn = CausalSelfAttentionRoPE(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            rope_base=rope_base,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.norm2 = make_norm(norm_type, embed_dim)
        self.ff = FeedForward(
            embed_dim,
            dropout,
            mlp_type=mlp_type,
            use_bias=use_bias,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
        rope_base: float,
        qkv_bias: bool,
        norm_type: str,
        mlp_type: str,
        use_bias: bool | None,
        mlp_hidden_dim: int | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    rope_base=rope_base,
                    qkv_bias=qkv_bias,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    use_bias=use_bias,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = make_norm(norm_type, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class SoloGPT_v2(nn.Module):
    """Decoder-only GPT-style model for SoloLLM v2."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.model_type = "Transformer"

        self.embed_dim = int(config["n_embd"])
        self.n_heads = int(config["n_head"])
        self.n_layers = int(config["n_layer"])
        self.dropout = float(config.get("dropout", 0.1))
        self.vocab_size = int(config["vocab_size"])
        self.max_seq_len = int(
            config.get("max_seq_len", config.get("seq_length", config.get("n_positions", 512)))
        )
        self.rope_base = float(config.get("rope_base", 10000.0))
        self.tie_weights = bool(config.get("tie_weights", True))
        self.qkv_bias = bool(config.get("qkv_bias", False))
        self.norm_type = str(config.get("norm_type", "layernorm")).lower()
        self.mlp_type = str(config.get("mlp_type", "gelu")).lower()
        raw_use_bias = config.get("use_bias", None)
        self.use_bias = None if raw_use_bias is None else bool(raw_use_bias)
        self.mlp_hidden_dim = (
            int(config["mlp_hidden_dim"]) if config.get("mlp_hidden_dim") is not None else None
        )
        self.init_std = float(config.get("init_std", 0.02))
        self.scale_residual_init = bool(config.get("scale_residual_init", False))

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.decoder = DecoderBlock(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            rope_base=self.rope_base,
            qkv_bias=self.qkv_bias,
            norm_type=self.norm_type,
            mlp_type=self.mlp_type,
            use_bias=self.use_bias,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )

        self.apply(self._init_weights)

        if self.tie_weights:
            self.decoder.lm_head.weight = self.input_embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.init_std
            if self.scale_residual_init and getattr(module, "_is_residual_projection", False):
                std = std / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape (batch, seq), got {tuple(input_ids.shape)}")

        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.input_embed(input_ids)
        return self.decoder(x)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    def summary(self) -> dict[str, int | float | bool | str | None]:
        return {
            "model_type": self.model_type,
            "parameter_count": self.num_parameters(),
            "trainable_parameter_count": self.num_parameters(trainable_only=True),
            "vocab_size": self.vocab_size,
            "n_embd": self.embed_dim,
            "n_layer": self.n_layers,
            "n_head": self.n_heads,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "rope_base": self.rope_base,
            "tie_weights": self.tie_weights,
            "qkv_bias": self.qkv_bias,
            "norm_type": self.norm_type,
            "mlp_type": self.mlp_type,
            "use_bias": self.use_bias,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "init_std": self.init_std,
            "scale_residual_init": self.scale_residual_init,
        }
