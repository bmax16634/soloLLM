# model.py
# SoloGPT_v2 with RoPE (rotary positional embeddings) + causal self-attention
# Drop-in replacement for your previous model.py (same forward signature).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# RoPE helpers
# -----------------------------
def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, D) where D is even
    cos: (1, 1, T, D/2)
    sin: (1, 1, T, D/2)
    """
    x1 = x[..., 0::2]  # (B,H,T,D/2)
    x2 = x[..., 1::2]  # (B,H,T,D/2)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """
    Precompute cos/sin cache for RoPE and apply to Q/K.
    head_dim must be even.
    """
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        self.head_dim = head_dim
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._cached_len = 0

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device, dtype):
        # (T,)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # (T, D/2)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        self.cos_cached = cos
        self.sin_cached = sin
        self._cached_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (B, H, T, D)
        """
        T = q.size(-2)
        if (T > self._cached_len) or (self.cos_cached.device != q.device) or (self.cos_cached.dtype != q.dtype):
            self._build_cache(T, q.device, q.dtype)

        cos = self.cos_cached[:T][None, None, :, :]  # (1,1,T,D/2)
        sin = self.sin_cached[:T][None, None, :, :]  # (1,1,T,D/2)

        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


# -----------------------------
# Causal Self-Attention w/ RoPE
# -----------------------------
class CausalSelfAttentionRoPE(nn.Module):
    """
    Decoder-only causal self-attention using RoPE on Q/K.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")

        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.embed_dim // self.n_heads

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim. Got head_dim={self.head_dim} "
                f"(n_embd={embed_dim}, n_head={n_heads}). Choose n_embd/n_head even."
            )

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=max_seq_len)

        # causal mask cache (re-built if T grows)
        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)
        self._mask_len = 0

    def _get_causal_mask(self, T: int, device):
        # (T, T) with True above diagonal
        if (T > self._mask_len) or (self._causal_mask.device != device):
            mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask = mask
            self._mask_len = T
        return self._causal_mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q/K
        q, k = self.rope(q, k)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)

        # causal mask
        causal = self._get_causal_mask(T, x.device)
        att = att.masked_fill(causal, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B,H,T,D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)

        y = self.resid_drop(self.proj(y))
        return y


# -----------------------------
# MLP / FeedForward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),              # more GPT-like than ReLU
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Decoder Layer / Block
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, rope_base: float, max_seq_len: int) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttentionRoPE(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)

        self.ff = FeedForward(embed_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.self_attn(x)))
        x = self.norm2(x + self.drop2(self.ff(x)))
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
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, dropout, rope_base=rope_base, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# -----------------------------
# SoloGPT_v2 (RoPE version)
# -----------------------------
class SoloGPT_v2(nn.Module):
    """
    Config expected keys (same as your original):
      - n_embd
      - n_head
      - n_layer
      - vocab_size
    Optional:
      - dropout (default 0.1)
      - max_seq_len (default 8192)
      - rope_base (default 10000.0)
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model_type = "Transformer"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.embed_dim = int(config["n_embd"])
        self.n_heads = int(config["n_head"])
        self.n_layers = int(config["n_layer"])
        self.dropout = float(config.get("dropout", 0.1))
        self.vocab_size = int(config["vocab_size"])

        # new (optional) config
        self.max_seq_len = int(config.get("max_seq_len", 8192))
        self.rope_base = float(config.get("rope_base", 10000.0))

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # NOTE: Removed absolute PositionalEncoding; RoPE is applied in attention
        self.decoder = DecoderBlock(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            rope_base=self.rope_base,
            max_seq_len=self.max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) token ids
        returns logits: (B, T, vocab_size)
        """
        x = self.input_embed(x)     # (B, T, C)
        logits = self.decoder(x)    # (B, T, vocab)
        return logits
