import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoloGPT_v2(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model_type = "Transformer"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.embed_dim = config["n_embd"]
        self.n_heads = config["n_head"]
        self.n_layers = config["n_layer"]
        self.dropout = config.get("dropout", 0.1)
        self.vocab_size = config["vocab_size"]

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.decoder = DecoderBlock(self.embed_dim, self.n_heads, self.n_layers, self.vocab_size, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.input_embed(x)
        x = self.pos_encoder(x)
        x = self.decoder(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, vocab_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = FeedForward(embed_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        attn_output, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])