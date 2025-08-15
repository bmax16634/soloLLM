import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset, Subset


class SoloGPT_v1(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model_type = 'Transformer'

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            

        #self.batch_size = config['training']['batch_size']
        #self.seq_length = config['training']['seq_length']
        self.embed_dim = config['n_embd']
        
        self.n_head = config['n_head']
        self.n_layers = config['n_layer']
        self.dropout = config["dropout"]
        self.vocab_size = config['vocab_size']

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.decoder = Custom_Decoder(self.embed_dim, self.n_head, self.n_layers, self.vocab_size, self.dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X = self.input_embed(X)
        X = self.pos_encoder(X)
        X = self.decoder(X)
        return X


class Custom_Decoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, vocab_size, dropout=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Custom_DecoderLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X)
        X = self.norm(X)
        X = self.output_layer(X)
        return X


class Custom_DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        seq_len = X.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()

        residual = X
        X, _ = self.mha(X, X, X, attn_mask=attn_mask)
        X = self.dropout1(X)
        X = self.norm1(residual + X)

        residual = X
        X = self.ff(X)
        X = self.dropout2(X)
        X = self.norm2(residual + X)

        return X


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout=0.1) -> None:
        super().__init__()
        self.linear_layer1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_layer2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.linear_layer1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.linear_layer2(X)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
