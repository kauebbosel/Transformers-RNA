import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe: torch.Tensor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        x = x + self.pe[:L, :].unsqueeze(0)
        return self.dropout(x)


class TransformerSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_id: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = input_ids == self.pad_id
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1)
        x = x * valid
        denom = valid.sum(dim=1).clamp_min(1)
        pooled = x.sum(dim=1) / denom

        logits = self.classifier(pooled)
        return logits

