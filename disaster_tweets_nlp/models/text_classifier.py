from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)  # [B, T, D]
        _, (h_n, _) = self.lstm(embeddings)

        if self.lstm.bidirectional:
            last_fw = h_n[-2]
            last_bw = h_n[-1]
            features = torch.cat([last_fw, last_bw], dim=-1)
        else:
            features = h_n[-1]

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits
