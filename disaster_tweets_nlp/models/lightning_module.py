from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from disaster_tweets_nlp.models.text_classifier import BiLSTMClassifier


class DisasterTweetsLitModule(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_id=pad_id,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch.input_ids)
        loss = self.criterion(logits, batch.label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        logits = self(batch.input_ids)
        loss = self.criterion(logits, batch.label)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= 0.5).long()

        self.val_acc.update(preds, batch.label)
        self.val_f1.update(preds, batch.label)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        return optimizer
