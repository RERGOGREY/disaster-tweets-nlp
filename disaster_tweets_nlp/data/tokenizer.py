from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_PAD = "<pad>"
_UNK = "<unk>"


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_id: int
    unk_id: int


class SimpleTokenizer:
    """
    Whitespace tokenizer + vocabulary built from training texts.
    Saved/loaded via torch.save/torch.load.
    """

    def __init__(self, vocab: Vocab, max_len: int) -> None:
        self.vocab = vocab
        self.max_len = max_len

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    @classmethod
    def build(cls, texts: list[str], min_freq: int, max_len: int) -> SimpleTokenizer:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls._tokenize(text))

        itos = [_PAD, _UNK]
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in (_PAD, _UNK):
                itos.append(token)

        stoi = {token: idx for idx, token in enumerate(itos)}
        vocab = Vocab(stoi=stoi, itos=itos, pad_id=stoi[_PAD], unk_id=stoi[_UNK])
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._tokenize(text)
        token_ids = [self.vocab.stoi.get(token, self.vocab.unk_id) for token in tokens]
        token_ids = token_ids[: self.max_len]
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [self.vocab.pad_id] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stoi": self.vocab.stoi,
                "itos": self.vocab.itos,
                "pad_id": self.vocab.pad_id,
                "unk_id": self.vocab.unk_id,
                "max_len": self.max_len,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SimpleTokenizer:
        obj: dict[str, Any] = torch.load(Path(path), map_location="cpu")
        vocab = Vocab(
            stoi=obj["stoi"],
            itos=obj["itos"],
            pad_id=int(obj["pad_id"]),
            unk_id=int(obj["unk_id"]),
        )
        return cls(vocab=vocab, max_len=int(obj["max_len"]))
