from __future__ import annotations

# from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from disaster_tweets_nlp.data.tokenizer import SimpleTokenizer

# @dataclass(frozen=True)
# class TweetBatch:
#     input_ids: torch.Tensor
#     label: torch.Tensor


class TweetDataset(Dataset):
    def __init__(
        self, texts: list[str], labels: list[int] | None, tokenizer: SimpleTokenizer
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.tokenizer.encode(self.texts[index])

        if self.labels is None:
            label = torch.tensor(-1, dtype=torch.long)
        else:
            label = torch.tensor(int(self.labels[index]), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": label,
        }
