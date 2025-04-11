"""
dataset.py
==========
Dataset object for storing word list.
"""

import random
from typing import Union

import torch
from tqdm import tqdm

import preprocessing as pp


class WordDataset(torch.utils.data.Dataset):
    """Dataset object containing word list."""

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

        raw_words = pp.extract_raw_words(data_path)
        self.word_list = pp.clean_list(raw_words)

        self.alphabet = pp.alphabet_from_list(self.word_list)
        self.data = [
            pp.tensor_from_word(word, self.alphabet) for word in tqdm(self.word_list)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self, n: int = 1) -> Union[str, list[str]]:
        """Sample one or more words from the dataset."""
        sample_ = random.sample(self.data, n)
        if n == 1:
            return sample_[0]
        return sample_
