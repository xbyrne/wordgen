"""
preprocessing.py
================
Functions for preprocessing the list of words.
"""

import pandas as pd
import torch
from torch.nn import functional as F


## ------------------------------
## Change for other applications!
def extract_raw_words(data_file: str = "../data/IPN_GB_2023.csv") -> list[str]:
    """
    Extracts list of place names from the data file.

    Arguments:
        data_file (str): path to the data file containing the place names.

    Returns:
        list[str]: List of place names.

    """
    df = pd.read_csv(data_file, encoding="latin-1")
    country_field = "ctry22nm"
    desired_country = "England"
    name_field = "place22nm"
    return df[df[country_field] == desired_country][name_field].tolist()


## ------------------------------


def clean_list(word_list: list[str]) -> list[str]:
    """
    Cleans up a list of place names, removing names that contain extra-special
        characters (looking at you, Westward Ho!).

    Arguments:
        word_list (list[str]): List of strings, some of which may contain special
            characters ("&", ",", "(", ")", "!", ":", "/")

    Returns:
        list[str]: List of words in `word_list` that do not contain any of the extra
            special characters.

    """
    disqualifying_chars = ["&", ",", "(", ")", "!", ":", "/"]
    cleaned_word_list = [
        word
        for word in word_list
        if not any(
            disqualifying_char in word for disqualifying_char in disqualifying_chars
        )
    ]
    return cleaned_word_list


def alphabet_from_list(word_list: list[str]) -> str:
    """
    Collates a list of unique (lower-case) letters contained in `word_list`

    Arguments:
        word_list (list[str]): List of words.

    Returns:
        String of unique characters contained in `word_list`

    """
    alphabet = []
    for word in word_list:
        for letter in word:
            if letter.lower() not in alphabet:
                alphabet.append(letter.lower())
    return sorted(alphabet) + ["<EOW>"]


def tensor_from_word(word: str, alphabet: str) -> torch.tensor:
    """
    Converts a word into a one-hot tensor representation. Also

    Arguments:
        word (str): ...a word.
        alphabet (str): String of characters to create the tensor representation from.
            Should include '<EOW>' as its final character.

    Returns:
        torch.tensor: Tensor representation of the word.

    """
    word = word.lower()
    word_tensor = torch.zeros((len(word) + 1, len(alphabet)))

    for i, letter in enumerate(word):
        word_tensor[i] = F.one_hot(torch.tensor(alphabet.index(letter)), len(alphabet))

    # <EOW>
    word_tensor[-1, -1] = 1.0

    return word_tensor


def word_from_tensor(tensor: torch.tensor, alphabet: str) -> str:
    """
    Converts tensor representation of a word into a string.

    Arguments:
        tensor (torch.tensor): Tensor representation of the word. Final row should be
            the <EOW>.
        alphabet (str): Alphabet to which the characters belong.

    Returns:
        str: Word represented by the input tensor.

    """
    word = ""
    for row in tensor[0, :-1]:  # Ignore EOW
        word += alphabet[torch.argmax(row)]
    return word
