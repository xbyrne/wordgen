"""
model.py
========
Word generation model.
"""

import torch
from torch import nn

import preprocessing as pp


class WordGenerator(nn.Module):
    """Module to generate words based on training data."""

    def __init__(self, alphabet_size: int, model_width: int, hidden_size: int):
        super().__init__()

        self.input_size = alphabet_size
        self.hidden_size = hidden_size

        self.input_to_intermediate = nn.Linear(alphabet_size + hidden_size, model_width)
        self.input_to_hidden = nn.Linear(alphabet_size + hidden_size, hidden_size)
        self.intermediate_to_output = nn.Linear(
            model_width + hidden_size, alphabet_size
        )

    def forward(self, word_tensor: torch.tensor):
        """
        Recurrently processes an input tensor.

        Arguments:
            word_tensor (torch.tensor): Tensor representing a word, including <EOW>
                character. Shape [1, len(word), len(alphabet)]

        Returns:
            torch.tensor: Shape [1, len(word)-1, len(alphabet)]

        """
        device = next(self.parameters()).device
        hidden = self.init_hidden().to(device)  # Hidden state of module
        output_tensor = torch.zeros_like(word_tensor, device=device)[:, :-1]

        for i in range(word_tensor.shape[1] - 1):
            combined_input = torch.cat([word_tensor[:, i], hidden], dim=-1)

            intermediate = self.input_to_intermediate(combined_input)
            hidden = self.input_to_hidden(combined_input)

            x = torch.cat([intermediate, hidden], dim=-1)
            x = self.intermediate_to_output(x)

            output_tensor[:, i] = x

        return output_tensor

    def init_hidden(self):
        """Generates an initial value of the hidden state (zero vector)."""
        return torch.zeros(1, self.hidden_size)

    def generate(self, alphabet: str, initial_letter: str) -> str:
        """
        Stochastically generates a word, from an alphabet and an initial letter.
        Recurrently generates a probability distribution of the next letter, and keeps
        adding on letters until <EOW> is generated.

        Arguments:
            alphabet (str): Alphabet from which to draw letters.
            initial_letter (str): First letter of word to generate.

        Returns:
            str: Generated word.

        """
        word = initial_letter
        next_letter = ""

        while next_letter != "<EOW>":
            word_tensor = pp.tensor_from_word(word, alphabet).unsqueeze(0)
            next_letter_distr = torch.distributions.categorical.Categorical(
                logits=self.forward(word_tensor)[0][-1]
            )
            next_letter = alphabet[next_letter_distr.sample()]
            word += next_letter

        return pp.word_from_tensor(word_tensor, alphabet).title()
