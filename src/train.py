"""
train.py
========
Training loop
"""

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def train_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    n_batches: int,
    batch_size: int = 64,
) -> tuple[nn.Module, torch.tensor]:
    """
    Trains a model on the words in a dataset.

    Arguments:
        model (nn.Module): Model to train.
        dataset (torch.utils.data.Dataset): Source of word data.
        optimizer (torch.optim.Optimizer): Optimizer with which to train the model.
        n_batches (int): Number of batches on which to train the model.
        batch_size (int): Size of training batches.

    Returns:
        nn.Module: Trained model.
        torch.tensor: Mean batch losses.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model = model.to(device)

    batch_losses = torch.zeros(n_batches)
    pbar = tqdm(range(n_batches), total=n_batches)

    for i in pbar:
        optimizer.zero_grad()
        mean_batch_loss = 0.0
        for word_tensor in dataset.sample(batch_size):
            word_tensor = word_tensor.to(device)
            word_guess_logits = model(word_tensor.unsqueeze(0))
            mean_batch_loss += (
                calculate_word_loss(word_guess_logits, word_tensor) / batch_size
            )

        batch_losses[i] = mean_batch_loss
        pbar.set_postfix({"BL": f"{mean_batch_loss:.3f}"})

        mean_batch_loss.backward()
        optimizer.step()

    return model, batch_losses


def calculate_word_loss(word_guess_logits, word_tensor):
    """
    Calculates the loss for a next-letter prediction.

    Arguments:
        word_guess_logits (torch.tensor): Model output. Shape:
            [1, len(word)-1, len(alphabet)] (excludes EOW).
        word_tensor (torch.tensor): Ground truth word; we'll trim off the first letter
            as we're not 'predicting' that. Shape: [len(word), len(alphabet)].

    Returns:
        torch.tensor: Scalar loss.

    """
    word_guess_logits_reshaped = torch.swapaxes(word_guess_logits, 1, 2)
    #       ^^^---- [1, len(alphabet), len(word) - 1]
    word_tensor_reshaped = torch.swapaxes(word_tensor[1:], 0, 1).unsqueeze(0)
    #       ^^^---- [1, len(alphabet), len(word) - 1]
    return F.cross_entropy(word_guess_logits_reshaped, word_tensor_reshaped)
