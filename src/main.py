"""
main.py
=======
Train model on word list and generate lots of examples.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset
import model
import train

## ------------------------------
## Change for other applications!
DATA_FILE = "../data/IPN_GB_2023.csv"
## ------------------------------
MODEL_SAVE_FILE = "../results/trained_model.pt"
LOSS_CURVE_SAVE_FILE = "../results/loss_curve.png"
MODEL_WIDTH = 256
HIDDEN_SIZE = 64
N_BATCHES = 2000
BATCH_SIZE = 256

N_WORDS_TO_GENERATE = 100
GENERATED_WORDS_SAVE_FILE = "../results/generated_words.txt"


def main():
    print("Assembling training set...")
    word_dataset = dataset.WordDataset(DATA_FILE)

    print("Constructing model...")
    word_generator = model.WordGenerator(
        alphabet_size=len(word_dataset.alphabet),
        model_width=MODEL_WIDTH,
        hidden_size=HIDDEN_SIZE,
    )
    optimizer = torch.optim.AdamW(params=word_generator.parameters(), lr=3e-4)

    print("Training model...")
    trained_model, batch_losses = train.train_model(
        word_generator,
        word_dataset,
        optimizer,
        n_batches=N_BATCHES,
        batch_size=BATCH_SIZE,
    )

    print("Saving model...")
    torch.save(trained_model, MODEL_SAVE_FILE)

    print("Plotting loss curve...")
    fg, ax = plt.subplots(figsize=(7, 4))
    ax.plot(batch_losses.detach(), c="k", label="Training loss", lw=1)
    ax.axhline(
        np.log(len(word_dataset.alphabet)),
        c="k",
        ls=":",
        label="Baseline",
    )
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    fg.savefig(LOSS_CURVE_SAVE_FILE, bbox_inches="tight")

    print("Generating words...")
    first_letters = [word[0].lower() for word in word_dataset.word_list]
    letters, abundances = np.unique(first_letters, return_counts=True)
    first_letter_distribution = np.array(abundances) / sum(abundances)

    generated_words = []
    for _ in range(N_WORDS_TO_GENERATE):
        first_letter = np.random.choice(letters, p=first_letter_distribution)
        generated_words.append(
            word_generator.generate(word_dataset.alphabet, first_letter)
        )
    with open(GENERATED_WORDS_SAVE_FILE, "w+") as f:
        for word in generated_words:
            f.write(word)
            f.write("\n")


if __name__ == "__main__":
    main()
