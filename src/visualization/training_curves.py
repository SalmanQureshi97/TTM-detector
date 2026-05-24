from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_curves(history, out_path):
    """Plot training and validation loss vs epoch on a single figure.

    Args:
        history: dict with keys "epoch", "train_loss", "val_loss".
        out_path: destination PNG path.
    """
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax.plot(epochs, history["val_loss"], marker="s", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
