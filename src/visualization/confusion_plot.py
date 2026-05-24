from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion(cm, class_names, out_path, title="Confusion Matrix"):
    """Save an annotated confusion-matrix heatmap (raw counts).

    Args:
        cm: 2D array of counts (rows = true, cols = predicted).
        class_names: list of class names for both axes.
        out_path: destination PNG path.
        title: figure title.
    """
    cm = np.asarray(cm)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.4 * n + 2, 1.4 * n + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
