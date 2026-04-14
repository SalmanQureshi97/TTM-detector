"""Embedding space visualization via t-SNE / UMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, dict[str, list]]:
    """Extract backbone embeddings (before the head) for a dataset.

    Registers a forward hook on ``model.head`` to capture its *input*,
    which is the backbone embedding.

    Parameters
    ----------
    model : UnifiedAudioModel
    dataloader : DataLoader
        Should yield dicts with ``"audio"`` and ``"meta"`` keys.
    device : torch.device
    max_samples : int
        Stop after collecting this many samples.

    Returns
    -------
    embeddings : np.ndarray [N, embed_dim]
    metadata : dict[str, list]
        Keys mirror the manifest columns (e.g. ``"auth_label"``,
        ``"source_dataset"``, ``"generator"``, ``"encoder"``,
        ``"class4_label"``).
    """
    embeddings_list: list[np.ndarray] = []
    meta_dict: dict[str, list] = {}
    collected = 0

    # Hook to capture head input
    captured: list[torch.Tensor] = []

    def _hook(_module, inp, _out):
        # inp is a tuple; first element is the embedding tensor
        captured.append(inp[0].detach().cpu())

    handle = model.head.register_forward_hook(_hook)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if collected >= max_samples:
                break
            audio = batch["audio"].to(device)
            captured.clear()
            model(audio)

            if captured:
                emb = captured[0].numpy()  # [B, D]
                embeddings_list.append(emb)

                # Collect metadata
                meta = batch["meta"]
                for key in meta:
                    if key not in meta_dict:
                        meta_dict[key] = []
                    vals = meta[key]
                    if isinstance(vals, torch.Tensor):
                        vals = vals.cpu().numpy().tolist()
                    elif isinstance(vals, (list, tuple)):
                        vals = list(vals)
                    else:
                        vals = [vals]
                    meta_dict[key].extend(vals)

                collected += emb.shape[0]

    handle.remove()

    embeddings = np.concatenate(embeddings_list, axis=0)[:max_samples]
    for key in meta_dict:
        meta_dict[key] = meta_dict[key][:max_samples]

    return embeddings, meta_dict


# ---------------------------------------------------------------------------
# 2-D projection
# ---------------------------------------------------------------------------

def _project_2d(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2-D using t-SNE or UMAP.

    Returns
    -------
    np.ndarray [N, 2]
    """
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, embeddings.shape[0] - 1),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP projection.  "
                "Install with:  pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Unknown projection method: {method}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embedding_space(
    embeddings: np.ndarray,
    metadata: dict[str, list],
    method: str = "tsne",
    color_by: str = "auth_label",
    perplexity: int = 30,
    n_neighbors: int = 15,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    point_size: float = 8,
    point_alpha: float = 0.6,
) -> Figure:
    """Project embeddings to 2-D and produce a colored scatter plot.

    Parameters
    ----------
    embeddings : np.ndarray [N, D]
    metadata : dict[str, list]
    method : str  ``"tsne"`` or ``"umap"``
    color_by : str  key in *metadata* to use for coloring
    output_path : Path, optional  save figure here
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    proj = _project_2d(embeddings, method=method, perplexity=perplexity, n_neighbors=n_neighbors)
    labels = metadata.get(color_by, ["unknown"] * embeddings.shape[0])
    unique_labels = sorted(set(str(l) for l in labels))
    color_map = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = [str(l) == lbl for l in labels]
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=point_size,
            alpha=point_alpha,
            label=lbl,
            color=color_map(i),
        )

    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"Embedding space ({method.upper()}, colored by {color_by})")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_embedding_grid(
    embeddings: np.ndarray,
    metadata: dict[str, list],
    color_fields: list[str],
    method: str = "tsne",
    perplexity: int = 30,
    n_neighbors: int = 15,
    output_path: Optional[Path] = None,
    figsize_per_panel: tuple[float, float] = (6, 5),
    point_size: float = 6,
    point_alpha: float = 0.5,
) -> Figure:
    """Plot multiple embedding scatter subplots sharing the same 2-D projection.

    Each subplot is colored by a different metadata field (e.g. label,
    source_dataset, generator).
    """
    proj = _project_2d(embeddings, method=method, perplexity=perplexity, n_neighbors=n_neighbors)

    n = len(color_fields)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1])
    )
    if n == 1:
        axes = [axes]

    for ax, field in zip(axes, color_fields):
        labels = metadata.get(field, ["unknown"] * embeddings.shape[0])
        unique_labels = sorted(set(str(l) for l in labels))
        cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

        for i, lbl in enumerate(unique_labels):
            mask = np.array([str(l) == lbl for l in labels])
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=point_size,
                alpha=point_alpha,
                label=lbl,
                color=cmap(i),
            )
        ax.legend(markerscale=2, fontsize=7, loc="best")
        ax.set_title(f"Color: {field}")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    fig.suptitle(f"Embedding space ({method.upper()})", fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
