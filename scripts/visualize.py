"""Entry point for model interpretability and visualization.

Usage::

    python scripts/visualize.py \\
        --model   configs/models/sonics_vit.yaml \\
        --task    configs/tasks/authenticity_binary.yaml \\
        --checkpoint outputs/vit_binary/best.pt \\
        --manifest   data/master_manifest.csv \\
        --output-dir outputs/viz/vit_binary/ \\
        --modes gradcam attention_rollout embeddings frequency \\
        --num-samples 8 \\
        --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset
from src.models.model_factory import UnifiedAudioModel
from src.utils.config import load_yaml
from src.visualization.attention_rollout import AttentionRollout
from src.visualization.embeddings import extract_embeddings, plot_embedding_grid
from src.visualization.frequency import FrequencySensitivity
from src.visualization.gradcam import GradCAM
from src.visualization.overlay import (
    get_spectrogram_from_model,
    overlay_heatmap_on_spectrogram,
    plot_side_by_side,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TTM-detector visualization")
    p.add_argument("--model", required=True, help="Model config YAML")
    p.add_argument("--task", required=True, help="Task config YAML")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--manifest", required=True, help="Master manifest CSV")
    p.add_argument("--output-dir", required=True, help="Directory for output figures")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["gradcam", "attention_rollout", "embeddings", "frequency", "all"],
        default=["all"],
        help="Which visualizations to produce",
    )
    p.add_argument("--num-samples", type=int, default=8, help="Samples for per-sample analyses")
    p.add_argument("--split", default="test")
    p.add_argument("--dataset-filter", nargs="*", default=None)
    p.add_argument(
        "--embedding-color-by",
        nargs="+",
        default=["auth_label", "source_dataset"],
        help="Metadata fields for embedding coloring",
    )
    p.add_argument("--embedding-method", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--max-embedding-samples", type=int, default=2000)
    p.add_argument("--freq-bands", type=int, default=16)
    p.add_argument("--head-fusion", choices=["mean", "max", "min"], default="mean")
    p.add_argument("--task-output-key", default=None, help="For multitask: 'auth' or 'enc'")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--segment-seconds", type=int, default=30)
    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()

    # ----- Load configs & model -----
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    backbone_type = model_cfg["backbone"]["type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model on %s", device)

    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    log.info("Loaded checkpoint: %s", args.checkpoint)

    # ----- Dataset -----
    ds = AudioManifestDataset(
        manifest_path=args.manifest,
        split=args.split,
        task_cfg=task_cfg,
        dataset_filter=args.dataset_filter,
        sample_rate=44100,
        max_seconds=args.segment_seconds,
    )
    log.info("Dataset: %d samples (split=%s)", len(ds), args.split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = set(args.modes)
    if "all" in modes:
        modes = {"gradcam", "attention_rollout", "embeddings", "frequency"}

    # ===================================================================
    # Per-sample analyses: Grad-CAM and Attention Rollout
    # ===================================================================
    if "gradcam" in modes or "attention_rollout" in modes:
        cam = GradCAM(model, model_cfg) if "gradcam" in modes else None
        rollout = None
        if "attention_rollout" in modes and backbone_type != "speccnn":
            rollout = AttentionRollout(model, model_cfg, head_fusion=args.head_fusion)
        elif "attention_rollout" in modes:
            log.warning("Attention rollout skipped: not supported for SpecCNN backbone.")

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break

            audio = batch["audio"].to(device)
            meta = batch["meta"]
            sample_id = meta.get("track_id", [str(i)])[0] if isinstance(meta.get("track_id"), list) else str(i)
            spec = get_spectrogram_from_model(model, audio)

            heatmaps: dict[str, any] = {}

            if cam is not None:
                log.info("Grad-CAM: sample %d / %d", i + 1, args.num_samples)
                hmap = cam.compute(audio, task_output_key=args.task_output_key)
                heatmaps["Grad-CAM"] = hmap
                overlay_heatmap_on_spectrogram(
                    spec, hmap,
                    title=f"Grad-CAM — sample {sample_id}",
                    output_path=output_dir / f"gradcam_{i:03d}.png",
                )

            if rollout is not None:
                log.info("Attention rollout: sample %d / %d", i + 1, args.num_samples)
                hmap = rollout.compute(audio)
                heatmaps["Attn Rollout"] = hmap
                overlay_heatmap_on_spectrogram(
                    spec, hmap,
                    title=f"Attention Rollout — sample {sample_id}",
                    output_path=output_dir / f"rollout_{i:03d}.png",
                )

            # Side-by-side comparison if both were computed
            if len(heatmaps) > 1:
                plot_side_by_side(
                    spec, heatmaps,
                    title=f"Comparison — sample {sample_id}",
                    output_path=output_dir / f"comparison_{i:03d}.png",
                )

    # ===================================================================
    # Embedding visualization
    # ===================================================================
    if "embeddings" in modes:
        log.info("Extracting embeddings (max %d samples)...", args.max_embedding_samples)
        emb_loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        embeddings, metadata = extract_embeddings(
            model, emb_loader, device, max_samples=args.max_embedding_samples
        )
        log.info("Extracted %d embeddings of dim %d", *embeddings.shape)

        plot_embedding_grid(
            embeddings, metadata,
            color_fields=args.embedding_color_by,
            method=args.embedding_method,
            output_path=output_dir / "embeddings.png",
        )
        log.info("Saved embedding plot → %s", output_dir / "embeddings.png")

    # ===================================================================
    # Frequency sensitivity
    # ===================================================================
    if "frequency" in modes:
        log.info("Running frequency sensitivity analysis...")
        sample = ds[0]
        audio = sample["audio"].unsqueeze(0).to(device)

        freq_analyzer = FrequencySensitivity(model, device)

        # Band occlusion
        occ_result = freq_analyzer.band_occlusion(
            audio, n_bands=args.freq_bands, task_output_key=args.task_output_key
        )
        freq_analyzer.plot_sensitivity(
            occ_result, method="occlusion",
            output_path=output_dir / "freq_occlusion.png",
        )
        log.info("Saved → %s", output_dir / "freq_occlusion.png")

        # Gradient frequency profile
        grad_result = freq_analyzer.gradient_frequency_profile(
            audio, task_output_key=args.task_output_key
        )
        freq_analyzer.plot_sensitivity(
            grad_result, method="gradient",
            output_path=output_dir / "freq_gradient.png",
        )
        log.info("Saved → %s", output_dir / "freq_gradient.png")

    log.info("Done. All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
