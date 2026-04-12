"""Orchestrate generator-holdout (S5) or encoder-holdout (S6) experiments.

For each unique value in the holdout field (e.g., each generator or encoder),
this script:
  1. Builds a filtered manifest that excludes rows matching the held-out value.
  2. Trains a model on the filtered data.
  3. Evaluates the trained model on test-split rows matching the held-out value.

Usage:
    # Generator holdout (S5) — hold out each generator, one at a time
    python scripts/run_holdout.py \
        --holdout-config configs/experiments/generator_holdout.yaml \
        --base-experiment configs/experiments/c5_all.yaml \
        --model configs/models/sonics_spectttra.yaml \
        --task configs/tasks/multitask_two_head.yaml \
        --runtime configs/runtime/train_default.yaml \
        --eval-runtime configs/runtime/eval_default.yaml \
        --manifest data/manifests/master_manifest_with_splits.csv

    # Encoder holdout (S6) — same idea
    python scripts/run_holdout.py \
        --holdout-config configs/experiments/encoder_holdout.yaml \
        --base-experiment configs/experiments/c5_all.yaml \
        --model configs/models/sonics_spectttra.yaml \
        --task configs/tasks/multitask_two_head.yaml \
        --runtime configs/runtime/train_default.yaml \
        --eval-runtime configs/runtime/eval_default.yaml \
        --manifest data/manifests/master_manifest_with_splits.csv

    # Only run a subset of holdout values
    python scripts/run_holdout.py ... --only suno,udio
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset
from src.evaluation.metrics import (
    binary_metrics,
    four_class_projected_metrics,
    multitask_metrics,
)
from src.evaluation.robustness import apply_bandpass
from src.experiments.run_experiment import run_training
from src.models.model_factory import UnifiedAudioModel
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.seed import set_seed

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run holdout experiments (S5 / S6).")
    p.add_argument("--holdout-config", required=True, help="Path to generator_holdout.yaml or encoder_holdout.yaml")
    p.add_argument("--base-experiment", required=True, help="Base experiment config (e.g., c5_all.yaml)")
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--runtime", required=True, help="Training runtime config")
    p.add_argument("--eval-runtime", required=True, help="Evaluation runtime config")
    p.add_argument("--manifest", required=True)
    p.add_argument("--only", default=None, help="Comma-separated subset of holdout values to run")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=None, help="Override output directory")
    return p.parse_args()


def evaluate_checkpoint(model_cfg, task_cfg, eval_runtime, checkpoint, manifest_path,
                        dataset_filter, split, device, bandpass_low=None, bandpass_high=None):
    """Evaluate a checkpoint on filtered data and return metrics dict."""
    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = AudioManifestDataset(
        manifest_path=manifest_path,
        split=split,
        task_cfg=task_cfg,
        dataset_filter=dataset_filter,
        sample_rate=44100,
        max_seconds=eval_runtime["segment_seconds"],
    )
    loader = DataLoader(
        ds,
        batch_size=eval_runtime["batch_size"],
        shuffle=False,
        num_workers=eval_runtime["num_workers"],
    )
    threshold = eval_runtime["threshold"]

    with torch.no_grad():
        if task_cfg["type"] == "binary":
            y_true, y_prob = [], []
            for batch in loader:
                audio = batch["audio"].to(device)
                if bandpass_low or bandpass_high:
                    audio = apply_bandpass(audio, 44100, bandpass_low, bandpass_high)
                output = model(audio)
                y_true.extend(batch["target"].cpu().numpy().astype(int).tolist())
                y_prob.extend(torch.sigmoid(output).cpu().numpy().tolist())
            return binary_metrics(np.array(y_true), np.array(y_prob), threshold)

        elif task_cfg["type"] in {"multitask", "hierarchical"}:
            auth_true, auth_prob, enc_true, enc_prob = [], [], [], []
            for batch in loader:
                audio = batch["audio"].to(device)
                if bandpass_low or bandpass_high:
                    audio = apply_bandpass(audio, 44100, bandpass_low, bandpass_high)
                output = model(audio)
                auth_true.extend(batch["target"]["auth"].cpu().numpy().astype(int).tolist())
                enc_true.extend(batch["target"]["enc"].cpu().numpy().astype(int).tolist())
                auth_prob.extend(torch.sigmoid(output["auth"]).cpu().numpy().tolist())
                enc_prob.extend(torch.sigmoid(output["enc"]).cpu().numpy().tolist())
            return multitask_metrics(
                np.array(auth_true), np.array(auth_prob),
                np.array(enc_true), np.array(enc_prob), threshold,
            )

        elif task_cfg["type"] == "multiclass":
            y_true, y_prob = [], []
            for batch in loader:
                audio = batch["audio"].to(device)
                if bandpass_low or bandpass_high:
                    audio = apply_bandpass(audio, 44100, bandpass_low, bandpass_high)
                output = model(audio)
                y_true.extend(batch["target"].cpu().numpy().astype(int).tolist())
                y_prob.extend(torch.softmax(output, dim=1).cpu().numpy().tolist())
            return four_class_projected_metrics(np.array(y_true), np.array(y_prob), threshold)

        else:
            raise ValueError(f"Unsupported task type: {task_cfg['type']}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    holdout_cfg = load_yaml(args.holdout_config)
    base_exp = load_yaml(args.base_experiment)
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    runtime_cfg = load_yaml(args.runtime)
    eval_runtime = load_yaml(args.eval_runtime)
    if args.num_workers is not None:
        runtime_cfg["num_workers"] = args.num_workers
        eval_runtime["num_workers"] = args.num_workers

    set_seed(runtime_cfg.get("seed", 42))
    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")

    holdout_field = holdout_cfg["holdout_field"]
    notation = holdout_cfg["notation"]  # S5 or S6

    manifest = pd.read_csv(args.manifest)
    if holdout_field not in manifest.columns:
        log.error("Holdout field '%s' not found in manifest columns: %s", holdout_field, list(manifest.columns))
        sys.exit(1)

    # Determine which holdout values to iterate over
    all_values = sorted(manifest[holdout_field].dropna().unique().tolist())
    if args.only:
        selected = [v.strip() for v in args.only.split(",")]
        all_values = [v for v in all_values if v in selected]

    log.info("%s holdout (%s) — %d values: %s", notation, holdout_field, len(all_values), all_values)

    out_root = Path(args.output_dir) if args.output_dir else Path(runtime_cfg["save_dir"])
    suite_dir = ensure_dir(out_root / f"{notation.lower()}_{holdout_field}_holdout" / model_cfg["name"] / task_cfg["name"])

    all_results = {}

    for holdout_value in all_values:
        log.info("=" * 60)
        log.info("Holding out %s=%s", holdout_field, holdout_value)
        log.info("=" * 60)

        # Build filtered manifest: exclude held-out value from train/val
        train_val_mask = manifest[holdout_field] != holdout_value
        filtered = manifest[train_val_mask].copy()

        # Also filter by base experiment's train datasets
        filtered = filtered[filtered["source_dataset"].isin(base_exp["train_datasets"])]

        if len(filtered) == 0:
            log.warning("No training data after excluding %s=%s. Skipping.", holdout_field, holdout_value)
            continue

        # Write temporary filtered manifest
        run_dir = ensure_dir(suite_dir / str(holdout_value))
        filtered_manifest_path = run_dir / "filtered_manifest.csv"
        filtered.to_csv(filtered_manifest_path, index=False)

        # Build experiment config for this run
        holdout_exp = {
            "name": f"{notation.lower()}_{holdout_field}_{holdout_value}",
            "train_datasets": base_exp["train_datasets"],
            "val_datasets": base_exp.get("val_datasets", base_exp["train_datasets"]),
            "test_suites": [],
        }

        # Override save_dir to point into the holdout run directory
        holdout_runtime = {**runtime_cfg, "save_dir": str(run_dir)}

        # Train
        log.info("Training with %s=%s excluded (%d rows)", holdout_field, holdout_value, len(filtered))
        ckpt = run_training(model_cfg, task_cfg, holdout_exp, holdout_runtime, str(filtered_manifest_path))
        log.info("Checkpoint saved: %s", ckpt)

        # Evaluate on held-out value only (test split)
        # Build a manifest containing only the held-out value
        held_out = manifest[
            (manifest[holdout_field] == holdout_value)
            & (manifest["split"] == "test")
        ].copy()

        if len(held_out) == 0:
            log.warning("No test data for %s=%s. Skipping evaluation.", holdout_field, holdout_value)
            continue

        held_out_manifest_path = run_dir / "heldout_test_manifest.csv"
        held_out.to_csv(held_out_manifest_path, index=False)

        # Evaluate — pass all datasets since we already filtered to the held-out value
        held_out_datasets = held_out["source_dataset"].unique().tolist()
        metrics = evaluate_checkpoint(
            model_cfg, task_cfg, eval_runtime, ckpt,
            str(held_out_manifest_path), held_out_datasets, "test", device,
        )

        all_results[str(holdout_value)] = {
            "metrics": metrics,
            "train_rows": len(filtered),
            "test_rows": len(held_out),
        }
        log.info("Results for %s=%s: %s", holdout_field, holdout_value, json.dumps(metrics, indent=2))

        # Save per-value results
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save aggregated results
    summary_path = suite_dir / "holdout_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    log.info("Holdout summary saved to %s", summary_path)

    # Print summary table
    log.info("=" * 60)
    log.info("%s HOLDOUT SUMMARY (%s)", notation, holdout_field)
    log.info("=" * 60)
    for val, res in all_results.items():
        log.info("  %s: %s", val, json.dumps(res["metrics"], indent=None))


if __name__ == "__main__":
    main()
