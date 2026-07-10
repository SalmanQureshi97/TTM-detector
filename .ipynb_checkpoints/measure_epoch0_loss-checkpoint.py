"""Measure epoch-0 loss (untrained model) on train + val for a config.

Instantiates the model from the given configs with *fresh random weights*
(and, for SpecTTTra, whatever pretrained backbone the config points at --
that's the correct 'epoch 0' since fine-tuning starts from that state).
No checkpoint is loaded.

Runs a bounded number of batches over the same train and val loaders the
training pipeline uses (same task/experiment/runtime configs), and
reports:

  * empirical train / val cross-entropy at epoch 0
  * closed-form chance value log(num_classes) for comparison
  * an ok/warn tag if empirical deviates from chance by more than 0.05

Use it to answer 'does epoch-0 loss match chance performance?' for
already-trained SpecCNN and SpecTTTra runs, without needing to re-train.

Usage:
    python scripts/measure_epoch0_loss.py \\
        --model configs/models/deezer_speccnn_amplitude_robust.yaml \\
        --task configs/tasks/four_class_flat_smooth.yaml \\
        --experiment configs/experiments/c1_sonics_only.yaml \\
        --runtime configs/runtime/train_robust_speccnn.yaml \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_with_splits.csv \\
        --num-batches 100
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_experiment import (  # noqa: E402
    compute_loss,
    make_dataloader,
    move_target_to_device,
)
from src.models.model_factory import UnifiedAudioModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--num-batches", type=int, default=100,
                   help="Cap on batches per split. Chance value converges "
                        "fast; 50-100 is plenty.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _chance_value(task_cfg):
    """Closed-form expected loss when the model outputs uniform logits."""
    t = task_cfg["type"]
    if t == "binary":
        # BCEWithLogits at logits=0: -[y log 0.5 + (1-y) log 0.5] = log(2).
        return math.log(2.0), "log(2)"
    if t == "multiclass":
        c = int(task_cfg.get("num_classes", 4))
        return math.log(c), f"log({c})"
    if t in {"multitask", "hierarchical"}:
        # Sum of per-head chance losses; user should read the coefficients
        # from task_cfg["loss"] and reason about their sum manually. We
        # print a best-effort estimate assuming binary + 4-class heads,
        # both weighted equally.
        return math.log(2.0) + math.log(4.0), "log(2) + log(4)  [assumed]"
    raise ValueError(t)


@torch.no_grad()
def _epoch0_loss(model, loader, task_cfg, device, num_batches, desc):
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, total=min(num_batches, len(loader)), desc=desc):
        audio = batch["audio"].to(device, non_blocking=True)
        target = move_target_to_device(batch["target"], device)
        outputs = model(audio)
        loss = compute_loss(task_cfg, outputs, target)
        total += float(loss.item())
        n += 1
        if n >= num_batches:
            break
    return total / max(n, 1), n


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("epoch0")
    args = parse_args()

    torch.manual_seed(args.seed)

    model_cfg = _load(args.model)
    task_cfg = _load(args.task)
    experiment_cfg = _load(args.experiment)
    runtime_cfg = _load(args.runtime)

    # Same defaults run_experiment.py sets.
    runtime_cfg.setdefault("sample_rate",
                           model_cfg.get("frontend", {}).get("sample_rate", 44100))

    device = torch.device(args.device)
    log.info("Building model (RANDOM INIT -- no checkpoint loaded).")
    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)

    train_loader = make_dataloader(
        manifest=args.manifest, split="train", task_cfg=task_cfg,
        datasets=experiment_cfg["train_datasets"], runtime_cfg=runtime_cfg,
        shuffle=True,
        balanced=runtime_cfg.get("balanced_sampling", True),
    )
    val_loader = make_dataloader(
        manifest=args.manifest, split="val", task_cfg=task_cfg,
        datasets=experiment_cfg["val_datasets"], runtime_cfg=runtime_cfg,
        shuffle=False,
        balance_subset=runtime_cfg.get("balanced_val", False),
    )
    log.info("Train batches available: %d  |  Val batches available: %d",
             len(train_loader), len(val_loader))

    train_l, train_n = _epoch0_loss(
        model, train_loader, task_cfg, device, args.num_batches, "epoch-0 train"
    )
    val_l, val_n = _epoch0_loss(
        model, val_loader, task_cfg, device, args.num_batches, "epoch-0 val"
    )
    chance, chance_expr = _chance_value(task_cfg)

    def _tag(x):
        return "OK" if abs(x - chance) < 0.05 else "WARN (differs from chance by >0.05)"

    print("\n=== Epoch-0 loss (untrained model) ===")
    print(f"  model     : {model_cfg['name']}")
    print(f"  task      : {task_cfg['name']}  (type={task_cfg['type']}, "
          f"label_smoothing={task_cfg.get('label_smoothing', 0.0)})")
    print(f"  device    : {device}")
    print(f"  seed      : {args.seed}")
    print()
    print(f"  train_loss (over {train_n} batches) = {train_l:.4f}   [{_tag(train_l)}]")
    print(f"  val_loss   (over {val_n} batches)   = {val_l:.4f}   [{_tag(val_l)}]")
    print()
    print(f"  chance    : {chance_expr} = {chance:.4f}")
    print("  Arithmetic: model with uniform logits gives log-softmax = -log(C) "
          "for every class,")
    print("              so CE = -sum_i q_i * log p_i = log(C) regardless of "
          "label smoothing.")


if __name__ == "__main__":
    main()
