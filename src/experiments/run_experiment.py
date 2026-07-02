from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.manifest_dataset import AudioManifestDataset
from src.datasets.samplers import (
    balanced_subset_indices,
    make_task_balanced_sampler,
    task_balance_labels,
)
from src.evaluation.confusion import compute_confusion
from src.evaluation.predict import collect_predictions
from src.losses.four_class import four_class_loss
from src.losses.hierarchical import hierarchical_loss
from src.losses.multitask import multitask_loss
from src.models.model_factory import UnifiedAudioModel
from src.utils.io import ensure_dir
from src.visualization.confusion_plot import plot_confusion
from src.visualization.training_curves import plot_loss_curves

CLASS4_NAMES = ["real", "real_enc", "fake", "fake_enc"]

log = logging.getLogger(__name__)


def collate_batch(batch):
    """Collate audio + target, but keep ``meta`` as a list of dicts.

    The dataset returns a ``meta`` dict with string fields (filepath, track_id,
    ...). PyTorch's default collate tries to tensor-ise every dict value and
    fails on the strings, so we handle the three keys explicitly here.
    """
    audio = torch.stack([b["audio"] for b in batch])
    targets = [b["target"] for b in batch]
    if isinstance(targets[0], dict):
        target = {k: torch.stack([t[k] for t in targets]) for k in targets[0]}
    else:
        target = torch.stack(targets)
    return {"audio": audio, "target": target, "meta": [b["meta"] for b in batch]}


def make_dataloader(manifest, split, task_cfg, datasets, runtime_cfg, shuffle,
                    balanced=False, balance_subset=False, max_per_class=None):
    ds = AudioManifestDataset(
        manifest_path=manifest,
        split=split,
        task_cfg=task_cfg,
        dataset_filter=datasets,
        sample_rate=runtime_cfg.get("sample_rate", 44100),
        max_seconds=runtime_cfg["segment_seconds"],
    )

    if balance_subset:
        # Downsample to equal per-class counts: a fixed, leakage-neutral
        # balanced subset (used for the balanced validation set and for the
        # train confusion-matrix inference).
        labels = task_balance_labels(ds.df, task_cfg["type"])
        idx = balanced_subset_indices(
            labels, seed=runtime_cfg.get("seed", 42), max_per_class=max_per_class
        )
        ds.df = ds.df.iloc[idx].reset_index(drop=True)
        shuffle = False
        log.info("Balanced subset for split=%s: %d samples", split, len(ds))

    sampler = None
    if balanced:
        sampler = make_task_balanced_sampler(ds)
        shuffle = False  # sampler and shuffle are mutually exclusive
        log.info("Using balanced sampler for split=%s (%d samples)", split, len(ds))

    return DataLoader(
        ds,
        batch_size=runtime_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=runtime_cfg["num_workers"],
        collate_fn=collate_batch,
    )


def _save_confusion(model, loader, task_cfg, device, out_dir, split_name, max_batches=None):
    """Run inference and persist a confusion matrix as PNG + CSV."""
    task_type = task_cfg["type"]
    y_true, y_pred = collect_predictions(model, loader, task_type, device, max_batches=max_batches)

    if task_type == "multiclass":
        labels = list(range(task_cfg.get("num_classes", 4)))
        class_names = CLASS4_NAMES[: len(labels)]
    else:  # binary
        labels = [0, 1]
        class_names = ["neg", "pos"]

    cm = compute_confusion(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / f"confusion_{split_name}.csv"
    )
    plot_confusion(
        cm, class_names, out_dir / f"confusion_{split_name}.png",
        title=f"Confusion Matrix ({split_name})",
    )
    log.info("Saved confusion_%s.png / .csv (%d samples)", split_name, len(y_true))


def compute_loss(task_cfg, outputs, targets):
    eps = float(task_cfg.get("label_smoothing", 0.0))
    if task_cfg["type"] == "binary":
        if eps > 0:
            # Two-class label smoothing on {0, 1}: y' = y*(1-2*eps) + eps.
            targets = targets * (1.0 - 2.0 * eps) + eps
        return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
    if task_cfg["type"] == "multiclass":
        if eps > 0:
            # F.cross_entropy supports label smoothing natively from torch 1.10.
            return torch.nn.functional.cross_entropy(outputs, targets, label_smoothing=eps)
        return four_class_loss(outputs, targets)
    if task_cfg["type"] == "multitask":
        return multitask_loss(outputs, targets, **task_cfg["loss"])
    if task_cfg["type"] == "hierarchical":
        return hierarchical_loss(outputs, targets, **task_cfg["loss"])
    raise ValueError(task_cfg["type"])


def move_target_to_device(target, device):
    if isinstance(target, dict):
        return {k: v.to(device) for k, v in target.items()}
    return target.to(device)


def run_training(model_cfg, task_cfg, experiment_cfg, runtime_cfg, manifest_path,
                 resume=False):
    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Model: %s | Trainable params: %s / %s total",
             model_cfg["name"], f"{trainable:,}", f"{total:,}")

    # Audio loader uses the frontend's sample rate (e.g. 16k for SpecTTTra-120s).
    runtime_cfg.setdefault("sample_rate",
                           model_cfg.get("frontend", {}).get("sample_rate", 44100))

    use_balanced = runtime_cfg.get("balanced_sampling", True)
    train_loader = make_dataloader(
        manifest=manifest_path,
        split="train",
        task_cfg=task_cfg,
        datasets=experiment_cfg["train_datasets"],
        runtime_cfg=runtime_cfg,
        shuffle=True,
        balanced=use_balanced,
    )
    val_loader = make_dataloader(
        manifest=manifest_path,
        split="val",
        task_cfg=task_cfg,
        datasets=experiment_cfg["val_datasets"],
        runtime_cfg=runtime_cfg,
        shuffle=False,
        balance_subset=runtime_cfg.get("balanced_val", False),
    )
    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=runtime_cfg["learning_rate"],
        weight_decay=runtime_cfg["weight_decay"],
    )

    epochs = runtime_cfg["epochs"]
    # LR schedule with optional warmup. If warmup_epochs > 0, ramp the LR
    # linearly from warmup_lr up to learning_rate over that many epochs,
    # then cosine-decay for the remaining epochs. This is what SONICS'
    # upstream training used and prevents the fresh classifier head from
    # destroying the pretrained backbone in the first few optimizer steps.
    warmup_epochs = int(runtime_cfg.get("warmup_epochs", 0) or 0)
    if warmup_epochs > 0:
        target_lr = float(runtime_cfg["learning_rate"])
        warmup_lr = float(runtime_cfg.get("warmup_lr", 1e-6))
        start_factor = max(warmup_lr / target_lr, 1e-8) if target_lr > 0 else 1.0
        warmup_sched = LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_epochs = max(epochs - warmup_epochs, 1)
        main_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched],
            milestones=[warmup_epochs],
        )
        log.info("LR schedule: linear warmup %d epochs from %.2e -> %.2e, "
                 "then cosine decay over %d epochs",
                 warmup_epochs, warmup_lr, target_lr, cosine_epochs)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    grad_clip = runtime_cfg.get("grad_clip", 1.0)
    patience = runtime_cfg.get("early_stopping_patience", 5)

    # Gradient accumulation: run this many forward/backward passes before an
    # optimizer step. Effective batch = batch_size * grad_accum_steps. Useful
    # when the physical batch is memory-limited (SpecTTTra 120s full fine-tune
    # is stuck at batch=2, so grad_accum_steps=64 gets us SONICS' effective
    # batch of 128 without extra memory).
    grad_accum_steps = int(runtime_cfg.get("grad_accum_steps", 1) or 1)
    if grad_accum_steps > 1:
        log.info("Gradient accumulation: %d steps -> effective batch = %d",
                 grad_accum_steps,
                 grad_accum_steps * runtime_cfg["batch_size"])

    # Mixed precision: fp16 forward/backward roughly halves activation memory
    # and speeds up matmul/conv on Ampere. No-op when amp=false or on CPU.
    use_amp = bool(runtime_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        log.info("AMP enabled (fp16 autocast + GradScaler)")

    out_dir = ensure_dir(
        Path(runtime_cfg["save_dir"]) / experiment_cfg["name"] / model_cfg["name"] / task_cfg["name"]
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    limit_batches = runtime_cfg.get("limit_batches")  # cap steps/epoch for smoke tests; None = no cap
    start_epoch = 1

    # --- Resume from last.pt (full training state) if requested ---
    last_ckpt = out_dir / "last.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", best_val)
        epochs_without_improvement = state.get("epochs_without_improvement", 0)
        history = state.get("history", history)
        log.info("Resuming from %s: continuing at epoch %d (best_val=%.4f)",
                 last_ckpt, start_epoch, best_val)
    elif resume:
        log.info("--resume set but no checkpoint at %s; starting fresh.", last_ckpt)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for batch in pbar:
            audio = batch["audio"].to(device, non_blocking=True)
            target = move_target_to_device(batch["target"], device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(audio)
                loss = compute_loss(task_cfg, outputs, target)
            # Divide loss so summed gradients over grad_accum_steps mini-batches
            # equal the gradient of a single larger batch.
            scaler.scale(loss / grad_accum_steps).backward()
            train_loss += loss.item()
            train_steps += 1
            if train_steps % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if limit_batches and train_steps >= limit_batches:
                break

        # Flush any remaining accumulated gradient at the end of the epoch.
        if train_steps % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                audio = batch["audio"].to(device, non_blocking=True)
                target = move_target_to_device(batch["target"], device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(audio)
                    loss = compute_loss(task_cfg, outputs, target)
                val_loss += loss.item()
                val_steps += 1
                if limit_batches and val_steps >= limit_batches:
                    break

        avg_val_loss = val_loss / max(val_steps, 1)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | time=%.1fs",
            epoch, epochs, avg_train_loss, avg_val_loss, lr, elapsed,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(lr)

        scheduler.step()

        # --- Checkpoint ---
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                out_dir / "best.pt",
            )
            log.info("  -> Saved new best checkpoint (val_loss=%.4f)", avg_val_loss)
        else:
            epochs_without_improvement += 1
            log.info(
                "  -> No improvement for %d epoch(s) (patience=%d)",
                epochs_without_improvement, patience,
            )

        # --- Resume checkpoint: full state, every epoch (overwrites) ---
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": avg_val_loss,
                "best_val": best_val,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_ckpt,
        )

        if epochs_without_improvement >= patience:
            log.info("Early stopping triggered after %d epochs.", epoch)
            break

    log.info("Training complete. Best val_loss=%.4f saved to %s", best_val, out_dir / "best.pt")

    # --- Persist loss history (Req 6) ---
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- Post-training reports: loss curves + confusion matrices (Req 5/6) ---
    if runtime_cfg.get("make_reports", True):
        plot_loss_curves(history, out_dir / "loss_curves.png")
        log.info("Saved loss_curves.png")

        task_type = task_cfg["type"]
        ckpt_path = out_dir / "best.pt"
        if task_type in {"binary", "multiclass"} and ckpt_path.exists():
            # Reload best checkpoint so the matrices reflect the saved model.
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            # Train confusion: balanced subset (optionally capped) for tractable inference.
            train_eval_loader = make_dataloader(
                manifest=manifest_path,
                split="train",
                task_cfg=task_cfg,
                datasets=experiment_cfg["train_datasets"],
                runtime_cfg=runtime_cfg,
                shuffle=False,
                balance_subset=True,
                max_per_class=runtime_cfg.get("confusion_max_per_class"),
            )
            _save_confusion(model, train_eval_loader, task_cfg, device, out_dir, "train",
                            max_batches=limit_batches)
            # Val confusion: reuse the balanced validation loader.
            _save_confusion(model, val_loader, task_cfg, device, out_dir, "val",
                            max_batches=limit_batches)
        else:
            log.info("Skipping confusion matrices (task_type=%s)", task_type)

    return out_dir / "best.pt"
