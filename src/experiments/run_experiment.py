from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.manifest_dataset import AudioManifestDataset
from src.losses.four_class import four_class_loss
from src.losses.hierarchical import hierarchical_loss
from src.losses.multitask import multitask_loss
from src.models.model_factory import UnifiedAudioModel
from src.utils.io import ensure_dir

log = logging.getLogger(__name__)


def make_dataloader(manifest, split, task_cfg, datasets, runtime_cfg, shuffle):
    ds = AudioManifestDataset(
        manifest_path=manifest,
        split=split,
        task_cfg=task_cfg,
        dataset_filter=datasets,
        sample_rate=44100,
        max_seconds=runtime_cfg["segment_seconds"],
    )
    return DataLoader(
        ds,
        batch_size=runtime_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=runtime_cfg["num_workers"],
    )


def compute_loss(task_cfg, outputs, targets):
    if task_cfg["type"] == "binary":
        return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
    if task_cfg["type"] == "multiclass":
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


def run_training(model_cfg, task_cfg, experiment_cfg, runtime_cfg, manifest_path):
    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s | Trainable params: %s", model_cfg["name"], f"{param_count:,}")

    train_loader = make_dataloader(
        manifest=manifest_path,
        split="train",
        task_cfg=task_cfg,
        datasets=experiment_cfg["train_datasets"],
        runtime_cfg=runtime_cfg,
        shuffle=True,
    )
    val_loader = make_dataloader(
        manifest=manifest_path,
        split="val",
        task_cfg=task_cfg,
        datasets=experiment_cfg["val_datasets"],
        runtime_cfg=runtime_cfg,
        shuffle=False,
    )
    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=runtime_cfg["learning_rate"],
        weight_decay=runtime_cfg["weight_decay"],
    )

    epochs = runtime_cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    grad_clip = runtime_cfg.get("grad_clip", 1.0)
    patience = runtime_cfg.get("early_stopping_patience", 5)

    out_dir = ensure_dir(
        Path(runtime_cfg["save_dir"]) / experiment_cfg["name"] / model_cfg["name"] / task_cfg["name"]
    )

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for batch in pbar:
            audio = batch["audio"].to(device)
            target = move_target_to_device(batch["target"], device)
            optimizer.zero_grad()
            outputs = model(audio)
            loss = compute_loss(task_cfg, outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                audio = batch["audio"].to(device)
                target = move_target_to_device(batch["target"], device)
                outputs = model(audio)
                val_loss += compute_loss(task_cfg, outputs, target).item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | time=%.1fs",
            epoch, epochs, avg_train_loss, avg_val_loss, lr, elapsed,
        )

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

        if epochs_without_improvement >= patience:
            log.info("Early stopping triggered after %d epochs.", epoch)
            break

    log.info("Training complete. Best val_loss=%.4f saved to %s", best_val, out_dir / "best.pt")
    return out_dir / "best.pt"
