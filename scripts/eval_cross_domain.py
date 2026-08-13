"""Generate the in-domain and cross-domain confusion matrices for a checkpoint.

Training only writes ``confusion_train.csv`` and ``confusion_val.csv`` when it
finishes (see the post-training block in ``run_experiment.py``). The
cross-domain slices the thesis reports, FMA and the two FakeMusicCaps
partitions, were produced outside the repository and there was no committed
script for them. This is that script, so the numbers can be regenerated from
any checkpoint without waiting for early stopping.

It deliberately reuses ``run_experiment``'s own ``_save_confusion`` and
``make_dataloader`` rather than reimplementing them, so the CSV and PNG it
writes are byte-identical in format to the ones already in the output
directories and drop straight into the existing tables.

The FakeMusicCaps slices need a filter the dataloader does not expose, since
``AudioManifestDataset`` selects on split and source dataset only. For those
the dataset is built directly and its dataframe filtered on ``class4_label``
before it is wrapped, which is the same thing ``balance_subset`` does upstream.

Example:

    python scripts/eval_cross_domain.py \
      --model configs/models/deezer_speccnn_amplitude_standardised.yaml \
      --task configs/tasks/four_class_flat_smooth.yaml \
      --experiment configs/experiments/c1_sonics_only.yaml \
      --runtime configs/runtime/train_robust_speccnn.yaml \
      --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_with_splits_chunked.csv \
      --checkpoint outputs/c1_sonics_only/deezer_speccnn_amplitude_standardised/four_class_flat_smooth/best.pt \
      --num-workers 4 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset
from src.experiments.run_experiment import _save_confusion, make_dataloader
from src.models.model_factory import UnifiedAudioModel
from src.utils.config import load_yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default=None,
                   help="Default: the directory the checkpoint lives in.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap batches per slice. For a smoke test only.")
    return p.parse_args()


def class4_loader(manifest, split, datasets, class4, task_cfg, runtime_cfg,
                  batch_size, num_workers):
    """A loader for one four-class label within one split and dataset."""
    ds = AudioManifestDataset(
        manifest_path=manifest,
        split=split,
        task_cfg=task_cfg,
        dataset_filter=datasets,
        sample_rate=runtime_cfg.get("sample_rate", 44100),
        max_seconds=runtime_cfg["segment_seconds"],
    )
    ds.df = ds.df[ds.df["class4_label"] == class4].reset_index(drop=True)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)


def main():
    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    experiment_cfg = load_yaml(args.experiment)
    runtime_cfg = load_yaml(args.runtime)

    # make_dataloader reads these off runtime_cfg, so override them there
    # rather than passing them down; otherwise the flags reach only the two
    # class4 loaders below and the val/FMA/FMC-all loaders keep the training
    # batch size.
    if args.batch_size is not None:
        runtime_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        runtime_cfg["num_workers"] = args.num_workers
    batch_size = runtime_cfg["batch_size"]
    num_workers = runtime_cfg["num_workers"]

    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    # Training runs under fp16 autocast (amp in the runtime config), which is
    # what makes its batch size fit. Evaluating in fp32 needs roughly twice
    # the activation memory and OOMs on the first conv at 30 s / 44.1 kHz, so
    # match the training precision here.
    use_amp = bool(runtime_cfg.get("amp", False)) and device.type == "cuda"
    # Pinned host memory buys faster H2D copies during training; at eval the
    # gain is marginal and the pages are unswappable, so drop it.
    runtime_cfg["pin_memory"] = False
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {args.checkpoint}"
          + (f" (epoch {ckpt['epoch']}, val_loss {ckpt.get('val_loss', float('nan')):.4f})"
             if "epoch" in ckpt else ""))
    print(f"writing to {out_dir}\n")

    def save(loader, name):
        """Score one slice, then tear the loader down.

        make_dataloader sets persistent_workers, so a loader that stays in
        scope keeps its worker processes alive, and every worker holds its
        own copy of the manifest dataframe. Holding all five slices at once
        multiplies that until the host runs out of RAM, so each loader is
        released as soon as its matrix is written.
        """
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _save_confusion(model, loader, task_cfg, device, out_dir, name,
                                max_batches=args.max_batches)
        finally:
            it = getattr(loader, "_iterator", None)
            if it is not None:
                loader._iterator = None
                del it
            del loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # In-domain validation, built exactly as training builds it so the matrix
    # is comparable with the one training would have written itself.
    val_loader = make_dataloader(
        manifest=args.manifest, split="val", task_cfg=task_cfg,
        datasets=experiment_cfg["val_datasets"], runtime_cfg=runtime_cfg,
        shuffle=False, balance_subset=runtime_cfg.get("balanced_val", False),
    )
    print(f"val                  n = {len(val_loader.dataset):,}")
    save(val_loader, "val")
    del val_loader

    # Out of domain: FMA holds only real audio, FakeMusicCaps only generated.
    fma_loader = make_dataloader(
        manifest=args.manifest, split="test", task_cfg=task_cfg,
        datasets=["FMA"], runtime_cfg=runtime_cfg, shuffle=False,
    )
    print(f"fma_test             n = {len(fma_loader.dataset):,}")
    save(fma_loader, "fma_test")
    del fma_loader

    for name, class4 in [("fmc_natural_test", 2), ("fmc_encoded_test", 3)]:
        ds, loader = class4_loader(
            args.manifest, "test", ["FakeMusicCaps"], class4,
            task_cfg, runtime_cfg, batch_size, num_workers,
        )
        if len(ds) == 0:
            print(f"{name:20} EMPTY, skipping. This manifest has no "
                  f"FakeMusicCaps rows with class4_label={class4}.")
            continue
        print(f"{name:20} n = {len(ds):,}")
        save(loader, name)
        del loader, ds

    fmc_all = make_dataloader(
        manifest=args.manifest, split="test", task_cfg=task_cfg,
        datasets=["FakeMusicCaps"], runtime_cfg=runtime_cfg, shuffle=False,
    )
    print(f"fmc_test_all         n = {len(fmc_all.dataset):,}")
    save(fmc_all, "fmc_test_all")
    del fmc_all

    print("\ndone. Written:")
    for f in sorted(out_dir.glob("confusion_*.csv")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
