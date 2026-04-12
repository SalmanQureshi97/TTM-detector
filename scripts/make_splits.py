import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.build_splits import assign_group_splits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-col", default="track_id")
    parser.add_argument("--stratify-col", default="class4_label",
                        help="Column to stratify on for balanced class distribution across splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = assign_group_splits(
        df,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in sorted(df["split"].unique()):
        df[df["split"] == split_name].to_csv(out_dir / f"{split_name}.csv", index=False)
    if args.output_manifest:
        output_manifest = Path(args.output_manifest)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_manifest, index=False)
        print(f"Saved manifest with assigned splits to {output_manifest}")
    print(f"Saved split CSVs to {out_dir}")


if __name__ == "__main__":
    main()
