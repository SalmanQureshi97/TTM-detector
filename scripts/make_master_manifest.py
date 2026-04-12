import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV files to concatenate")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(Path(path)) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved master manifest to {args.output}")


if __name__ == "__main__":
    main()
