import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_experiment import run_training
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    experiment_cfg = load_yaml(args.experiment)
    runtime_cfg = load_yaml(args.runtime)
    if args.num_workers is not None:
        runtime_cfg["num_workers"] = args.num_workers
    set_seed(runtime_cfg.get("seed", 42))
    ckpt = run_training(model_cfg, task_cfg, experiment_cfg, runtime_cfg, args.manifest)
    print(f"Saved best checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
