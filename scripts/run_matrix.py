import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.build_experiment_matrix import build_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Generate train, eval, and holdout commands.")
    parser.add_argument("--models", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--manifest", default="data/manifests/master_manifest_with_splits.csv")
    parser.add_argument("--train-runtime", default="configs/runtime/train_default.yaml")
    parser.add_argument("--eval-runtime", default="configs/runtime/eval_default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "holdout", "all"], default="all",
                        help="Which commands to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    matrix = build_matrix(models, tasks, experiments)

    if args.mode in ("train", "all"):
        print("# ===== TRAINING COMMANDS =====")
        for row in matrix:
            print(
                "python scripts/train.py "
                f"--model configs/models/{row['model']}.yaml "
                f"--task configs/tasks/{row['task']}.yaml "
                f"--experiment configs/experiments/{row['experiment']}.yaml "
                f"--runtime {args.train_runtime} "
                f"--manifest {args.manifest}"
            )
        print()

    if args.mode in ("eval", "all"):
        print("# ===== EVALUATION COMMANDS =====")
        for row in matrix:
            ckpt = f"outputs/{row['experiment']}/{row['model']}/{row['task']}/best.pt"
            print(
                "python scripts/evaluate.py "
                f"--model configs/models/{row['model']}.yaml "
                f"--task configs/tasks/{row['task']}.yaml "
                f"--experiment configs/experiments/{row['experiment']}.yaml "
                f"--runtime {args.eval_runtime} "
                f"--manifest {args.manifest} "
                f"--checkpoint {ckpt} "
                f"--output-json outputs/{row['experiment']}/{row['model']}/{row['task']}/eval_results.json"
            )
        print()

    if args.mode in ("holdout", "all"):
        print("# ===== HOLDOUT COMMANDS (S5 generator, S6 encoder) =====")
        for model in models:
            for task in tasks:
                for base_exp in experiments:
                    # S5 generator holdout
                    print(
                        "python scripts/run_holdout.py "
                        f"--holdout-config configs/experiments/generator_holdout.yaml "
                        f"--base-experiment configs/experiments/{base_exp}.yaml "
                        f"--model configs/models/{model}.yaml "
                        f"--task configs/tasks/{task}.yaml "
                        f"--runtime {args.train_runtime} "
                        f"--eval-runtime {args.eval_runtime} "
                        f"--manifest {args.manifest}"
                    )
                    # S6 encoder holdout
                    print(
                        "python scripts/run_holdout.py "
                        f"--holdout-config configs/experiments/encoder_holdout.yaml "
                        f"--base-experiment configs/experiments/{base_exp}.yaml "
                        f"--model configs/models/{model}.yaml "
                        f"--task configs/tasks/{task}.yaml "
                        f"--runtime {args.train_runtime} "
                        f"--eval-runtime {args.eval_runtime} "
                        f"--manifest {args.manifest}"
                    )
        print()


if __name__ == "__main__":
    main()
