"""Emit training / evaluation / holdout commands for the experimental matrix.

The core study is a 3 (arch) x 4 (config) x 2 (core task) = 24 model sweep.
An optional follow-up task-sensitivity study runs the three auxiliary tasks
on the single best (arch, config) cell selected from the core sweep, adding
3 models for a grand total of 27.

Phases:
  * core  — full factorial over --models, CORE_TASKS, and CORE_CONFIGS.
  * aux   — auxiliary tasks on a single best cell (supply --best-model and
            --best-experiment). Runs AUX_TASKS on that single cell only.

Explicit --models / --tasks / --experiments still override the phase
defaults for custom matrices.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.build_experiment_matrix import build_matrix


# Core tasks: run across the full arch x config matrix.
CORE_TASKS = ["authenticity_binary", "multitask_two_head"]

# Auxiliary tasks: alternative loss formulations of the joint (auth x enc)
# signal. Run only on the single best (arch, config) cell from the core
# sweep, to isolate loss-formulation sensitivity without inflating the matrix.
AUX_TASKS = ["encoding_binary", "four_class_flat", "hierarchical"]

# Kept training configurations after the C2 (FMA+FMC, no-SONICS) drop.
# Naming on disk is unchanged; the thesis renumbers C1/C3/C4/C5 -> C1-C4.
CORE_CONFIGS = [
    "c1_sonics_only",
    "c3_fma_sonics",
    "c4_sonics_fakemusiccaps",
    "c5_all",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate train, eval, and holdout commands.")
    parser.add_argument("--phase", choices=["core", "aux", "custom"], default="core",
                        help="core: full core matrix; aux: aux tasks on best cell; custom: honour --models/--tasks/--experiments verbatim.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list. In 'core'/'aux' phase, required.")
    parser.add_argument("--tasks", default=None,
                        help="Overrides CORE_TASKS / AUX_TASKS if given.")
    parser.add_argument("--experiments", default=None,
                        help="Overrides CORE_CONFIGS if given.")
    parser.add_argument("--best-model", default=None,
                        help="(aux phase) The single winning architecture from the core sweep.")
    parser.add_argument("--best-experiment", default=None,
                        help="(aux phase) The single winning training config from the core sweep.")
    parser.add_argument("--manifest", default="data/manifests/master_manifest_with_splits.csv")
    parser.add_argument("--train-runtime", default="configs/runtime/train_default.yaml")
    parser.add_argument("--eval-runtime", default="configs/runtime/eval_default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "holdout", "all"], default="all",
                        help="Which commands to generate")
    return parser.parse_args()


def _split(csv):
    return [s.strip() for s in csv.split(",") if s.strip()]


def resolve_phase(args):
    """Resolve (models, tasks, experiments) given --phase and overrides."""
    if args.phase == "core":
        if not args.models:
            raise SystemExit("--models is required for --phase core")
        models = _split(args.models)
        tasks = _split(args.tasks) if args.tasks else list(CORE_TASKS)
        experiments = _split(args.experiments) if args.experiments else list(CORE_CONFIGS)
    elif args.phase == "aux":
        if not (args.best_model and args.best_experiment):
            raise SystemExit("--best-model and --best-experiment are required for --phase aux")
        models = [args.best_model]
        tasks = _split(args.tasks) if args.tasks else list(AUX_TASKS)
        experiments = [args.best_experiment]
    else:  # custom
        if not (args.models and args.tasks and args.experiments):
            raise SystemExit("--models, --tasks, --experiments are all required for --phase custom")
        models = _split(args.models)
        tasks = _split(args.tasks)
        experiments = _split(args.experiments)
    return models, tasks, experiments


def main():
    args = parse_args()
    models, tasks, experiments = resolve_phase(args)

    matrix = build_matrix(models, tasks, experiments)

    print(f"# Phase: {args.phase} | {len(models)} models x {len(experiments)} configs x {len(tasks)} tasks = {len(matrix)} planned runs")
    print()

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
