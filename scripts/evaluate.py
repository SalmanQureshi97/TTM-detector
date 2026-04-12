import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset
from src.evaluation.metrics import binary_metrics, four_class_projected_metrics, multitask_metrics
from src.evaluation.robustness import apply_bandpass
from src.models.model_factory import UnifiedAudioModel
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--bandpass-low", type=float, default=None)
    parser.add_argument("--bandpass-high", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    experiment_cfg = load_yaml(args.experiment)
    runtime_cfg = load_yaml(args.runtime)
    batch_size = args.batch_size or runtime_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else runtime_cfg["num_workers"]

    device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    summary = {}
    for suite in experiment_cfg["test_suites"]:
        ds = AudioManifestDataset(
            manifest_path=args.manifest,
            split=suite["split"],
            task_cfg=task_cfg,
            dataset_filter=suite["datasets"],
            sample_rate=44100,
            max_seconds=runtime_cfg["segment_seconds"],
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        with torch.no_grad():
            if task_cfg["type"] == "binary":
                y_true, y_prob = [], []
                for batch in loader:
                    audio = batch["audio"].to(device)
                    if args.bandpass_low is not None or args.bandpass_high is not None:
                        audio = apply_bandpass(audio, 44100, args.bandpass_low, args.bandpass_high)
                    output = model(audio)
                    y_true.extend(batch["target"].cpu().numpy().astype(int).tolist())
                    y_prob.extend(torch.sigmoid(output).cpu().numpy().tolist())
                metrics = binary_metrics(np.array(y_true), np.array(y_prob), runtime_cfg["threshold"])
            elif task_cfg["type"] in {"multitask", "hierarchical"}:
                auth_true, auth_prob, enc_true, enc_prob = [], [], [], []
                for batch in loader:
                    audio = batch["audio"].to(device)
                    if args.bandpass_low is not None or args.bandpass_high is not None:
                        audio = apply_bandpass(audio, 44100, args.bandpass_low, args.bandpass_high)
                    output = model(audio)
                    auth_true.extend(batch["target"]["auth"].cpu().numpy().astype(int).tolist())
                    enc_true.extend(batch["target"]["enc"].cpu().numpy().astype(int).tolist())
                    auth_prob.extend(torch.sigmoid(output["auth"]).cpu().numpy().tolist())
                    enc_prob.extend(torch.sigmoid(output["enc"]).cpu().numpy().tolist())
                metrics = multitask_metrics(
                    auth_true=np.array(auth_true),
                    auth_prob=np.array(auth_prob),
                    enc_true=np.array(enc_true),
                    enc_prob=np.array(enc_prob),
                    threshold=runtime_cfg["threshold"],
                )
            elif task_cfg["type"] == "multiclass":
                y_true, y_prob = [], []
                for batch in loader:
                    audio = batch["audio"].to(device)
                    if args.bandpass_low is not None or args.bandpass_high is not None:
                        audio = apply_bandpass(audio, 44100, args.bandpass_low, args.bandpass_high)
                    output = model(audio)
                    y_true.extend(batch["target"].cpu().numpy().astype(int).tolist())
                    y_prob.extend(torch.softmax(output, dim=1).cpu().numpy().tolist())
                metrics = four_class_projected_metrics(
                    y_true=np.array(y_true),
                    y_prob=np.array(y_prob),
                    threshold=runtime_cfg["threshold"],
                )
            else:
                raise ValueError(f"Unsupported task type: {task_cfg['type']}")

        summary[suite["name"]] = metrics
        print(suite["name"], json.dumps(metrics, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
