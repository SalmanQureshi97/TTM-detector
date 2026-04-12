# AI Music Generalization Benchmark

Unified benchmark repository for the experimental design in:

- generalized AI-generated music detection
- authenticity vs encoding factorization
- cross-domain evaluation across FMA, FakeMusicCaps, and SONICS
- generator-holdout and encoder-holdout testing
- robustness testing under bandpass filtering

This repository is designed to operationalize the experimental design directly. It provides:

- a manifest-driven dataset layer
- task definitions for:
  - `authenticity_binary`
  - `encoding_binary`
  - `multitask_two_head`
  - `four_class_flat`
  - `hierarchical`
- model definitions for:
  - `deezer_speccnn_amplitude`
  - `sonics_vit`
  - `sonics_spectttra`
- experiment configs for:
  - `C1` SONICS only
  - `C2` FMA + FakeMusicCaps
  - `C3` FMA + SONICS
  - `C4` SONICS + FakeMusicCaps
  - `C5` All datasets
  - `S5` generator holdout template
  - `S6` encoder holdout template

The repository is structured around the exact task modes in the experimental design:

- `authenticity_binary`
- `encoding_binary`
- `multitask_two_head`
- `four_class_flat`
- `hierarchical`

## Repository Structure

```text
ai-music-generalization/
├── configs/
├── data/
├── scripts/
├── src/
└── outputs/
```

## Setup

```bash
cd ai-music-generalization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The SONICS backbones expect the local workspace layout to include the `sonics/` repository as a sibling of this repo:

```text
Code/
├── ai-music-generalization/
└── sonics/
```

## Manifest Schema

All experiments operate on a single master manifest CSV. The expected columns are:

```text
filepath,track_id,source_dataset,split,auth_label,enc_label,class4_label,generator,encoder,duration
```

Required columns:

- `filepath`
- `track_id`
- `source_dataset`
- `split`
- `auth_label`
- `enc_label`
- `class4_label`

Optional but recommended:

- `generator`
- `encoder`
- `duration`

### Label Conventions

- `auth_label`: `0=real`, `1=fake`
- `enc_label`: `0=not_encoded`, `1=encoded`
- `class4_label`:
  - `0=real_not_encoded`
  - `1=real_encoded`
  - `2=fake_not_encoded`
  - `3=fake_encoded`

## Main Workflows

### 0. Assign leakage-safe train/val/test splits by `track_id`

This keeps all variants of the same track in the same split, which is the core leakage-control rule in the design.

```bash
python scripts/make_splits.py \
  --manifest data/manifests/master_manifest.csv \
  --output-dir data/manifests/splits/default \
  --output-manifest data/manifests/master_manifest_with_splits.csv \
  --group-col track_id \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 1. Train a single experiment

```bash
python scripts/train.py \
  --model configs/models/sonics_spectttra.yaml \
  --task configs/tasks/multitask_two_head.yaml \
  --experiment configs/experiments/c1_sonics_only.yaml \
  --runtime configs/runtime/train_default.yaml \
  --manifest data/manifests/master_manifest_with_splits.csv
```

### 2. Evaluate a trained checkpoint

```bash
python scripts/evaluate.py \
  --model configs/models/sonics_spectttra.yaml \
  --task configs/tasks/multitask_two_head.yaml \
  --experiment configs/experiments/c1_sonics_only.yaml \
  --runtime configs/runtime/eval_default.yaml \
  --manifest data/manifests/master_manifest_with_splits.csv \
  --checkpoint outputs/c1_sonics_only/sonics_spectttra/multitask_two_head/best.pt
```

### 2a. Evaluate robustness under bandpass filtering

```bash
python scripts/evaluate.py \
  --model configs/models/sonics_spectttra.yaml \
  --task configs/tasks/multitask_two_head.yaml \
  --experiment configs/experiments/c1_sonics_only.yaml \
  --runtime configs/runtime/eval_default.yaml \
  --manifest data/manifests/master_manifest_with_splits.csv \
  --checkpoint outputs/c1_sonics_only/sonics_spectttra/multitask_two_head/best.pt \
  --bandpass-low 3000 \
  --bandpass-high 8000 \
  --output-json outputs/c1_sonics_only/sonics_spectttra/multitask_two_head/bandpass_3_8khz.json
```

### 3. Generate the experiment matrix

```bash
python scripts/run_matrix.py \
  --models sonics_spectttra,sonics_vit,deezer_speccnn_amplitude \
  --tasks authenticity_binary,multitask_two_head,four_class_flat \
  --experiments c1_sonics_only,c2_fma_fakemusiccaps,c3_fma_sonics,c4_sonics_fakemusiccaps,c5_all
```

By default, `run_matrix.py` prints the commands that should be executed. It can also be extended to submit jobs.

## Experiment Config Conventions

The experiment YAMLs are now written to mirror the thesis notation directly.

- `notation` maps each config to the document notation such as `C1`, `C2`, or `S5`
- `description` gives the benchmark role of that configuration
- `study_question` explains what that configuration is meant to test
- `available_suites` lists the thesis suite IDs relevant to the config
- `test_suites` contains the runnable evaluation suites that the current scripts execute directly
- `advanced_suites` contains thesis-aligned suites that require additional manifest generation or retraining logic, such as generator holdout and encoder holdout

This split is deliberate:

- `test_suites` is for things the current training and evaluation scripts can run immediately
- `advanced_suites` is for the next orchestration layer, where we build explicit `S5` and `S6` jobs from generator and encoder metadata in the manifest

In practice, that means:

- `C1` to `C5` are now readable as thesis configs rather than just dataset filters
- `S1` to `S4` are represented directly where runnable
- `S5` and `S6` are represented as templates and metadata, ready for the next automation step

## Design Notes

- The SONICS `SpecTTTra` and `ViT` backbones are wrapped into a shared PyTorch training pipeline.
- The Deezer detector is represented here as a PyTorch spectrogram CNN configured to match the `specnn_amplitude` setup as closely as possible at the benchmark level.
- The repository is designed around a common training and evaluation interface so all architectures can be compared under the same task definitions, splits, metrics, and robustness conditions.

## Status

This repository is now a runnable benchmark scaffold. The grouped split assignment, shared train loop, task-aware evaluation, and robustness hooks are in place. The next step is filling the master manifest with the real thesis datasets and then extending the holdout orchestration for generator-specific and encoder-specific retraining runs.
