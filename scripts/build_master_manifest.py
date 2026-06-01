"""Build the master manifest CSV from the raw data directory.

Walks the data root and assigns labels, generator, and encoder metadata
based on folder structure and filename conventions.

Directory structure expected:
    data/
    ├── fmc/                          # FakeMusicCaps (folder renamed from FakeMusicCaps/)
    │   ├── audioldm2/
    │   ├── musicgen_medium/
    │   ├── mustango/
    │   └── ...
    ├── fmc_encoded/                  # FakeMusicCaps re-encoded (capped at 10k/codec)
    │   ├── encodec3/
    │   │   ├── audioldm2/
    │   │   ├── musicgen_medium/
    │   ├── encodec6/
    │   ├── encodec24/
    │   ├── griffin256/
    │   └── griffin512/
    ├── fma_real/
    │   └── *.wav / *.mp3 / ...
    ├── fma_encoded/
    │   ├── encodec3/
    │   ├── griffin256/
    │   └── ...
    ├── sonics_real/
    │   └── *.mp3 / ...
    ├── sonics_fake/
    │   └── fake_<id>_<generator>_<variant>.mp3
    ├── sonics_fake_encoded/
    │   ├── encodec3/
    │   ├── griffin256/
    │   └── ...
    └── sonics_real_encoded/
        ├── encodec3/
        ├── griffin256/
        └── ...

The `source_dataset` label for both `fmc/` and `fmc_encoded/` is kept as
"FakeMusicCaps" so existing experiment/dataset YAML configs (which filter
on `source_dataset == "FakeMusicCaps"`) keep working without changes.

Usage:
    python scripts/build_master_manifest.py \
        --data-root /home/jovyan/Thesis/Code/data \
        --output /home/jovyan/Thesis/Code/data/manifests/master_manifest.csv
"""

import argparse
import csv
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

FOLDER_CONFIG = {
    "fmc": {
        "source_dataset": "FakeMusicCaps",
        "auth_label": 1,
        "enc_label": 0,
    },
    "fmc_encoded": {
        "source_dataset": "FakeMusicCaps",
        "auth_label": 1,
        "enc_label": 1,
    },
    "fma_real": {
        "source_dataset": "FMA",
        "auth_label": 0,
        "enc_label": 0,
    },
    "fma_encoded": {
        "source_dataset": "FMA",
        "auth_label": 0,
        "enc_label": 1,
    },
    "sonics_real": {
        "source_dataset": "SONICS",
        "auth_label": 0,
        "enc_label": 0,
    },
    "sonics_fake": {
        "source_dataset": "SONICS",
        "auth_label": 1,
        "enc_label": 0,
    },
    "sonics_fake_encoded": {
        "source_dataset": "SONICS",
        "auth_label": 1,
        "enc_label": 1,
    },
    "sonics_real_encoded": {
        "source_dataset": "SONICS",
        "auth_label": 0,
        "enc_label": 1,
    },
}

FIELDNAMES = [
    "filepath",
    "track_id",
    "source_dataset",
    "split",
    "auth_label",
    "enc_label",
    "class4_label",
    "generator",
    "encoder",
    "duration",
]


def compute_class4(auth, enc):
    return auth * 2 + enc


def extract_generator_from_sonics_filename(stem):
    """Parse generator from SONICS fake filenames.

    Expected format: fake_<id>_<generator>_<variant>
    Example: fake_36081_udio_1 -> udio
    """
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "fake":
        # Rejoin everything between the numeric id and the final variant number
        # to handle generators with underscores (e.g., stable_audio_open)
        # Parts: ['fake', '<id>', '<gen_part1>', ..., '<gen_partN>', '<variant>']
        return "_".join(parts[2:-1])
    return "unknown"


def extract_track_id_from_sonics(stem):
    """Extract a stable track_id from SONICS filenames.

    fake_36081_udio_1 -> 36081
    This ensures all variants of the same track (across generators and
    encoded versions) share the same track_id for leakage-safe splitting.
    """
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "fake":
        return parts[1]
    return stem


def get_top_folder(path, data_root):
    return path.relative_to(data_root).parts[0]


def get_subfolder(path, data_root, depth=1):
    """Get the subfolder at a given depth relative to the top-level folder.

    For data_root/fma_encoded/encodec3/track.wav with depth=1, returns 'encodec3'.
    For data_root/fmc/audioldm2/track.wav with depth=1, returns 'audioldm2'.
    For data_root/fmc_encoded/encodec3/audioldm2/track.wav with depth=2, returns 'audioldm2'.
    """
    rel_parts = path.relative_to(data_root).parts
    if len(rel_parts) > depth:
        return rel_parts[depth]
    return ""


def build_manifest(data_root, output_path):
    data_root = Path(data_root)
    rows = []
    skipped = 0

    for file_path in sorted(data_root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        # Skip Jupyter checkpoint artifacts
        if ".ipynb_checkpoints" in file_path.parts:
            skipped += 1
            continue

        try:
            top_folder = get_top_folder(file_path, data_root)
        except Exception:
            skipped += 1
            continue

        if top_folder not in FOLDER_CONFIG:
            skipped += 1
            continue

        config = FOLDER_CONFIG[top_folder]
        auth_label = config["auth_label"]
        enc_label = config["enc_label"]
        class4_label = compute_class4(auth_label, enc_label)
        stem = file_path.stem

        # --- Generator ---
        if top_folder == "sonics_fake":
            generator = extract_generator_from_sonics_filename(stem)
            track_id = extract_track_id_from_sonics(stem)
        elif top_folder == "sonics_fake_encoded":
            # Files inside encoder subfolders, but filenames still follow
            # the fake_<id>_<generator>_<variant> convention
            generator = extract_generator_from_sonics_filename(stem)
            track_id = extract_track_id_from_sonics(stem)
        elif top_folder == "fmc":
            # Generator is the subfolder name (e.g., audioldm2, musicgen_medium)
            generator = get_subfolder(file_path, data_root, depth=1)
            track_id = stem
        elif top_folder == "fmc_encoded":
            # Layout: fmc_encoded/<encoder>/<generator>/<file>
            # The encoder lives at depth=1 and the generator at depth=2,
            # so the encoded variant inherits the same generator and the
            # same track_id (stem) as its fmc/ source for group-safe splits.
            generator = get_subfolder(file_path, data_root, depth=2)
            track_id = stem
        else:
            # Real audio (FMA, SONICS real)
            generator = "real"
            track_id = stem

        # --- Encoder ---
        if top_folder in {"fma_encoded", "sonics_fake_encoded",
                          "sonics_real_encoded", "fmc_encoded"}:
            encoder = get_subfolder(file_path, data_root, depth=1)
        else:
            encoder = "none"

        row = {
            "filepath": str(file_path.resolve()),
            "track_id": track_id,
            "source_dataset": config["source_dataset"],
            "split": "",
            "auth_label": auth_label,
            "enc_label": enc_label,
            "class4_label": class4_label,
            "generator": generator,
            "encoder": encoder,
            "duration": "",
        }
        rows.append(row)

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"Manifest created with {len(rows)} entries ({skipped} files skipped)")
    print(f"Saved to: {output_path}")
    print()

    # Print breakdown
    from collections import Counter
    ds_counts = Counter(r["source_dataset"] for r in rows)
    gen_counts = Counter(r["generator"] for r in rows)
    enc_counts = Counter(r["encoder"] for r in rows)

    print("--- By source_dataset ---")
    for k, v in sorted(ds_counts.items()):
        print(f"  {k}: {v}")

    print("--- By generator ---")
    for k, v in sorted(gen_counts.items()):
        print(f"  {k}: {v}")

    print("--- By encoder ---")
    for k, v in sorted(enc_counts.items()):
        print(f"  {k}: {v}")

    print("--- By class4_label ---")
    c4_counts = Counter(r["class4_label"] for r in rows)
    labels = {0: "real_not_encoded", 1: "real_encoded", 2: "fake_not_encoded", 3: "fake_encoded"}
    for k in sorted(c4_counts):
        print(f"  {labels.get(k, k)}: {c4_counts[k]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build master manifest from data directory.")
    parser.add_argument("--data-root", required=True, help="Root data directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_manifest(args.data_root, args.output)
