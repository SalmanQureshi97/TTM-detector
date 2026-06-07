# pretrained/

This folder holds locally-stored pretrained checkpoints that the model factory loads
on demand. Files here are intentionally **not committed** (they are typically large
binary blobs). Drop the file you need on the GPU box manually; the path in the
relevant model YAML is resolved relative to the repo root.

## SONICS SpecTTTra ALPHA-120s

Expected file: `pretrained/sonics-spectttra-alpha-120s.bin`

Backbone hyperparams (from the upstream `config.json` that produced it):

| Field | Value |
|---|---|
| Encoder | `SpecTTTra` |
| `input_shape` | `[128, 3744]` |
| `embed_dim` | 384 |
| `num_heads` | 6 |
| `num_layers` | 12 |
| `t_clip` / `f_clip` | 3 / 1 |
| `mlp_ratio` | 2.67 |
| `pe_learnable` | true |
| `pre_norm` | true |
| Audio | 16 kHz, 120 s, mel n_fft 2048 / hop 512 / 128 mels / f_min 20 / f_max 8000 |
| Original task | BCE binary (`num_classes=1`) — head is discarded; only the encoder is loaded |

The matching local model config is `configs/models/sonics_spectttra_alpha120s.yaml`.
