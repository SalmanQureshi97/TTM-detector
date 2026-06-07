from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
SONICS_ROOT = WORKSPACE_ROOT / "sonics"
if str(SONICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SONICS_ROOT))

from sonics.models.spectttra import SpecTTTra  # noqa: E402


# SpecTTTra constructor kwargs that we forward from the YAML if present.
_OPTIONAL_KWARGS = (
    "mlp_ratio",
    "attn_drop_rate",
    "pos_drop_rate",
    "proj_drop_rate",
)


class SpectTTTraBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_kwargs = dict(
            input_spec_dim=cfg["input_shape"][0],
            input_temp_dim=cfg["input_shape"][1],
            embed_dim=cfg["embed_dim"],
            t_clip=cfg["t_clip"],
            f_clip=cfg["f_clip"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            pre_norm=cfg.get("pre_norm", False),
            pe_learnable=cfg.get("pe_learnable", False),
        )
        for k in _OPTIONAL_KWARGS:
            if k in cfg:
                encoder_kwargs[k] = cfg[k]
        self.encoder = SpecTTTra(**encoder_kwargs)
        self.embed_dim = cfg["embed_dim"]

        weights_path = cfg.get("pretrained_weights")
        if weights_path:
            self._load_pretrained(weights_path)

        self.frozen = bool(cfg.get("freeze", False))
        if self.frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def _resolve_weights_path(self, weights_path):
        p = Path(weights_path)
        if p.is_absolute():
            return p
        # Repo-root-relative (e.g. ``pretrained/sonics-spectttra-alpha-120s.bin``)
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / p

    def _unwrap_state(self, raw):
        """Strip common checkpoint wrappers down to a flat key->tensor dict."""
        if isinstance(raw, dict):
            for key in ("state_dict", "model", "model_state", "weights"):
                inner = raw.get(key)
                if isinstance(inner, dict) and any(
                    isinstance(v, torch.Tensor) for v in inner.values()
                ):
                    return inner
        return raw

    def _load_pretrained(self, weights_path):
        path = self._resolve_weights_path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SpecTTTra pretrained weights not found at {path}. "
                "Drop the .bin in the repo's pretrained/ folder."
            )
        raw = torch.load(path, map_location="cpu")
        state = self._unwrap_state(raw)

        own_keys = set(self.encoder.state_dict().keys())
        # The upstream SONICS model usually wraps SpecTTTra as a submodule
        # (``spectttra.xxx`` or ``backbone.xxx``); try several prefixes and
        # pick whichever yields the most matching keys.
        candidate_prefixes = (
            "",
            "spectttra.",
            "encoder.",
            "backbone.",
            "model.",
            "module.",
            "model.spectttra.",
            "model.encoder.",
            "model.backbone.",
        )
        best = (None, None, -1)
        for prefix in candidate_prefixes:
            cleaned = {
                (k[len(prefix):] if prefix and k.startswith(prefix) else k): v
                for k, v in state.items()
                if (not prefix) or k.startswith(prefix)
            }
            match = sum(1 for k in cleaned if k in own_keys)
            if match > best[2]:
                best = (prefix, cleaned, match)

        prefix, cleaned, match = best
        if match == 0:
            sample = list(state.keys())[:5]
            sample_own = list(own_keys)[:5]
            raise RuntimeError(
                f"Could not match any SpecTTTra keys from {path.name}.\n"
                f"  Sample checkpoint keys: {sample}\n"
                f"  Sample encoder keys: {sample_own}"
            )

        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        print(
            f"[SpecTTTra] loaded {path.name}: matched {match}/{len(own_keys)} encoder keys "
            f"(prefix={prefix!r}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # ------------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        # Keep the encoder in eval mode when frozen so dropout/etc. are off.
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        return self.encoder(x).mean(dim=1)
