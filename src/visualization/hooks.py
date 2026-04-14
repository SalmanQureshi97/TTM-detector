"""Hook utilities for extracting activations, gradients, and attention weights."""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers to locate modules by type
# ---------------------------------------------------------------------------

def find_attention_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Walk the module tree and return all Attention instances (from sonics)."""
    results = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == "Attention" and hasattr(module, "qkv"):
            results.append((name, module))
    return results


def find_last_conv2d(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """Return the (name, module) of the last Conv2d in the model."""
    last_name, last_mod = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name, last_mod = name, module
    if last_name is None:
        raise ValueError("No Conv2d found in model")
    return last_name, last_mod


# ---------------------------------------------------------------------------
# Feature extractor (forward hooks)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Register forward hooks on named layers to capture their output tensors.

    Usage::

        extractor = FeatureExtractor(model, ["backbone.features.5"])
        output = model(x)
        feature_maps = extractor.get()   # {"backbone.features.5": Tensor}
        extractor.remove()
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        for name in layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            self._features[name] = output.detach()
        return hook_fn

    def get(self) -> dict[str, torch.Tensor]:
        return dict(self._features)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Gradient extractor (backward hooks on activations)
# ---------------------------------------------------------------------------

class GradientExtractor:
    """Register hooks that capture the gradient flowing through named layers.

    The gradient is captured on the *output* of each layer, which requires
    storing the activation with ``requires_grad`` and using ``register_hook``
    on the tensor itself.

    Usage::

        grad_ext = GradientExtractor(model, ["backbone.features.5"])
        output = model(x)
        output.backward(...)
        grads = grad_ext.get()  # {"backbone.features.5": Tensor}
        grad_ext.remove()
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self._grads: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[str, torch.Tensor] = {}
        for name in layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            if isinstance(output, torch.Tensor):
                act = output
            else:
                act = output[0]
            act.retain_grad()
            self._activations[name] = act
            act.register_hook(lambda g, n=name: self._grads.__setitem__(n, g.detach()))
        return hook_fn

    def get_activations(self) -> dict[str, torch.Tensor]:
        return dict(self._activations)

    def get(self) -> dict[str, torch.Tensor]:
        return dict(self._grads)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Attention extractor (monkey-patches Attention.forward)
# ---------------------------------------------------------------------------

def _patched_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for sonics Attention.forward that stores weights."""
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    self._saved_attn = attn.detach()  # <-- store for visualization
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class AttentionExtractor:
    """Extract attention weight matrices from all transformer layers.

    Works by temporarily monkey-patching each ``Attention`` module's
    ``forward`` method with a version that stores the attention matrix
    on the module as ``_saved_attn``.

    This is necessary because when ``fused_attn=True`` (the default on
    modern hardware), ``F.scaled_dot_product_attention`` does not expose
    the attention weight matrix.

    Usage::

        extractor = AttentionExtractor(model)
        output = model(x)
        attn_weights = extractor.get()   # list[Tensor[B, H, N, N]]
        extractor.remove()
    """

    def __init__(self, model: nn.Module) -> None:
        self._attn_modules: list[tuple[str, nn.Module]] = find_attention_modules(model)
        self._originals: list[tuple[nn.Module, bool, types.MethodType]] = []

        for _name, module in self._attn_modules:
            # Save originals so we can restore later
            orig_fused = getattr(module, "fused_attn", False)
            orig_forward = module.forward
            self._originals.append((module, orig_fused, orig_forward))

            # Patch
            module.fused_attn = False
            module.forward = types.MethodType(_patched_attention_forward, module)

    def get(self) -> list[torch.Tensor]:
        """Return attention weights [B, num_heads, N, N] per layer, in order."""
        weights = []
        for _name, module in self._attn_modules:
            if hasattr(module, "_saved_attn"):
                weights.append(module._saved_attn)
        return weights

    def remove(self) -> None:
        """Restore original forward methods and fused_attn flags."""
        for (module, orig_fused, orig_forward) in self._originals:
            module.fused_attn = orig_fused
            module.forward = orig_forward
            if hasattr(module, "_saved_attn"):
                del module._saved_attn
        self._originals.clear()
