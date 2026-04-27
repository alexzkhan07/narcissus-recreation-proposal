"""BadNets-c (clean-label BadNets): stamps a small fixed pixel pattern in
the corner of every input image. Used as a clean-label baseline by overlaying
the pattern onto target-class training images without changing their labels.

Reference: Gu et al. "BadNets: Identifying Vulnerabilities in the Machine
Learning Model Supply Chain." 2017.
"""

from __future__ import annotations

import torch

TriggerFn = "Callable[[torch.Tensor], torch.Tensor]"


def make_trigger_fn(
    patch_size: int = 3,
    location: str = "bottom-right",
    color: float = 1.0,
) -> TriggerFn:
    """Return a `trigger_fn` that stamps a `patch_size`x`patch_size` patch.

    Args:
        patch_size: side length in pixels (3 in the original paper).
        location: one of {"bottom-right", "bottom-left", "top-right", "top-left"}.
        color: pixel value in [0,1] for all 3 channels (1.0 = white).
    """
    if patch_size <= 0 or patch_size > 32:
        raise ValueError(f"patch_size must be in [1, 32], got {patch_size}")

    def _slice(img_size: int) -> tuple[slice, slice]:
        if location == "bottom-right":
            return slice(img_size - patch_size, img_size), slice(img_size - patch_size, img_size)
        if location == "bottom-left":
            return slice(img_size - patch_size, img_size), slice(0, patch_size)
        if location == "top-right":
            return slice(0, patch_size), slice(img_size - patch_size, img_size)
        if location == "top-left":
            return slice(0, patch_size), slice(0, patch_size)
        raise ValueError(f"unknown location: {location}")

    def trigger_fn(images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        h, w = out.shape[-2], out.shape[-1]
        if h != w:
            raise ValueError(f"expected square images, got {h}x{w}")
        rs, cs = _slice(h)
        out[..., rs, cs] = color
        return out

    return trigger_fn
