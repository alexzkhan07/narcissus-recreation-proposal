"""Blend attack: alpha-blend a fixed pattern across the entire image.

The pattern is a deterministic pseudo-random tensor seeded for reproducibility,
so the same `pattern_seed` always produces the same trigger across runs.

Reference: Chen et al. "Targeted Backdoor Attacks on Deep Learning Systems
Using Data Poisoning." 2017.
"""

from __future__ import annotations

import torch

TriggerFn = "Callable[[torch.Tensor], torch.Tensor]"


def make_trigger_fn(
    alpha: float = 0.2,
    pattern_seed: int = 1234,
    image_size: int = 32,
) -> TriggerFn:
    """Return a `trigger_fn` that returns `(1 - alpha) * x + alpha * pattern`.

    Args:
        alpha: blend strength in [0, 1]. Original Chen et al. paper uses 0.2.
        pattern_seed: seed for the fixed pattern (any integer).
        image_size: spatial size; the returned trigger_fn will only accept
            images of this size, since the pattern is precomputed.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    g = torch.Generator().manual_seed(pattern_seed)
    pattern = torch.rand((1, 3, image_size, image_size), generator=g)

    def trigger_fn(images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != image_size or images.shape[-2] != image_size:
            raise ValueError(
                f"blend trigger built for {image_size}x{image_size}, "
                f"got {tuple(images.shape[-2:])}"
            )
        p = pattern.to(images.device, images.dtype)
        return (1.0 - alpha) * images + alpha * p

    return trigger_fn
