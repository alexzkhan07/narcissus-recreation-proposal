"""Label-Consistent clean-label backdoor baseline.

Turner et al.'s Label-Consistent attack poisons target-class training images
with two ingredients:

  1. a small L_inf-bounded perturbation that makes the clean target features
     less reliable, and
  2. an arbitrary visible trigger that the victim can associate with the
     target label.

The full LC attack usually builds image-specific adversarial perturbations
from a separate surrogate model. For this lightweight Figure-3 recreation
pipeline, we use fresh bounded noise as the surrogate-free perturbation
component and keep the test-time trigger as the visible corner patch only.
The changing perturbation prevents the victim from memorizing one fixed noise
pattern and keeps the patch as the stable backdoor feature.
"""

from __future__ import annotations

from typing import Callable

import torch

from attacks.badnets import make_trigger_fn as make_badnets_trigger_fn

TriggerFn = Callable[[torch.Tensor], torch.Tensor]


def _bounded_noise_like(
    images: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    return torch.empty_like(images).uniform_(-epsilon, epsilon)


def make_trigger_fns(
    epsilon: float = 16 / 255,
    perturb_seed: int | None = None,
    patch_size: int = 3,
    location: str = "bottom-right",
    color: float = 1.0,
    image_size: int = 32,
) -> tuple[TriggerFn, TriggerFn]:
    """Return `(poison_trigger_fn, eval_trigger_fn)` for the LC baseline.

    The poison-time trigger adds the bounded perturbation and then stamps the
    visible patch. The ASR evaluation trigger stamps only the visible patch,
    matching the intended test-time behavior of Label-Consistent attacks.
    """
    _ = perturb_seed  # Kept for CLI compatibility; run-level seeding controls noise.
    patch_trigger_fn = make_badnets_trigger_fn(
        patch_size=patch_size,
        location=location,
        color=color,
    )

    def poison_trigger_fn(images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != image_size or images.shape[-2] != image_size:
            raise ValueError(
                f"label-consistent trigger built for {image_size}x{image_size}, "
                f"got {tuple(images.shape[-2:])}"
            )
        delta = _bounded_noise_like(images, epsilon)
        return patch_trigger_fn((images + delta).clamp(0, 1))

    return poison_trigger_fn, patch_trigger_fn


def make_trigger_fn(**kwargs) -> TriggerFn:
    """Compatibility wrapper returning the poison-time LC trigger."""
    poison_trigger_fn, _ = make_trigger_fns(**kwargs)
    return poison_trigger_fn
