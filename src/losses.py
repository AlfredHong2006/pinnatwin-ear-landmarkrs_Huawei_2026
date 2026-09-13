"""
Role B — losses and training-time monitoring metrics.

SCOPE
-----
Role C owns the OFFICIAL metric. It lives in ``src/evaluate.py``
(``landmark_errors`` / ``summarize_errors``) and is computed in GLOBAL
millimetre coordinates after ``inverse_transform_points``. Nothing in this file
replaces it — ``src/train.py`` calls C's functions directly for any number that
gets reported.

What lives here:
  * ``smooth_l1_loss``  — what training minimises, in canonical space.
  * ``mean_euclidean_error`` — a cheap canonical-space progress readout.
  * ``canonical_to_mm`` — the exact conversion between the two (see below).

THE CANONICAL -> MM RELATIONSHIP IS EXACT, NOT APPROXIMATE
----------------------------------------------------------
Role A's inverse transform (src/geometry.py) is::

    p_global = mirror(p_canonical) * scale + centre

Euclidean distance is invariant under translation (``+ centre``) and under
negating one axis (``mirror``), and scales linearly with ``scale``. Therefore::

    mm_error == canonical_error * transform.scale     (exactly, per ear)

That identity is asserted in tests/test_model.py. It means a canonical-space
number can be converted to the competition's units without guessing — but the
per-ear ``scale`` differs between ears, so a *mean* canonical error cannot be
converted with a single multiply. Use the per-ear path in train.py for anything
you report.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Role A's canonicalisation divides by half the crop-box extent, so canonical
# coordinates sit in roughly [-1, 1] and landmark errors land around 0.01-0.15.
#
# PyTorch's default beta of 1.0 would put every error deep in the quadratic
# region, silently turning Smooth-L1 into plain MSE and discarding the
# robustness that motivated using it. 0.05 puts the elbow near the error scale
# we actually expect.
DEFAULT_BETA = 0.05


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """
    Smooth-L1 (Huber) loss on canonical landmark residuals.

    Quadratic below ``beta``, linear above it. The linear tail stops a handful
    of badly-placed landmarks from dominating the gradient.

    Args:
        pred:   [B, 85, 3] predicted residuals.
        target: [B, 85, 3] true residuals (canonical GT minus template).
    """
    _check_shapes(pred, target)
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="mean")


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain L1 — the 'one simple alternative' the role brief permits."""
    _check_shapes(pred, target)
    return F.l1_loss(pred, target, reduction="mean")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE. Included only so the loss comparison is a fair three-way."""
    _check_shapes(pred, target)
    return F.mse_loss(pred, target, reduction="mean")


@torch.no_grad()
def mean_euclidean_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean straight-line distance between predicted and true landmarks, in
    CANONICAL units. A monitoring proxy, not the official score.

    The template cancels exactly, so measuring on residuals and measuring on
    final canonical coordinates give identical answers::

        (template + pred) - (template + true) = pred - true
    """
    _check_shapes(pred, target)
    return torch.linalg.norm(pred - target, dim=-1).mean()


@torch.no_grad()
def per_landmark_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-landmark mean canonical error, shape [85].

    Diagnostic: the 85 landmarks are four ordered contours (outer helix 0-24,
    concha 25-54, inner helix 55-74, superior antihelix 75-84 — confirm indices
    with Role C). If one contour is consistently worse, that is far more
    actionable than a single averaged number.
    """
    _check_shapes(pred, target)
    return torch.linalg.norm(pred - target, dim=-1).mean(dim=0)


@torch.no_grad()
def per_ear_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-ear mean canonical error, shape [B]. Feeds the mm conversion below."""
    _check_shapes(pred, target)
    return torch.linalg.norm(pred - target, dim=-1).mean(dim=1)


@torch.no_grad()
def canonical_to_mm(error_canonical: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Convert per-ear canonical error to millimetres. EXACT, not approximate —
    see the module docstring for why.

    Args:
        error_canonical: [B] per-ear canonical error (from ``per_ear_error``).
        scale:           [B] per-ear ``transform.scale`` from Role A's cache.
    """
    if error_canonical.shape != scale.shape:
        raise ValueError(
            f"shape mismatch: error {tuple(error_canonical.shape)} vs scale {tuple(scale.shape)}"
        )
    return error_canonical * scale


def get_loss_fn(name: str, beta: float = DEFAULT_BETA):
    """Look up a loss by config name."""
    if name == "smooth_l1":
        return lambda p, t: smooth_l1_loss(p, t, beta=beta)
    if name == "l1":
        return l1_loss
    if name == "mse":
        return mse_loss
    raise ValueError(f"unknown loss {name!r}; expected 'smooth_l1', 'l1' or 'mse'")


def _check_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    if pred.dim() != 3 or pred.shape[-1] != 3:
        raise ValueError(f"expected [B, L, 3], got {tuple(pred.shape)}")