"""
Role B — training-time augmentation.

Applied only to the TRAINING split (src/train.py never passes this to the
validation dataset) — validation must stay a fixed, honest read of the model,
not a moving target.

Two kinds of operator, per the role brief's warning that "target landmarks
[must be] transformed consistently for geometric augmentations":

  * Geometric (rotation, scale) act on the points AND the target landmarks
    TOGETHER, with the identical transform, so the ear/landmark relationship
    stays exact. Role C's template is never touched — the residual for an
    augmented example is simply ``target_aug - template``, computed fresh.
  * Non-geometric (jitter, point dropout/resampling) act on the input points
    only. The ground-truth landmark position does not depend on which points
    happened to be sampled off the surface, or on sensor noise.

Uses the plain ``numpy.random`` global functions (not a local Generator) on
purpose: src.train.set_seed() already calls ``np.random.seed(seed)``, so a
whole run stays reproducible, while different epochs still see different
augmented views of the same ear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    """Off by default. Enable only once the non-augmented baseline beats B1 —
    see docs/ROLE_B_GLOBAL_LANDMARK_MODEL.md: 'Augmentation: Only after
    baseline works.'
    """

    enabled: bool = False
    rotation_deg: float = 5.0   # max |angle| of the random rotation, degrees
    scale_min: float = 0.95
    scale_max: float = 1.05
    jitter_std: float = 0.0     # canonical units; 0 disables jitter
    dropout_frac: float = 0.0   # fraction of points resampled; 0 disables


def _random_rotation_matrix(max_deg: float) -> np.ndarray:
    """A single random rotation, angle in [-max_deg, max_deg] about a random axis."""
    if max_deg <= 0:
        return np.eye(3)
    axis = np.random.normal(size=3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    # Rodrigues' rotation formula.
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def augment_ear(
    points: np.ndarray,
    target: np.ndarray,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points: [N, 3] canonical ear points.
        target: [85, 3] canonical ground-truth landmarks, same frame as points.
    Returns:
        (points_aug [N, 3], target_aug [85, 3]). The caller must compute the
        residual as ``target_aug - template`` — never reuse a residual that
        was computed against the un-augmented ``target``.
    """
    if not cfg.enabled:
        return points, target

    points = np.array(points, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)

    rotation = _random_rotation_matrix(cfg.rotation_deg)
    scale = np.random.uniform(cfg.scale_min, cfg.scale_max)
    points = (points @ rotation.T) * scale
    target = (target @ rotation.T) * scale

    if cfg.jitter_std > 0:
        points = points + np.random.normal(scale=cfg.jitter_std, size=points.shape)

    if cfg.dropout_frac > 0:
        n = points.shape[0]
        n_drop = int(round(n * cfg.dropout_frac))
        if n_drop > 0:
            pool = points.copy()  # fill from the pre-dropout cloud, never from
            drop_idx = np.random.choice(n, size=n_drop, replace=False)          # an already-duplicated point
            fill_idx = np.random.choice(n, size=n_drop, replace=True)
            points[drop_idx] = pool[fill_idx]

    return points, target
