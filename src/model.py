"""
Role B — Global Landmark Model.

Predicts canonical landmark residuals from a canonical ear point cloud.

Input:  points   [B, N, in_dim]        (B ears, N points each, in_dim=3 for xyz)
Output: residual [B, n_landmarks, 3]

The residual is added to Role C's canonical template downstream:
    pred_canonical = canonical_template + residual

Architecture (PointNet-style, deliberately small):
    [B, N, in_dim]
      -> shared per-point MLP  in_dim -> 64 -> 128 -> 256
      -> global max pool over N
      -> head MLP  256 -> 512 -> 256 -> 255
      -> reshape   [B, 85, 3] 
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EarLandmarkNet(nn.Module):
    """Small PointNet-style regressor: point cloud in, landmark residuals out."""

    def __init__(
        self,
        in_dim: int = 3,
        n_landmarks: int = 85,
        point_dim1: int = 64,
        point_dim2: int = 128,
        point_dim3: int = 256,
        head_dim1: int = 512,
        head_dim2: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stored verbatim so a checkpoint can rebuild this exact architecture
        # without anyone having to guess. See train.save_checkpoint().
        self.config: dict[str, Any] = {
            "in_dim": in_dim,
            "n_landmarks": n_landmarks,
            "point_dim1": point_dim1,
            "point_dim2": point_dim2,
            "point_dim3": point_dim3,
            "head_dim1": head_dim1,
            "head_dim2": head_dim2,
            "dropout": dropout,
        }

        self.in_dim = in_dim
        self.n_landmarks = n_landmarks

        # ---- per-point layers -------------------------------------------
        # Conv1d with kernel_size=1 applies the SAME weighted sum to every
        # point independently. No point ever sees its neighbours here.
        self.conv1 = nn.Conv1d(in_dim, point_dim1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(point_dim1)

        self.conv2 = nn.Conv1d(point_dim1, point_dim2, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(point_dim2)

        self.conv3 = nn.Conv1d(point_dim2, point_dim3, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(point_dim3)

        # ---- head -------------------------------------------------------
        # After max pooling there is no per-point axis left, so plain Linear.
        self.fc1 = nn.Linear(point_dim3, head_dim1)
        self.fc_bn1 = nn.BatchNorm1d(head_dim1)

        self.fc2 = nn.Linear(head_dim1, head_dim2)
        self.fc_bn2 = nn.BatchNorm1d(head_dim2)

        # Final layer: no BatchNorm, no ReLU. Its outputs ARE the answer, and
        # residuals must be free to be negative.
        self.fc3 = nn.Linear(head_dim2, n_landmarks * 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim] canonical ear point cloud.
        Returns:
            [B, n_landmarks, 3] canonical landmark residuals.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B, N, F], got shape {tuple(x.shape)}")
        if x.shape[2] != self.in_dim:
            raise ValueError(
                f"expected last dim {self.in_dim}, got {x.shape[2]} "
                f"(is your tensor [B, F, N] instead of [B, N, F]?)"
            )

        batch_size = x.shape[0]

        # Conv1d wants [B, channels, N]; our data is [B, N, channels].
        x = x.transpose(1, 2)                            # [B, in_dim, N]

        x = self.relu(self.bn1(self.conv1(x)))           # [B, 64,  N]
        x = self.relu(self.bn2(self.conv2(x)))           # [B, 128, N]
        x = self.relu(self.bn3(self.conv3(x)))           # [B, 256, N]

        # Global max pool over the point axis. This is what makes the model
        # permutation invariant: max ignores ordering.
        x = torch.max(x, dim=2).values                   # [B, 256]

        x = self.relu(self.fc_bn1(self.fc1(x)))          # [B, 512]
        x = self.dropout(x)
        x = self.relu(self.fc_bn2(self.fc2(x)))          # [B, 256]
        x = self.dropout(x)

        x = self.fc3(x)                                  # [B, 255]

        return x.view(batch_size, self.n_landmarks, 3)   # [B, 85, 3]

    # ---------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EarLandmarkNet":
        """Rebuild a model from a stored config dict (used when loading ckpts)."""
        return cls(**config)

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)