# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Transform layer for registering previous frame to current frame using Kabsch algorithm.
Handles batched rigid body registration with known vertex correspondences.

Supports two modes:
- Standard Kabsch: Non-differentiable SVD (requires STE for gradient flow)
- RoMa Kabsch: Fully differentiable (gradients flow through transformation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import roma

    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    roma = None


class TransformLayer(nn.Module):
    """
    Registers previous frame positions to current frame positions using optimal rigid transformation.
    Uses Kabsch algorithm for finding the best rotation and translation.

    Input shapes:
        batch_last_position: (B, N, 3) - positions from previous frame
        batch_current_position: (B, N, 3) - positions from current frame (target)

    Output shape:
        (B, N, 3) - transformed previous positions aligned to current frame
    """

    def __init__(self, use_scale=False, regularize_rotation=False, if_roma=False):
        """
        Args:
            use_scale: If True, use Umeyama algorithm (similarity transform with scale).
                      If False, use Kabsch algorithm (rigid transform only).
            regularize_rotation: If True, add small regularization to avoid numerical issues.
            if_roma: If True, use RoMa's differentiable rigid_points_registration.
                    If False, use standard SVD-based Kabsch (non-differentiable).
        """
        super().__init__()
        self.use_scale = use_scale
        self.regularize_rotation = regularize_rotation
        self.if_roma = if_roma

        if self.if_roma:
            if not ROMA_AVAILABLE:
                raise ImportError(
                    "RoMa library is not installed. Install with: pip install roma"
                )
            if self.use_scale:
                raise ValueError(
                    "RoMa mode does not support scale (Umeyama). Set use_scale=False."
                )

    def forward(self, batch_last_position, batch_current_position):
        """
        Register batch_last_position to batch_current_position.

        Args:
            batch_last_position: (B, N, 3) - source positions to be transformed
            batch_current_position: (B, N, 3) - target positions to align to

        Returns:
            transformed_positions: (B, N, 3) - batch_last_position after rigid transformation
        """
        B, N, D = batch_last_position.shape
        assert D == 3, f"Expected 3D points, got {D}D"
        assert batch_last_position.shape == batch_current_position.shape

        # RoMa mode: fully differentiable Kabsch
        if self.if_roma:
            # roma.rigid_points_registration: X -> Y
            # Returns R (B,3,3) and t_vec (B,3) such that: Y ≈ (R @ X.T).T + t
            R, t_vec = roma.rigid_points_registration(
                batch_last_position, batch_current_position  # Source (X)  # Target (Y)
            )
            # Apply transformation: Y = X @ R.T + t
            # R: (B,3,3), batch_last_position: (B,N,3)
            # X @ R.T: (B,N,3) @ (B,3,3).T = (B,N,3)
            transformed = torch.bmm(batch_last_position, R.transpose(1, 2))  # (B,N,3)
            transformed = transformed + t_vec.unsqueeze(1)  # (B,N,3) + (B,1,3)
            return transformed

        # Compute centroids
        centroid_last = batch_last_position.mean(dim=1, keepdim=True)  # (B, 1, 3)
        centroid_current = batch_current_position.mean(dim=1, keepdim=True)  # (B, 1, 3)

        # Center the point clouds
        last_centered = batch_last_position - centroid_last  # (B, N, 3)
        current_centered = batch_current_position - centroid_current  # (B, N, 3)

        # Compute covariance matrix H = X^T @ Y
        # where X is source (last) and Y is target (current)
        H = torch.bmm(last_centered.transpose(1, 2), current_centered)  # (B, 3, 3)

        # SVD decomposition with autocast disabled (SVD requires float32)
        with torch.amp.autocast("cuda", enabled=False):
            H_float32 = H.float()  # Cast to float32 for SVD
            U, S, Vt = torch.linalg.svd(
                H_float32
            )  # U: (B, 3, 3), S: (B, 3), Vt: (B, 3, 3)
            # Cast back to original dtype
            U, S, Vt = U.to(H.dtype), S.to(H.dtype), Vt.to(H.dtype)

        # Compute rotation matrix R = V @ U^T
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # (B, 3, 3)

        # Handle reflection case (ensure proper rotation with det(R) = 1)
        # Compute determinant in float32 to avoid half precision issues
        with torch.amp.autocast("cuda", enabled=False):
            R_float32 = R.float()
            det_R = torch.det(R_float32)  # (B,)
            det_R = det_R.to(R.dtype)

        # Create a diagonal matrix for flipping if needed
        flip_matrix = (
            torch.eye(3, device=R.device).unsqueeze(0).repeat(B, 1, 1)
        )  # (B, 3, 3)
        need_flip = det_R < 0  # (B,)

        if need_flip.any():
            # Flip the sign of the last singular vector
            flip_matrix[need_flip, 2, 2] = -1
            Vt_corrected = torch.bmm(flip_matrix, Vt)
            R = torch.bmm(Vt_corrected.transpose(1, 2), U.transpose(1, 2))

        # Apply regularization if requested (helps with numerical stability)
        if self.regularize_rotation:
            # Re-orthogonalize using another SVD
            with torch.amp.autocast("cuda", enabled=False):
                R_float32 = R.float()
                U_reg, S_reg, Vt_reg = torch.linalg.svd(R_float32)
                R = torch.bmm(U_reg, Vt_reg).to(R.dtype)

        # Compute scale if using Umeyama algorithm
        if self.use_scale:
            # Compute optimal scale
            # s = trace(S) / trace(X^T @ X)
            numerator = S.sum(dim=1)  # (B,)
            denominator = (last_centered**2).sum(dim=[1, 2])  # (B,)
            scale = (
                (numerator / (denominator + 1e-8)).unsqueeze(1).unsqueeze(2)
            )  # (B, 1, 1)
        else:
            scale = torch.ones(B, 1, 1, device=R.device)

        # Compute translation
        # t = centroid_current - scale * R @ centroid_last
        rotated_centroid = torch.bmm(R, centroid_last.transpose(1, 2)).transpose(
            1, 2
        )  # (B, 1, 3)
        translation = centroid_current - scale * rotated_centroid  # (B, 1, 3)

        # Apply transformation: y = scale * R @ x + t
        transformed = (
            scale * torch.bmm(batch_last_position, R.transpose(1, 2)) + translation
        )

        return transformed


class DifferentiableTransformLayer(TransformLayer):
    """
    Learnable transform layer that predicts rotation (6D) and translation.

    Modes:
      - absolute: predict the full R/t mapping last -> current
      - residual: reserved for composing a residual with rigid alignment (not implemented)
    """

    def __init__(
        self,
        use_scale=False,
        hidden_dim=256,
        use_6d_rotation=True,
        learnable_mode="absolute",
        return_params=True,
    ):
        super().__init__(use_scale=use_scale)
        self.use_6d_rotation = use_6d_rotation
        self.learnable_mode = learnable_mode  # 'absolute' | 'residual'
        self.return_params = return_params

        # Shallow PointNet over all vertices for each body (last/current)
        # Encode centered coordinates with 1x1 convs + BN + ReLU, global max-pool
        # last_feat, curr_feat each -> (B, 256)
        self.pn_conv1 = nn.Conv1d(3, 64, 1)
        self.pn_bn1 = nn.BatchNorm1d(64)
        self.pn_conv2 = nn.Conv1d(64, 128, 1)
        self.pn_bn2 = nn.BatchNorm1d(128)
        self.pn_conv3 = nn.Conv1d(128, 256, 1)
        self.pn_bn3 = nn.BatchNorm1d(256)

        # Head MLP maps fused global features (+ centroids for translation prior) to rot6d + t
        # Input dim: 256 (last) + 256 (current) + 6 (centroid_last/current)
        head_in_dim = 256 + 256 + 6
        out_dim = 9  # 6D rotation + 3D translation
        self.head_mlp = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def rotation_6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        a1 = rot6d[:, 0:3]
        a2 = rot6d[:, 3:6]
        b1 = F.normalize(a1, dim=1)
        a2_proj = a2 - (b1 * (a2 * b1).sum(dim=1, keepdim=True))
        b2 = F.normalize(a2_proj, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=2)  # (B,3,3)

    def forward(self, batch_last_position, batch_current_position):
        B, N, D = batch_last_position.shape
        assert D == 3, f"Expected 3D points, got {D}D"

        # Compute centroids for stability and translation prior
        centroid_last = batch_last_position.mean(dim=1)  # (B,3)
        centroid_current = batch_current_position.mean(dim=1)  # (B,3)

        # Center the point clouds (invariant to global translation)
        last_centered = batch_last_position - centroid_last.unsqueeze(1)  # (B,N,3)
        current_centered = batch_current_position - centroid_current.unsqueeze(
            1
        )  # (B,N,3)

        # PointNet feature extraction for both clouds
        def encode_points(x_centered: torch.Tensor) -> torch.Tensor:
            # x_centered: (B,N,3) -> (B,3,N)
            x = x_centered.transpose(1, 2)
            x = F.relu(self.pn_bn1(self.pn_conv1(x)))
            x = F.relu(self.pn_bn2(self.pn_conv2(x)))
            x = F.relu(self.pn_bn3(self.pn_conv3(x)))
            # Global max pooling over points -> (B,256)
            x = torch.amax(x, dim=2)
            return x

        last_feat = encode_points(last_centered)
        curr_feat = encode_points(current_centered)

        # Fuse global features and centroids
        fused = torch.cat(
            [last_feat, curr_feat, centroid_last, centroid_current], dim=1
        )  # (B, 518)

        # Predict rotation (6D) and translation (3D)
        pred = self.head_mlp(fused)  # (B,9)
        rot6d = pred[:, :6]
        t = pred[:, 6:9].unsqueeze(1)  # (B,1,3)
        R_pred = self.rotation_6d_to_matrix(rot6d)  # (B,3,3)

        if self.learnable_mode == "absolute":
            corrected = torch.bmm(batch_last_position, R_pred.transpose(1, 2)) + t

        return corrected, {"rot6d": rot6d, "trans": t.squeeze(1)}


class RobustTransformLayer(TransformLayer):
    """
    Robust variant that downweights outliers via iteratively reweighted
    Kabsch alignment (weights decay exponentially with residual distance).
    """

    def __init__(self, use_scale=False, num_iterations=3, outlier_threshold=0.1):
        super().__init__(use_scale=use_scale)
        self.num_iterations = num_iterations
        self.outlier_threshold = outlier_threshold

    def forward(self, batch_last_position, batch_current_position):
        B, N, _ = batch_last_position.shape

        # Initialize weights (all points equally weighted)
        weights = torch.ones(B, N, 1, device=batch_last_position.device)

        transformed = batch_last_position.clone()

        for iteration in range(self.num_iterations):
            # Weighted centroids
            weights_norm = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            centroid_last = (transformed * weights_norm).sum(dim=1, keepdim=True)
            centroid_current = (batch_current_position * weights_norm).sum(
                dim=1, keepdim=True
            )

            # Center with weighted centroids
            last_centered = transformed - centroid_last
            current_centered = batch_current_position - centroid_current

            # Weighted covariance
            last_weighted = last_centered * weights_norm.sqrt()
            current_weighted = current_centered * weights_norm.sqrt()
            H = torch.bmm(last_weighted.transpose(1, 2), current_weighted)

            # SVD and compute transformation with autocast disabled
            with torch.amp.autocast("cuda", enabled=False):
                H_float32 = H.float()
                U, S, Vt = torch.linalg.svd(H_float32)
                U, S, Vt = U.to(H.dtype), S.to(H.dtype), Vt.to(H.dtype)

            R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

            # Fix reflection
            # Compute determinant in float32 to avoid half precision issues
            with torch.amp.autocast("cuda", enabled=False):
                R_float32 = R.float()
                det_R = torch.det(R_float32)
                det_R = det_R.to(R.dtype)
            flip_matrix = torch.eye(3, device=R.device).unsqueeze(0).repeat(B, 1, 1)
            need_flip = det_R < 0
            if need_flip.any():
                flip_matrix[need_flip, 2, 2] = -1
                Vt_corrected = torch.bmm(flip_matrix, Vt)
                R = torch.bmm(Vt_corrected.transpose(1, 2), U.transpose(1, 2))

            # Translation
            translation = centroid_current - torch.bmm(
                R, centroid_last.transpose(1, 2)
            ).transpose(1, 2)

            # Apply transformation
            transformed = (
                torch.bmm(batch_last_position, R.transpose(1, 2)) + translation
            )

            # Update weights based on distances (downweight outliers)
            if iteration < self.num_iterations - 1:
                distances = torch.norm(
                    transformed - batch_current_position, dim=2, keepdim=True
                )
                weights = torch.exp(-distances / self.outlier_threshold)

        return transformed


class RoMaTransformLayer(nn.Module):
    """
    Differentiable Kabsch registration using RoMa library.

    Key difference from TransformLayer:
    - Fully differentiable (gradients flow through R and t)
    - No need for STE (Straight-Through Estimator)
    - Uses roma.rigid_points_registration() which handles SVD gradient

    Note: Requires 'roma' library to be installed.
    """

    def __init__(self):
        super().__init__()
        if not ROMA_AVAILABLE:
            raise RuntimeError(
                "RoMa library not available. Install with: pip install roma\n"
                "RoMa is required for differentiable Kabsch registration."
            )

    def forward(self, batch_last_position, batch_current_position):
        """
        Register batch_last_position to batch_current_position using RoMa.

        Args:
            batch_last_position: (B, N, 3) - source positions to be transformed
            batch_current_position: (B, N, 3) - target positions to align to

        Returns:
            transformed_positions: (B, N, 3) - transformed positions (differentiable)
        """
        B, N, D = batch_last_position.shape
        assert D == 3, f"Expected 3D points, got {D}D"
        assert batch_last_position.shape == batch_current_position.shape

        # RoMa's rigid_points_registration returns (R, t_vec)
        # R: (B, 3, 3) rotation matrix
        # t_vec: (B, 3) translation vector
        # Formula: y = R @ x + t
        R, t_vec = roma.rigid_points_registration(
            batch_last_position, batch_current_position  # source  # target
        )

        # Apply transformation: y = x @ R^T + t
        # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        transformed = torch.bmm(
            batch_last_position, R.transpose(1, 2)
        ) + t_vec.unsqueeze(1)

        return transformed


# Convenience function for creating the layer
def create_transform_layer(method="kabsch", if_roma=False, **kwargs):
    """
    Factory function to create transform layer.

    Args:
        method: Only 'kabsch' is supported
        if_roma: If True, use RoMa's differentiable Kabsch; otherwise use standard SVD Kabsch
        **kwargs: Additional arguments (e.g., regularize_rotation)

    Returns:
        TransformLayer instance with Kabsch algorithm
    """
    if method != "kabsch":
        raise ValueError(f"Only method='kabsch' is supported, got '{method}'")

    return TransformLayer(use_scale=False, if_roma=if_roma, **kwargs)
