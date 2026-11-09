# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Derived from the pinned author release; see THIRD_PARTY_LICENSES.

"""
GPU-accelerated data augmentation for training.

All augmentations operate on GPU tensors and are applied in the training
loop after the batch has been moved to the GPU.

data.augmentation keys:
- rotation_augmentation: rotation about the vertical (z) axis. One angle per
  batch, a multiple of angle_step_size, with probability prob_use_rotation;
  with per_sample: true every sample gets its own uniform angle in [0, 360)
  (probability prob_use_rotation per sample).
- reflection_augmentation: per sample with probability prob, the scene is
  mirrored in the vertical x = 0 plane (x -> -x). Mirrored rigid bodies under
  gravity, contact and isotropic friction follow the mirrored trajectories,
  so the mirrored sequence is again a valid one.
- rigid_noise: the input frames (all but the last frame of a sample) get a
  rigid random-walk perturbation per object (rotation about the object's
  centroid plus a translation), written to batch['obj_positions_input'];
  batch['obj_positions'] stays the ground truth the losses compare against.
  pos_std (m) and rot_std_deg are the standard deviations at the last input
  frame for step size 1, scaled by step_size ** step_size_exponent.
"""

from __future__ import annotations

import torch
import math
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) of rotation vectors (..., 3) (Rodrigues' formula)."""
    theta = rotvec.norm(dim=-1, keepdim=True)  # (..., 1)
    small = theta < 1e-8
    axis = rotvec / torch.where(small, torch.ones_like(theta), theta)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(
        *rotvec.shape[:-1], 3, 3
    )
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(K.shape)
    R = eye + s * K + (1.0 - c) * (K @ K)
    return torch.where(small.unsqueeze(-1), eye, R)


class GPUAugmentation:
    """
    GPU-accelerated augmentation pipeline.

    Features:
    - Rotation augmentation (Z-axis only, no ground mesh), per batch or per sample
    - Reflection augmentation (x -> -x, per sample)
    - Rigid random-walk noise on the input frames
    - All operations on GPU tensors
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize GPU augmentation.

        Args:
            config: Augmentation configuration
            device: Target GPU device
        """
        self.device = device
        self.config = config
        aug_cfg = config.get("data", {}).get("augmentation", {})

        # Rotation augmentation settings
        rotation_cfg = aug_cfg.get("rotation_augmentation", {})
        self.rotation_enabled = rotation_cfg.get("enable", False)
        self.rotation_angle_step = rotation_cfg.get("angle_step_size", 30)
        self.rotation_prob = rotation_cfg.get("prob_use_rotation", 0.3)
        self.rotation_per_sample = bool(rotation_cfg.get("per_sample", False))

        # Pre-compute rotation angles (only if rotation enabled and valid step size)
        if self.rotation_enabled and self.rotation_angle_step > 0:
            self.rotation_angles = torch.arange(
                self.rotation_angle_step,
                360,
                self.rotation_angle_step,
                dtype=torch.float32,
                device=device,
            )
        else:
            # Rotation disabled or invalid step size - create empty tensor
            self.rotation_angles = torch.tensor([], dtype=torch.float32, device=device)

        reflection_cfg = aug_cfg.get("reflection_augmentation", {}) or {}
        self.reflection_enabled = bool(reflection_cfg.get("enable", False))
        self.reflection_prob = float(reflection_cfg.get("prob", 0.5))

        noise_cfg = aug_cfg.get("rigid_noise", {}) or {}
        self.noise_enabled = bool(noise_cfg.get("enable", False))
        self.noise_pos_std = float(noise_cfg.get("pos_std", 0.0))
        self.noise_rot_std = math.radians(float(noise_cfg.get("rot_std_deg", 0.0)))
        self.noise_step_exponent = float(noise_cfg.get("step_size_exponent", 1.0))
        if self.noise_enabled and config.get("data", {}).get(
            "center_scene_by_t0", False
        ):
            # centring shifts the network inputs in place; with separate input frames the targets would not follow
            raise ValueError(
                "data.augmentation.rigid_noise cannot be combined with data.center_scene_by_t0"
            )

        # Counters for periodic logging.
        self.rotation_total_calls = 0
        self.rotation_applied_count = 0

        logger.info(f"\n{'='*80}")
        logger.info(f"[GPU Augmentation] Initialization")
        logger.info(f"{'='*80}")
        logger.info(f"Rotation Augmentation:")
        logger.info(f"  - Enabled: {self.rotation_enabled}")
        if self.rotation_enabled:
            if self.rotation_per_sample:
                logger.info(f"  - Per sample, uniform angle in [0, 360)")
            else:
                logger.info(f"  - Angle step size: {self.rotation_angle_step}°")
                logger.info(
                    f"  - Rotation angles: {self.rotation_angles.cpu().tolist()}"
                )
            logger.info(f"  - Probability: {self.rotation_prob:.2%}")
        logger.info(
            f"Reflection Augmentation (x -> -x): enabled={self.reflection_enabled}"
            + (
                f", probability {self.reflection_prob:.2%} per sample"
                if self.reflection_enabled
                else ""
            )
        )
        logger.info(
            f"Rigid input noise: enabled={self.noise_enabled}"
            + (
                f", pos_std={self.noise_pos_std} m, rot_std={math.degrees(self.noise_rot_std):.3f} deg at the last "
                f"input frame (step size 1), x step_size^{self.noise_step_exponent:g}"
                if self.noise_enabled
                else ""
            )
        )
        logger.info(f"{'='*80}\n")

    @property
    def enabled(self) -> bool:
        return self.rotation_enabled or self.reflection_enabled or self.noise_enabled

    def apply_rotation_augmentation(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Apply rotation augmentation around Z-axis (GPU version, no ground mesh).

        Args:
            batch: Batch dict with GPU tensors
            split: 'train', 'val', or 'test' (only train gets augmented)

        Returns:
            Augmented batch dict
        """
        # Only apply to training split
        if split != "train":
            batch["rotation_angle_deg"] = 0.0
            return batch

        self.rotation_total_calls += 1

        if not self.rotation_enabled:
            batch["rotation_angle_deg"] = 0.0
            return batch

        if self.rotation_per_sample:
            return self._rotate_per_sample(batch)

        if len(self.rotation_angles) == 0:
            raise ValueError(
                f"Rotation enabled but no valid angles available. "
                f"angle_step_size={self.rotation_angle_step} must be > 0"
            )

        # Decide whether to rotate (probabilistic)
        rand_val = torch.rand(1, device=self.device).item()
        if rand_val > self.rotation_prob:
            batch["rotation_angle_deg"] = 0.0
            return batch

        # Randomly select rotation angle
        angle_idx = torch.randint(
            0, len(self.rotation_angles), (1,), device=self.device
        )
        angle_deg = self.rotation_angles[angle_idx].item()

        if angle_deg == 0:
            batch["rotation_angle_deg"] = 0.0
            return batch

        self.rotation_applied_count += 1
        if self.rotation_applied_count <= 5 or self.rotation_total_calls % 100 == 0:
            apply_rate = self.rotation_applied_count / self.rotation_total_calls
            logger.info(
                f"[GPU Rotation Aug] Applying {angle_deg:.0f}° | "
                f"Batch {self.rotation_total_calls} | "
                f"Applied: {self.rotation_applied_count}/{self.rotation_total_calls} ({apply_rate:.2%})"
            )

        # Create rotation matrix (Z-axis)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        R = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        # Rotate obj_positions (B, T, Nobj, Nv, 3); all objects rotate since
        # there is no ground mesh.
        obj_pos = batch["obj_positions"]
        obj_pos_flat = obj_pos.reshape(-1, 3)
        obj_pos_rotated = torch.matmul(obj_pos_flat, R.T)
        batch["obj_positions"] = obj_pos_rotated.view(obj_pos.shape)

        # Rotate ref_obj_positions: (B, Nobj, Nv, 3)
        ref_pos = batch["ref_obj_positions"]
        ref_pos_flat = ref_pos.reshape(-1, 3)
        ref_pos_rotated = torch.matmul(ref_pos_flat, R.T)
        batch["ref_obj_positions"] = ref_pos_rotated.view(ref_pos.shape)

        # Record the applied angle so ground-truth overlays can be rotated to match.
        batch["rotation_angle_deg"] = angle_deg

        return batch

    def _per_sample_linear(
        self, batch: Dict[str, torch.Tensor], M: torch.Tensor
    ) -> None:
        """Apply the per-sample linear maps M (B, 3, 3) to obj_positions (B,T,Nobj,Nv,3) and ref_obj_positions."""
        # einsum may return non-contiguous strides; the model reshapes these tensors with .view()
        batch["obj_positions"] = torch.einsum(
            "bij,btonj->btoni", M, batch["obj_positions"]
        ).contiguous()
        ref = batch["ref_obj_positions"]
        if ref.dim() == 4:
            batch["ref_obj_positions"] = torch.einsum(
                "bij,bonj->boni", M, ref
            ).contiguous()
        else:
            raise ValueError(
                f"per-sample augmentation needs per-sample ref_obj_positions (B,Nobj,Nv,3); got {tuple(ref.shape)}"
            )

    def _rotate_per_sample(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        B = batch["obj_positions"].shape[0]
        angle = torch.rand(B, device=self.device) * (2 * math.pi)
        angle = torch.where(
            torch.rand(B, device=self.device) < self.rotation_prob,
            angle,
            torch.zeros_like(angle),
        )
        c, s = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        R = torch.stack([c, -s, zero, s, c, zero, zero, zero, one], dim=-1).view(
            B, 3, 3
        )
        self._per_sample_linear(batch, R)
        batch["rotation_angle_deg"] = (
            0.0  # no single angle: ground-truth overlays cannot be rotated to match
        )
        batch["rotation_angles_deg_per_sample"] = torch.rad2deg(angle)
        return batch

    def apply_reflection(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        if split != "train" or not self.reflection_enabled:
            return batch
        B = batch["obj_positions"].shape[0]
        flip = torch.rand(B, device=self.device) < self.reflection_prob
        sx = torch.where(
            flip, -torch.ones(B, device=self.device), torch.ones(B, device=self.device)
        )
        M = torch.diag_embed(
            torch.stack([sx, torch.ones_like(sx), torch.ones_like(sx)], dim=-1)
        )
        self._per_sample_linear(batch, M)
        batch["reflected"] = flip
        return batch

    def apply_rigid_noise(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        if split != "train" or not self.noise_enabled:
            return batch
        pos = batch["obj_positions"]  # (B,T,Nobj,Nv,3)
        B, T, Nobj, Nv, _ = pos.shape
        Tin = T - 1  # the network inputs are frames 0..T-2; frame T-1 is only a target
        if Tin < 1:
            return batch
        scale = torch.ones(B, device=pos.device)
        if "skip_step_size" in batch:
            scale = (
                batch["skip_step_size"].to(pos.device).float()
                ** self.noise_step_exponent
            )
        scale = scale.view(B, 1, 1, 1)
        # random walk over the input frames: the last input frame has the configured standard deviation
        rotvec = (
            torch.randn(B, Tin, Nobj, 3, device=pos.device)
            * (self.noise_rot_std / math.sqrt(Tin))
        ).cumsum(1) * scale
        shift = (
            torch.randn(B, Tin, Nobj, 3, device=pos.device)
            * (self.noise_pos_std / math.sqrt(Tin))
        ).cumsum(1) * scale
        R = rotvec_to_matrix(rotvec)  # (B,Tin,Nobj,3,3)
        x = pos[:, :Tin]
        vmask = batch["valid_vtx_mask"][:, :Tin].to(torch.bool)  # (B,Tin,Nobj,Nv)
        w = vmask.unsqueeze(-1).to(x.dtype)
        centroid = (x * w).sum(3, keepdim=True) / w.sum(3, keepdim=True).clamp_min(
            1.0
        )  # (B,Tin,Nobj,1,3)
        xn = (
            torch.einsum("btoij,btonj->btoni", R, x - centroid)
            + centroid
            + shift.unsqueeze(3)
        )
        xn = torch.where(vmask.unsqueeze(-1), xn, x)  # padding stays as it is
        batch["obj_positions_input"] = torch.cat([xn, pos[:, Tin:]], dim=1).contiguous()
        return batch

    def __call__(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all enabled augmentations.

        Args:
            batch: Batch dict with GPU tensors
            split: 'train', 'val', or 'test'

        Returns:
            Augmented batch dict
        """
        if self.rotation_enabled:
            batch = self.apply_rotation_augmentation(batch, split)
        else:
            # Rotation disabled: record a zero angle for consistency.
            batch["rotation_angle_deg"] = 0.0
        batch = self.apply_reflection(batch, split)
        # last: the noise perturbs the (rotated, mirrored) input frames
        batch = self.apply_rigid_noise(batch, split)

        return batch
