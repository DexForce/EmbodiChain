# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Database-driven joint seed selection for iterative IK multi-start.

Instead of filling the multi-start seed batch with uniform random draws,
:class:`QposSeedSelSampler` retrieves seeds from a precomputed forward-kinematics
database: the joint space is sampled once with a low-discrepancy sequence, the
end-effector pose of every sample is stored, and at query time the nearest
database entries to each target pose are returned as seeds. When a Jacobian
provider is available, candidates are re-ranked by the predicted joint-space
step ``||J^+ (target - seed_pose)||`` so that seeds requiring the smallest
correction are tried first.

The retrieval strategy follows the SELIK solver of the WRS framework
(Wan Weiwei, 2023, MIT licensed); this is an independent batched PyTorch
re-implementation adapted to the EmbodiChain seed-sampler contract.

The class is a drop-in extension of :class:`QposSeedSampler`: calls without a
target pose fall back to the parent's uniform random behaviour, and slot ``0``
of the returned batch preserves the caller-provided seed so warm-start chains
(for example sequential waypoint solving) keep their continuity.
"""

from __future__ import annotations

from typing import Callable

import torch

from embodichain.utils import logger
from embodichain.utils.math import axis_angle_from_quat, quat_from_matrix

from .qpos_seed_sampler import QposSeedSampler

__all__ = ["QposSeedSelSampler"]

FkFn = Callable[[torch.Tensor], torch.Tensor]
"""Batched forward kinematics: ``(N, dof)`` joints to ``(N, 4, 4)`` poses."""

JacobianFn = Callable[[torch.Tensor], torch.Tensor]
"""Batched geometric Jacobian: ``(N, dof)`` joints to ``(N, 6, dof)``."""


def _pose_error_twist(cur: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Compute the ``[dp, rotvec]`` twist between pose batches.

    Args:
        cur: Current poses with shape ``(N, 4, 4)``.
        tgt: Target poses with shape ``(N, 4, 4)``.

    Returns:
        Twist tensor with shape ``(N, 6)``.
    """
    dp = tgt[:, :3, 3] - cur[:, :3, 3]
    rel = tgt[:, :3, :3] @ cur[:, :3, :3].transpose(1, 2)
    # Quaternion route: numerically stable over the whole rotation range,
    # including relative rotations at and near 180 degrees where a direct
    # skew-symmetric axis extraction degenerates to zero.
    rotvec = axis_angle_from_quat(quat_from_matrix(rel.contiguous()))
    return torch.cat([dp, rotvec], dim=-1)


class QposSeedSelSampler(QposSeedSampler):
    """SELIK-style seed sampler backed by a forward-kinematics database.

    The database is built lazily on the first target-aware :meth:`sample` call
    using the joint limits supplied by the caller, and rebuilt automatically
    whenever those limits change. Building is a batched FK sweep and typically
    takes well under a second on GPU for the default database size.

    Args:
        num_samples: Number of seeds per target (including the caller seed).
        dof: Degrees of freedom.
        device: Target device.
        fk_fn: Batched forward kinematics of the solved chain, in the same
            frame and with the same TCP as the IK targets.
        jacobian_fn: Optional batched Jacobian provider. When given, retrieved
            candidates are re-ranked by predicted joint-space step length.
        db_size: Number of joint configurations stored in the database.
        rot_scale: Metres-per-radian weight applied to the rotation block of
            the pose metric used for nearest-neighbour retrieval.
        k_max: Number of nearest neighbours fetched before re-ranking.
        use_caller_seed: If ``True`` (default), slot ``0`` of every returned
            batch is the caller-provided seed; if ``False`` all slots come
            from the database.
        sobol_seed: Scramble seed of the low-discrepancy joint sampler.
    """

    _LIMITS_ATOL = 1e-6
    _FK_CHUNK = 10000
    _QUERY_CHUNK = 256

    def __init__(
        self,
        num_samples: int,
        dof: int,
        device: torch.device,
        *,
        fk_fn: FkFn,
        jacobian_fn: JacobianFn | None = None,
        db_size: int = 20000,
        rot_scale: float = 0.2,
        k_max: int = 200,
        use_caller_seed: bool = True,
        sobol_seed: int = 0,
    ) -> None:
        super().__init__(num_samples=num_samples, dof=dof, device=device)
        if db_size <= 0:
            raise ValueError(f"db_size must be positive, got {db_size}.")
        if rot_scale <= 0.0:
            raise ValueError(f"rot_scale must be positive, got {rot_scale}.")
        if k_max <= 0:
            raise ValueError(f"k_max must be positive, got {k_max}.")
        self._fk_fn = fk_fn
        self._jacobian_fn = jacobian_fn
        self._db_size = db_size
        self._rot_scale = rot_scale
        self._k_max = k_max
        self._use_caller_seed = use_caller_seed
        self._sobol_seed = sobol_seed

        self._db_qpos: torch.Tensor | None = None
        self._db_pose: torch.Tensor | None = None
        self._db_vec: torch.Tensor | None = None
        self._db_jinv: torch.Tensor | None = None
        self._db_limits: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ database

    @property
    def database_size(self) -> int:
        """Number of entries in the built database, or ``0`` before build."""
        return 0 if self._db_qpos is None else int(self._db_qpos.shape[0])

    def _limits_changed(
        self, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> bool:
        if self._db_limits is None:
            return True
        lo, hi = self._db_limits
        return not (
            torch.allclose(lo, lower_limits, atol=self._LIMITS_ATOL)
            and torch.allclose(hi, upper_limits, atol=self._LIMITS_ATOL)
        )

    def _pose_vec(self, pose: torch.Tensor) -> torch.Tensor:
        """Flatten poses into the retrieval metric space, shape ``(N, 12)``."""
        return torch.cat(
            [pose[:, :3, 3], self._rot_scale * pose[:, :3, :3].reshape(-1, 9)],
            dim=-1,
        )

    def _build_database(
        self, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> None:
        """Sample the joint space and store poses, metric vectors and J-pinv."""
        sobol = torch.quasirandom.SobolEngine(
            dimension=self.dof, scramble=True, seed=self._sobol_seed
        )
        unit = sobol.draw(self._db_size).to(
            device=lower_limits.device, dtype=lower_limits.dtype
        )
        qpos = lower_limits + unit * (upper_limits - lower_limits)

        poses, vecs, jinvs = [], [], []
        with torch.no_grad():
            for start in range(0, self._db_size, self._FK_CHUNK):
                chunk = qpos[start : start + self._FK_CHUNK]
                pose = self._fk_fn(chunk)
                poses.append(pose)
                vecs.append(self._pose_vec(pose))
                if self._jacobian_fn is not None:
                    jac = self._jacobian_fn(chunk)
                    jinvs.append(torch.linalg.pinv(jac, rcond=1e-4))

        self._db_qpos = qpos
        self._db_pose = torch.cat(poses)
        self._db_vec = torch.cat(vecs)
        self._db_jinv = torch.cat(jinvs) if jinvs else None
        self._db_limits = (lower_limits.clone(), upper_limits.clone())

    # ------------------------------------------------------------------ retrieval

    def _query(self, target_xpos: torch.Tensor, k: int) -> torch.Tensor:
        """Retrieve the top-``k`` seeds per target, shape ``(B, k, dof)``."""
        assert self._db_qpos is not None and self._db_vec is not None
        k = min(k, self._db_qpos.shape[0])
        k_max = min(self._k_max, self._db_qpos.shape[0])
        query_vec = self._pose_vec(target_xpos)

        index_chunks = []
        for start in range(0, query_vec.shape[0], self._QUERY_CHUNK):
            dist = torch.cdist(
                query_vec[start : start + self._QUERY_CHUNK], self._db_vec
            )
            index_chunks.append(dist.topk(k_max, largest=False).indices)
        indices = torch.cat(index_chunks)

        if self._db_jinv is None:
            return self._db_qpos[indices[:, :k]]

        batch = indices.shape[0]
        seed_pose = self._db_pose[indices].reshape(-1, 4, 4)
        target_rep = (
            target_xpos.unsqueeze(1).expand(batch, k_max, 4, 4).reshape(-1, 4, 4)
        )
        twist = _pose_error_twist(seed_pose, target_rep).view(batch, k_max, 6)
        step = torch.einsum("bkij,bkj->bki", self._db_jinv[indices], twist)
        order = step.pow(2).sum(dim=-1).argsort(dim=1)
        return self._db_qpos[torch.gather(indices, 1, order[:, :k])]

    def _pad_with_random(
        self,
        seeds: torch.Tensor,
        count: int,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        """Pad retrieved seeds up to ``count`` slots per target.

        Retrieval is capped by the database size and by ``k_max``, so it can
        return fewer candidates than requested. Shortfall slots are filled
        with uniform random draws within the limits — the parent sampler's
        behaviour — so the ``batch_size * num_samples`` output contract holds
        for every configuration.

        Args:
            seeds: Retrieved seeds with shape ``(B, k_got, dof)``.
            count: Required number of slots per target.
            lower_limits: ``(dof,)`` lower joint limits.
            upper_limits: ``(dof,)`` upper joint limits.

        Returns:
            torch.Tensor: Seeds with shape ``(B, count, dof)``.
        """
        shortfall = count - seeds.shape[1]
        if shortfall <= 0:
            return seeds[:, :count]
        random_fill = torch.rand(
            seeds.shape[0],
            shortfall,
            self.dof,
            device=seeds.device,
            dtype=seeds.dtype,
        )
        random_fill = lower_limits + random_fill * (upper_limits - lower_limits)
        return torch.cat([seeds, random_fill], dim=1)

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        qpos_seed: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
        batch_size: int,
        target_xpos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate joint seeds, retrieving from the database when possible.

        Args:
            qpos_seed: ``(batch_size, dof)`` or ``(dof,)`` caller seed.
            lower_limits: ``(dof,)`` lower joint limits.
            upper_limits: ``(dof,)`` upper joint limits.
            batch_size: Number of targets.
            target_xpos: Optional ``(batch_size, 4, 4)`` target poses in the
                same frame ``fk_fn`` produces. When omitted, behaviour is
                identical to :class:`QposSeedSampler`.

        Returns:
            torch.Tensor: ``(batch_size * num_samples, dof)`` joint seeds,
            target-major, slot ``0`` holding the caller seed unless
            ``use_caller_seed=False``. The shape contract holds for every
            configuration: when retrieval returns fewer candidates than
            requested (database smaller than the seed count, or
            ``num_samples - 1 > k_max``), the shortfall is filled with
            uniform random draws within the limits.
        """
        if target_xpos is None:
            return super().sample(qpos_seed, lower_limits, upper_limits, batch_size)

        if target_xpos.shape != (batch_size, 4, 4):
            logger.log_error(
                f"target_xpos must have shape ({batch_size}, 4, 4), "
                f"got {tuple(target_xpos.shape)}.",
                ValueError,
            )
        if qpos_seed.shape == (batch_size, self.dof):
            seed_head = qpos_seed[:, None, :]
        elif qpos_seed.shape == (self.dof,):
            seed_head = qpos_seed.unsqueeze(0).repeat(batch_size, 1)[:, None, :]
        else:
            logger.log_error(
                f"Invalid qpos_seed shape {qpos_seed.shape} for batch_size "
                f"{batch_size} and dof {self.dof}",
                ValueError,
            )

        if self._limits_changed(lower_limits, upper_limits):
            self._build_database(lower_limits, upper_limits)

        if self._use_caller_seed:
            n_retrieved = self.num_samples - 1
            if n_retrieved == 0:
                return seed_head.reshape(-1, self.dof)
            retrieved = self._pad_with_random(
                self._query(target_xpos, n_retrieved),
                n_retrieved,
                lower_limits,
                upper_limits,
            )
            joint_seeds = torch.cat([seed_head, retrieved], dim=1)
        else:
            joint_seeds = self._pad_with_random(
                self._query(target_xpos, self.num_samples),
                self.num_samples,
                lower_limits,
                upper_limits,
            )
        return joint_seeds.reshape(-1, self.dof)
