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

"""Deterministic reachable cases for the ``free-space-common`` track."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..config import SuiteCfg, TrackCfg, stable_hash
from ..models import BenchmarkCase
from ..registry import register_scenario_provider
from .base import ScenarioProvider

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["FreeSpaceScenario"]

_NOMINAL_QPOS = torch.tensor(
    [0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4],
    dtype=torch.float32,
)


def _clamp_with_margin(qpos: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Clamp qpos inside five-percent joint-limit margins."""
    lower, upper = limits[:, 0], limits[:, 1]
    margin = (upper - lower).clamp_min(1.0e-3) * 0.05
    return torch.maximum(torch.minimum(qpos, upper - margin), lower + margin)


def _start_qpos_for_bin(
    name: str,
    limits: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create one deterministic start posture for a named condition bin.

    ``near_singularity`` uses a fixed elbow-extended Franka posture. It is a
    reproducible low-manipulability seed for free-space-common v1, not a
    runtime singularity search.
    """
    lower, upper = limits[:, 0], limits[:, 1]
    midpoint = (lower + upper) * 0.5
    span = upper - lower
    nominal = _clamp_with_margin(_NOMINAL_QPOS.to(limits), limits)

    if name == "nominal":
        return nominal
    if name == "random_reachable":
        noise = torch.rand(limits.shape[0], generator=generator) - 0.5
        return _clamp_with_margin(midpoint + noise.to(limits) * span * 0.55, limits)
    if name == "near_limit":
        signs = torch.where(
            torch.arange(limits.shape[0]) % 2 == 0,
            torch.ones(limits.shape[0]),
            -torch.ones(limits.shape[0]),
        ).to(limits)
        return _clamp_with_margin(midpoint + signs * span * 0.42, limits)
    if name == "near_singularity":
        # Elbow nearly extended (q3≈0): a fixed Franka near-singularity seed.
        candidate = torch.tensor(
            [0.0, 0.0, 0.0, -0.15, 0.0, 0.20, 0.0], dtype=limits.dtype
        ).to(limits)
        return _clamp_with_margin(candidate, limits)
    raise ValueError(f"Unknown free-space start_state_bin {name!r}.")


def _joint_delta(path_shape: str, alpha: float, dofs: int) -> torch.Tensor:
    """Return a bounded reference joint displacement for one path sample."""
    delta = torch.zeros(dofs, dtype=torch.float32)
    if path_shape == "direct":
        delta[: min(dofs, 4)] = torch.tensor([0.22, -0.16, 0.12, 0.10])[:dofs]
        return delta * alpha
    if path_shape == "l_turn":
        first = torch.zeros_like(delta)
        second = torch.zeros_like(delta)
        first[: min(dofs, 3)] = torch.tensor([0.18, -0.12, 0.08])[:dofs]
        if dofs > 3:
            second[3 : min(dofs, 7)] = torch.tensor([0.12, -0.16, 0.18, -0.12])[
                : max(0, min(dofs, 7) - 3)
            ]
        if alpha <= 0.5:
            return first * (alpha * 2.0)
        return first + second * ((alpha - 0.5) * 2.0)
    if path_shape == "s_curve":
        if dofs > 0:
            delta[0] = 0.20 * alpha
        if dofs > 1:
            delta[1] = -0.16 * math.sin(math.pi * alpha)
        if dofs > 3:
            delta[3] = 0.12 * math.sin(2.0 * math.pi * alpha)
        return delta
    if path_shape == "orientation_only":
        if dofs > 4:
            delta[4] = 0.25 * alpha
        if dofs > 6:
            delta[6] = -0.30 * alpha
        return delta
    if path_shape == "combined":
        direct = _joint_delta("direct", alpha, dofs)
        orientation = _joint_delta("orientation_only", alpha, dofs)
        return direct + orientation
    raise ValueError(f"Unknown free-space path_shape {path_shape!r}.")


def _build_case(
    suite: SuiteCfg,
    robot: "Robot",
    control_part: str,
    *,
    seed: int,
    batch_size: int,
    num_waypoints: int,
    path_shape: str,
    start_state_bin: str,
    bin_index: int,
    track_id: str,
) -> BenchmarkCase:
    """Build one reachable env-batched case using FK reference targets.

    Start postures depend only on ``(seed, start_state_bin, env_index)`` so the
    same env row keeps a shared start across path shapes and waypoint counts.
    Path shape and waypoint count affect targets only.
    """
    limits = robot.get_qpos_limits(name=control_part)[0].detach().cpu()
    if limits.shape[0] != _NOMINAL_QPOS.shape[0]:
        raise ValueError(
            "free-space-common v1 expects a 7-DoF Franka arm, got "
            f"{limits.shape[0]} DoF."
        )

    starts: list[torch.Tensor] = []
    for env_index in range(batch_size):
        generator = torch.Generator(device="cpu")
        # Keep starts aligned across path_shape / num_waypoints comparisons.
        generator.manual_seed(seed * 100_003 + bin_index * 131 + env_index)
        starts.append(_start_qpos_for_bin(start_state_bin, limits, generator))

    start_qpos_cpu = torch.stack(starts)
    references: list[torch.Tensor] = []
    for waypoint_index in range(num_waypoints):
        alpha = float(waypoint_index + 1) / float(num_waypoints)
        delta = _joint_delta(path_shape, alpha, start_qpos_cpu.shape[-1])
        target = _clamp_with_margin(start_qpos_cpu + delta.unsqueeze(0), limits)
        references.append(target)
    reference_qpos_cpu = torch.stack(references, dim=1)

    start_qpos = start_qpos_cpu.to(robot.device)
    reference_qpos = reference_qpos_cpu.to(robot.device)
    waypoint_poses = []
    for waypoint_index in range(num_waypoints):
        waypoint_poses.append(
            robot.compute_fk(
                qpos=reference_qpos[:, waypoint_index],
                name=control_part,
                to_matrix=True,
            )
        )
    target_waypoints = torch.stack(waypoint_poses, dim=1)

    identity = {
        "suite_version": suite.suite_version,
        "track": track_id,
        "seed": seed,
        "batch_size": batch_size,
        "num_waypoints": num_waypoints,
        "path_shape": path_shape,
        "start_state_bin": start_state_bin,
    }
    case_id = f"free_space_{stable_hash(identity)[:16]}"
    return BenchmarkCase(
        suite_version=suite.suite_version,
        track=track_id,
        scenario_id="waypoint_path" if num_waypoints > 1 else "reach",
        case_id=case_id,
        seed=seed,
        batch_size=batch_size,
        num_waypoints=num_waypoints,
        path_shape=path_shape,
        start_state_bin=start_state_bin,
        start_qpos=start_qpos,
        target_waypoints=target_waypoints,
        reference_qpos=reference_qpos,
    )


class FreeSpaceScenario(ScenarioProvider):
    required_capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})

    def batch_sizes(
        self, suite: SuiteCfg, track: TrackCfg
    ) -> list[int]:  # noqa: ARG002
        return list(suite.free_space.batch_sizes)

    def generate_cases(
        self,
        suite: SuiteCfg,
        track: TrackCfg,
        robot: "Robot",
        control_part: str,
        batch_size: int,
    ) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for seed in suite.free_space.seeds:
            for num_waypoints in suite.free_space.waypoint_counts:
                for path_shape in suite.free_space.path_shapes:
                    for bin_index, start_state_bin in enumerate(
                        suite.free_space.start_state_bins
                    ):
                        cases.append(
                            _build_case(
                                suite,
                                robot,
                                control_part,
                                seed=seed,
                                batch_size=batch_size,
                                num_waypoints=num_waypoints,
                                path_shape=path_shape,
                                start_state_bin=start_state_bin,
                                bin_index=bin_index,
                                track_id=track.id,
                            )
                        )
        return cases


register_scenario_provider("free_space", FreeSpaceScenario)
