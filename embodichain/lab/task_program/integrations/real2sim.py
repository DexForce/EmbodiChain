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

"""Qpos-authoritative expert provider with Skill-owned plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
import math
import hashlib
from pathlib import Path

import torch

from embodichain.data_pipeline.real2sim.bundle import load_real2sim_bundle
from embodichain.data_pipeline.real2sim.normalization import NormalizedReal2SimUnit
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    PlanningContext,
    ResolvedActionRequest,
    SceneProvider,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
)
from embodichain.lab.sim.motion.expansion import (
    Real2SimSourceAdapter,
    SourceContext,
    SceneCase,
)
from embodichain.lab.task_program.semantics.profiles import RobotSkillProfile

__all__ = ["Real2SimExpertSourceProviderFactory", "Real2SimExpertSourceProvider"]


@dataclass(frozen=True)
class Real2SimExpertSourceProviderFactory:
    """Allowlisted source declaration pinned by a complete bundle digest.

    Args:
        bundle_path: Non-executable real2sim bundle declaration.
        bundle_sha256: Required content digest of that declaration.
        target_profile_id: Exact reviewed embodiment profile bound to this source.
        target_asset_sha256: Pinned resolved URDF/USD definition of the target robot.
    """

    provider_id: ClassVar[str] = "real2sim_qpos"
    revision: ClassVar[str] = "1"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"replay", "phase_annotation", "effect_validation"}
    )
    bundle_path: str
    bundle_sha256: str
    target_profile_id: str
    target_asset_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.target_profile_id
            or self.target_profile_id.strip() != self.target_profile_id
        ):
            raise ValueError("An exact target profile ID is required")
        if len(self.target_asset_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.target_asset_sha256
        ):
            raise ValueError("A lowercase target asset SHA-256 is required")
        # Admission is provider-free and happens before the environment is built.
        unit = load_real2sim_bundle(
            self.bundle_path, expected_sha256=self.bundle_sha256
        )
        _segment_lengths(unit)
        _template(unit)

    def create(
        self,
        *,
        call_id: str,
        scene_provider: SceneProvider,
        robot_profile: RobotSkillProfile,
        control_dt: float,
        engine: AtomicActionEngine,
    ) -> Real2SimExpertSourceProvider:
        """Create an owned B=1 Pick source on the host clock.

        Args:
            call_id: Selected semantic kind; v1 supports pick.
            scene_provider: Existing scene ownership boundary.
            robot_profile: Existing embodiment profile.
            control_dt: Fixed environment command interval.
            engine: Existing Skill construction and validation owner.

        Returns:
            Fresh fixed-qpos provider with independently loaded source evidence.
        """
        if call_id != "pick":
            raise ValueError("real2sim_qpos v1 supports one Pick call")
        if robot_profile.profile_id != self.target_profile_id:
            raise ValueError(
                "Source target profile differs from the selected embodiment"
            )
        asset_path = Path(engine.robot.cfg.fpath)
        if (
            hashlib.sha256(asset_path.read_bytes()).hexdigest()
            != self.target_asset_sha256
        ):
            raise ValueError(
                "Source target asset digest differs from the live robot model"
            )
        unit = load_real2sim_bundle(
            self.bundle_path, expected_sha256=self.bundle_sha256
        )
        if not math.isclose(
            control_dt, float(unit.report["control_dt"]), abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("Source control clock differs from the environment")
        return Real2SimExpertSourceProvider(
            unit, engine, robot_profile.profile_id, self.target_asset_sha256
        )


def _segment_lengths(unit) -> dict[str, int]:
    # These are the full semantic phases, never the inset augmentation windows.
    phases = unit.phases
    if (
        tuple(p.name for p in phases) != ("approach", "close", "lift")
        or phases[0].start != 0
        or phases[-1].stop != len(unit.positions)
    ):
        raise ValueError(
            "A recorded Pick source must have complete approach/close/lift phases"
        )
    if phases[1].kind != "contact" or phases[2].kind != "hold":
        raise ValueError(
            "Recorded Pick close/lift phases must be protected contact/hold"
        )
    # Interpolation-only gaps stay locked by the source adapter. Attribute them
    # to the preceding semantic segment, preserving every hard event sample.
    stops = [phases[1].start, phases[2].start, len(unit.positions)]
    return {p.name: stop - p.start for p, stop in zip(phases, stops)}


def _template(unit):
    return Real2SimSourceAdapter().export_template(
        unit,
        context=SourceContext(
            unit.source.source_id,
            unit.revision,
            unit.source.unit_id,
            SceneCase(
                "source_admission",
                "declared_initial",
                "source_scene",
                "pick",
                "source_profile",
            ),
            float(unit.report["control_dt"]),
        ),
    )


@dataclass
class Real2SimExpertSourceProvider:
    """One owned recording; no execution, reset or persistence lifecycle.

    Args:
        unit: Verified normalized source unit.
        engine: Existing engine that owns Skill plan construction.
        profile_id: Selected embodiment profile identity for provenance.
        asset_sha256: Verified live model definition digest.
    """

    unit: NormalizedReal2SimUnit
    engine: AtomicActionEngine
    profile_id: str
    asset_sha256: str

    def plan(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        """Build a full Pick plan from admitted full-joint source values.

        Args:
            request: Exact resolved PickUp invocation.
            context: Live initial robot and semantic scene state.

        Returns:
            Complete Skill-assembled plan with source diagnostics.
        """
        unit = self.unit
        _template(unit)
        if context.batch_size != 1:
            raise ValueError("Recorded qpos v1 requires B=1")
        if request.skill_id != "pick_up":
            raise ValueError("Recorded Pick source requires the PickUp Skill")
        if (
            request.revision != 0
            or request.recovery_policy.max_replans
            or request.recovery_policy.max_action_retries
        ):
            raise ValueError(
                "Fixed qpos replay cannot replan or retry without a fresh reset"
            )
        if tuple(self.engine.robot.joint_names) != unit.joint_names:
            raise ValueError(
                "Source joint order differs from the complete simulation robot"
            )
        if not math.isclose(
            context.require_control_dt(),
            float(unit.report["control_dt"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Source control clock differs from the planning context")
        q = torch.tensor(
            unit.positions.copy(),
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )[None]
        if not torch.allclose(q[:, 0], context.robot.qpos, atol=1e-4, rtol=0):
            raise ValueError(
                "Restore the exact recording initial state before fixed replay"
            )
        limits = self.engine.robot.get_qpos_limits(env_ids=context.env_ids)
        if not isinstance(limits, torch.Tensor) or limits.shape != (
            1,
            len(unit.joint_names),
            2,
        ):
            raise ValueError("Live robot must expose full-joint position limits")
        limits = limits.to(q)
        if not torch.isfinite(limits).all() or (limits[..., 0] > limits[..., 1]).any():
            raise ValueError("Live joint position limits must be finite and ordered")
        if (
            (q < limits[:, None, :, 0] - 1e-6) | (q > limits[:, None, :, 1] + 1e-6)
        ).any():
            raise ValueError("Source violates live robot joint position limits")
        velocity_limits = self.engine.robot.get_qvel_limits(env_ids=context.env_ids)
        if (
            not isinstance(velocity_limits, torch.Tensor)
            or velocity_limits.shape != (1, len(unit.joint_names))
            or not torch.isfinite(velocity_limits).all()
            or (velocity_limits <= 0).any()
        ):
            raise ValueError(
                "Live robot must expose positive finite joint velocity limits"
            )
        velocities = torch.tensor(
            unit.velocities.copy(), device=q.device, dtype=q.dtype
        )[None]
        if (velocities.abs() > velocity_limits.to(q)[:, None] + 1e-6).any():
            raise ValueError("Source violates live robot joint velocity limits")
        claimed = set()
        for endpoint in request.binding.endpoints:
            target = getattr(endpoint, "target", None)
            claimed.update(getattr(target, "joint_ids", ()))
        moving = (
            ((q[:, 1:] - q[:, :-1]).abs().amax(dim=(0, 1)) > 1e-8)
            .nonzero()
            .flatten()
            .tolist()
        )
        # Mimics are derived channels. The source adapter already locks their drivers.
        mimics = {unit.joint_names.index(m["joint"]) for m in unit.report["mimics"]}
        if set(moving) - claimed - mimics:
            raise ValueError(
                "Recording moves joints outside the call's authorized endpoints"
            )
        trajectory = TimedTrajectory(
            positions=q,
            velocities=velocities,
            accelerations=None,
            dt=torch.tensor(unit.dt.copy(), device=q.device, dtype=q.dtype)[None],
            env_ids=context.env_ids,
        )
        return self.engine.build_plan_from_recorded_trajectory(
            request,
            context,
            trajectory,
            segment_lengths=_segment_lengths(unit),
            diagnostics=PlannerDiagnostics(
                backend="real2sim_qpos",
                metadata={
                    "source_id": unit.source.source_id,
                    "source_revision": unit.source.source_revision,
                    "normalized_revision": unit.revision,
                    "unit_id": unit.source.unit_id,
                    "authority": "qpos",
                    "profile_id": self.profile_id,
                    "target_asset_sha256": self.asset_sha256,
                    "normalization": unit.report_dict(),
                    "interphase_segment_policy": "preceding_segment_locked",
                },
            ),
        )
