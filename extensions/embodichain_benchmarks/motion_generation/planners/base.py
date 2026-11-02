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

"""Backend-independent planner adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.planners.utils import PlanResult

from ..config import PlannerSpecCfg, stable_hash
from ..models import AlgorithmRole, BenchmarkCase, PlannerMetadata
from ..limits import resolve_dynamic_limit, resolve_robot_velocity_limits

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.motion.planners import PlanOptions

__all__ = ["PlannerAdapter", "PlannerContext"]

_MOTION_VALIDITY_CAPABILITIES = {
    "ordered_cartesian_waypoints": "eef_waypoint",
    "ordered_joint_waypoints": "joint_waypoint",
    "ordered_constraint_waypoints": "masked_constraints",
}


def _case_motion_capability(
    case: BenchmarkCase,
) -> tuple[str | None, str | None]:
    """Resolve the planner capability required by one benchmark case."""
    validity = case.case_parameters.get(
        "motion_validity", "ordered_cartesian_waypoints"
    )
    if not isinstance(validity, str):
        return None, f"unsupported motion_validity mode {validity!r}"
    capability = _MOTION_VALIDITY_CAPABILITIES.get(validity)
    if capability is None:
        return None, f"unsupported motion_validity mode {validity!r}"
    return capability, None


@dataclass(frozen=True)
class PlannerContext:
    """Runtime objects shared with a planner adapter."""

    robot: "Robot"
    control_part: str
    device: torch.device
    sample_interval: int
    robot_id: str = "unknown"
    joint_velocity_limit_rad_s: float | Sequence[float] | None = None
    """Suite speed budget intersected with effective Robot caps."""
    joint_acceleration_limit_rad_s2: float | Sequence[float] | None = None
    """The suite's acceleration limit, or ``None`` when it states none.

    An adapter that constrains its own output must use the same limits the
    metrics check it against, or a satisfied result says nothing. Velocity
    limits come from ``robot.get_qvel_limits()``, which the metrics also use;
    acceleration has no asset source and is suite-owned.
    """
    joint_jerk_limit_rad_s3: float | Sequence[float] | None = None
    """The suite's jerk limit, or ``None`` when it states none."""


class PlannerAdapter(ABC):
    """Uniform lifecycle around a motion-planning implementation."""

    capabilities: frozenset[str] = frozenset()
    configuration_features: frozenset[str] = frozenset()
    """Optional configuration contracts understood by this implementation.

    Hosts can query these before creating a simulator or constructing an
    adapter. They describe accepted configuration semantics, while instance
    ``capabilities`` describes case support under the selected configuration.
    """
    model_revision: str = "N/A"
    fixed_batch_size: int | None = None
    """Required execution batch size, if the loaded model fixes it."""
    separate_prepare: bool = False
    """Whether this backend exposes a distinct lazy preparation phase."""
    timing_is_solved: bool = False
    """Whether this backend solves trajectory timing instead of assuming it.

    Defaults to nominal, so a backend that has not declared otherwise is not
    judged on a clock it may never have solved: its dynamic-limit verdict
    reads N/A, which is visible, rather than passing or failing silently. A
    backend sets this only when its timing comes from a solve under limits.
    """

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        self.spec = spec
        self.context = context
        self._motion_limits: dict[str, object] | None = None

    @property
    def metadata(self) -> PlannerMetadata:
        """Return stable identity, role, configuration, and capabilities."""
        return PlannerMetadata(
            algorithm_id=self.spec.id,
            algorithm_role=AlgorithmRole(self.spec.role),
            adapter=self.spec.adapter,
            config_hash=stable_hash(self.spec.config),
            capabilities=self.capabilities,
            model_revision=self._resolved_model_revision(),
            timing_is_solved=self.timing_is_solved,
            supported_robots=(self.context.robot_id,),
            parameters={
                **self.spec.config,
                **(
                    {"resolved_motion_limits": dict(self._motion_limits)}
                    if self._motion_limits is not None
                    else {}
                ),
            },
        )

    def resolve_motion_limits(
        self, *, require_timing: bool = False
    ) -> dict[str, object]:
        """Freeze common budgets outside timing; expose them in run metadata.

        Backend-specific limit overrides must equal the common resolved budget.
        A stricter private budget changes the experiment and must be moved to
        the suite protocol instead of silently using a different comparison.
        """
        if self._motion_limits is None:
            reference = torch.empty(0, dtype=torch.float64)
            caps = self.context.robot.get_qvel_limits(name=self.context.control_part)
            if not isinstance(caps, torch.Tensor):
                raise ValueError("Robot must provide tensor joint velocity limits.")
            dof = caps.shape[-1]
            velocity = resolve_robot_velocity_limits(
                self.context.robot,
                self.context.control_part,
                self.context.joint_velocity_limit_rad_s,
                reference,
                dof=dof,
            )
            acceleration = resolve_dynamic_limit(
                self.context.joint_acceleration_limit_rad_s2, dof, reference
            )
            jerk = resolve_dynamic_limit(
                self.context.joint_jerk_limit_rad_s3, dof, reference
            )
            resolved = {
                "velocity": velocity,
                "acceleration": acceleration,
                "jerk": jerk,
            }
            for key, actual in resolved.items():
                if key not in self.spec.config:
                    continue
                legacy = (
                    resolve_robot_velocity_limits(
                        self.context.robot,
                        self.context.control_part,
                        self.spec.config[key],
                        reference,
                        dof=dof,
                    )
                    if key == "velocity"
                    else resolve_dynamic_limit(self.spec.config[key], dof, reference)
                )
                if (
                    actual is None
                    or legacy is None
                    or not torch.allclose(legacy, actual)
                ):
                    raise ValueError(
                        f"Planner config.{key} differs from the suite motion budget; "
                        "configure protocol motion limits for all planners instead."
                    )
            self._motion_limits = {
                key: value.tolist() if value is not None else None
                for key, value in resolved.items()
            }
        if require_timing:
            if self._motion_limits["velocity"] is None:
                raise ValueError(
                    "No usable joint velocity limit; configure "
                    "protocol.joint_velocity_limit_rad_s for placeholder asset limits."
                )
            if self._motion_limits["acceleration"] is None:
                raise ValueError(
                    "Timed planning requires protocol.joint_acceleration_limit_rad_s2."
                )
        return dict(self._motion_limits)

    def _resolved_model_revision(self) -> str:
        """Resolve the recorded model identity for this planner instance."""
        return str(self.spec.config.get("model_revision", self.model_revision))

    def availability(self) -> tuple[bool, str | None]:
        """Return whether this adapter can run in the current process."""
        return True, None

    def supports_case(self, case: BenchmarkCase) -> tuple[bool, str | None]:
        """Return whether one manifest case is representable without mutation."""
        capability, reason = _case_motion_capability(case)
        if capability is None:
            return False, reason
        if capability not in self.capabilities:
            return (
                False,
                f"case requires planner capability {capability!r}",
            )
        return True, None

    @abstractmethod
    def build(self) -> None:
        """Construct the underlying planner without preparing lazy backends."""

    def prepare(self, case: BenchmarkCase) -> dict[str, object] | None:
        """Prepare a lazy backend, or return ``None`` when not applicable."""
        return None

    def prepare_cases(self, cases: list[BenchmarkCase]) -> dict[str, object] | None:
        """Prepare all backend variants required by a supported case set."""
        if not cases:
            raise ValueError("prepare_cases requires at least one supported case.")
        return self.prepare(cases[0])

    @property
    def motion_policy_planner(self) -> str:
        """Return the backend name pinned into Atomic Action motion policies."""
        return self.spec.adapter

    def require_motion_generator(self) -> "MotionGenerator":
        """Return the adapter-owned MotionGenerator or fail clearly.

        Atomic Skill scenarios use this boundary instead of reaching into a
        backend implementation.  Every planner that opts into the
        ``atomic_action`` capability must expose its generator here.
        """
        motion_generator = getattr(self, "motion_generator", None)
        if motion_generator is None:
            raise RuntimeError(
                f"Planner adapter {self.spec.id!r} does not expose a MotionGenerator."
            )
        return motion_generator

    def atomic_plan_options(
        self, *, control_dt: float | None = None
    ) -> "PlanOptions | None":
        """Return planner options for Atomic Skill calls that leave them unset.

        Atomic Skill providers stay planner-neutral and usually build motion
        policies without backend options.  A backend whose own defaults cannot
        plan such calls returns the options it uses for free-space cases.

        Args:
            control_dt: Host control period in seconds, when known. Backends may
                sample a solved trajectory on this clock to avoid interpolation
                that would invalidate its motion limits.

        Returns:
            Options applied to every motion policy without ``plan_opts``, or
            ``None`` to keep the backend defaults.
        """
        return None

    @abstractmethod
    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan one env-batched benchmark case."""

    def _close_motion_generator(self) -> None:
        """Release ``motion_generator.planner`` when adapters own one."""
        motion_generator = getattr(self, "motion_generator", None)
        if motion_generator is None:
            return
        close_fn = getattr(getattr(motion_generator, "planner", None), "close", None)
        if close_fn is not None:
            close_fn()
        self.motion_generator = None

    def close(self) -> None:
        """Release backend resources when the implementation exposes them."""
