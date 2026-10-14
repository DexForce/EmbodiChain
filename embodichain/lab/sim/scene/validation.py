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

"""Measured scene checks and caller-driven settling without task dependencies."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal
from embodichain.utils import configclass
from .query import SceneView
from .state import SceneState, capture_state

__all__ = [
    "SceneCheck",
    "SceneCheckReport",
    "SettlingCfg",
    "validate_scene",
    "compare_states",
    "settle_scene",
]


@dataclass(frozen=True)
class SceneCheck:
    """One measured required check; missing evidence is unverified."""

    name: str
    status: Literal["passed", "failed", "unverified"]
    message: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.status not in ("passed", "failed", "unverified")
        ):
            raise ValueError("Scene checks need a name and supported status")
        metrics = dict(self.metrics)
        if any(
            not isinstance(key, str)
            or not key
            or type(value) not in (int, float)
            or not math.isfinite(value)
            for key, value in metrics.items()
        ):
            raise ValueError("Scene check metrics must be finite numbers with names")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@dataclass(frozen=True)
class SceneCheckReport:
    """Fail-closed required-check collection."""

    checks: tuple[SceneCheck, ...]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if any(not isinstance(check, SceneCheck) for check in checks) or len(
            {check.name for check in checks}
        ) != len(checks):
            raise ValueError("Scene checks must be valid with unique names")
        object.__setattr__(self, "checks", checks)

    @property
    def passed(self) -> bool:
        """Return true only when nonempty evidence passes every required check."""
        return bool(self.checks) and all(
            check.status == "passed" for check in self.checks
        )


def validate_scene(
    checks: Iterable[SceneCheck | Callable[[], SceneCheck]],
) -> SceneCheckReport:
    """Evaluate explicit checks of actual state without stepping physics.

    Args:
        checks: Measured results or zero-argument measurement callbacks.
    Returns:
        Required evidence; an empty report cannot accept a candidate.
    """
    return SceneCheckReport(
        tuple(check() if callable(check) else check for check in checks)
    )


def compare_states(expected: SceneState, actual: SceneState) -> SceneCheckReport:
    """Compare physical fields with quaternion sign equivalence.

    Args:
        expected: Original physical snapshot.
        actual: Snapshot measured after restoration.
    Returns:
        Restore check using absolute qpos tolerance 1e-4 and all others 1e-5.
    """
    if (
        expected.env_ids != actual.env_ids
        or expected.physics_backend != actual.physics_backend
        or expected._topology != actual._topology
        or expected._owner is not actual._owner
    ):
        return SceneCheckReport(
            (
                SceneCheck(
                    "scene.restore_state",
                    "failed",
                    "State ownership, topology, backend or environment selection differs",
                ),
            )
        )
    failed = []
    max_error = max_qpos_error = max_other_error = 0.0
    for before_batch, after_batch in zip(expected.batches, actual.batches, strict=True):
        if tuple(
            (row.kind, row.uid, row.body_type, row.joint_names) for row in before_batch
        ) != tuple(
            (row.kind, row.uid, row.body_type, row.joint_names) for row in after_batch
        ):
            return SceneCheckReport(
                (
                    SceneCheck(
                        "scene.restore_state",
                        "failed",
                        "State entity identities differ",
                    ),
                )
            )
        for before_row, after_row in zip(before_batch, after_batch, strict=True):
            for name in (
                "pose",
                "velocity",
                "qpos",
                "qvel",
                "target_qpos",
                "target_qvel",
                "qf",
            ):
                before, after = getattr(before_row, name), getattr(after_row, name)
                if (
                    name == "pose"
                    and sum(a * b for a, b in zip(before[3:], after[3:])) < 0
                ):
                    after = (*after[:3], *(-v for v in after[3:]))
                error = max(
                    (abs(a - b) for a, b in zip(before, after, strict=True)),
                    default=0.0,
                )
                max_error = max(max_error, error)
                if name == "qpos":
                    max_qpos_error = max(max_qpos_error, error)
                    tolerance = 1e-4
                else:
                    max_other_error = max(max_other_error, error)
                    tolerance = 1e-5
                if error > tolerance:
                    failed.append(f"{before_row.uid}.{name}")
    return SceneCheckReport(
        (
            SceneCheck(
                "scene.restore_state",
                "failed" if failed else "passed",
                (
                    "State differs after restoration: " + ", ".join(failed)
                    if failed
                    else "Measured qpos restored within 1e-4; all other saved physical fields within 1e-5; quaternion sign equivalence allowed"
                ),
                {
                    "max_abs_state_error": max_error,
                    "absolute_tolerance": 1e-5,
                    "qpos_absolute_tolerance": 1e-4,
                    "max_abs_qpos_error": max_qpos_error,
                    "max_abs_other_state_error": max_other_error,
                },
            ),
        )
    )


@configclass
class SettlingCfg:
    """Caller-driven settling budget and optional measured convergence thresholds.

    Fixed steps deliberately produce unverified stability evidence. Convergence
    requires consecutive measurements below all velocity thresholds.
    """

    mode: Literal["fixed_steps", "convergence"] = "fixed_steps"
    max_steps: int = 300
    linear_threshold: float = 0.01
    angular_threshold: float = 0.01
    joint_threshold: float = 0.01
    consecutive_steps: int = 5

    def __post_init__(self) -> None:
        if (
            self.mode not in ("fixed_steps", "convergence")
            or type(self.max_steps) is not int
            or self.max_steps < 0
            or type(self.consecutive_steps) is not int
            or self.consecutive_steps <= 0
        ):
            raise ValueError("Invalid settling mode or step budget")
        for value in (
            self.linear_threshold,
            self.angular_threshold,
            self.joint_threshold,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "Settling thresholds must be finite nonnegative numbers"
                )


def settle_scene(
    view: SceneView, step: Callable[[int], None], cfg: SettlingCfg
) -> SceneCheckReport:
    """Run caller-owned physics steps and measure optional velocity convergence.

    Args:
        view: B=1 scene; no temporary simulation is created here.
        step: Host callback advancing the supplied number of physics substeps.
        cfg: Fixed-step or convergence policy. This advances the whole world.
    Returns:
        Stability evidence, failed on timeout and unverified for fixed steps.
    """
    cfg.__post_init__()
    view.check_topology()
    if cfg.mode == "fixed_steps":
        if cfg.max_steps:
            step(cfg.max_steps)
        return SceneCheckReport(
            (
                SceneCheck(
                    "scene.stability",
                    "unverified",
                    "Fixed steps completed without convergence evidence",
                    {"steps": cfg.max_steps},
                ),
            )
        )
    # Verify the measurement contract before advancing the world.
    initial = capture_state(view)
    if not initial.batches[0]:
        return SceneCheckReport(
            (
                SceneCheck(
                    "scene.stability",
                    "unverified",
                    "No physical entities to measure",
                    {"steps": 0},
                ),
            )
        )
    count = 0
    metrics = {"steps": 0}
    for index in range(cfg.max_steps):
        step(1)
        snapshot = capture_state(view)
        rows = snapshot.batches[0]
        if not rows:
            return SceneCheckReport(
                (
                    SceneCheck(
                        "scene.stability",
                        "unverified",
                        "No physical entities to measure",
                        {"steps": index + 1},
                    ),
                )
            )
        linear = max(math.sqrt(sum(v * v for v in row.velocity[:3])) for row in rows)
        angular = max(math.sqrt(sum(v * v for v in row.velocity[3:])) for row in rows)
        joint = max((abs(v) for row in rows for v in row.qvel), default=0.0)
        metrics = {
            "steps": index + 1,
            "max_linear_speed": linear,
            "max_angular_speed": angular,
            "max_joint_speed": joint,
            "linear_threshold": cfg.linear_threshold,
            "angular_threshold": cfg.angular_threshold,
            "joint_threshold": cfg.joint_threshold,
        }
        count = (
            count + 1
            if linear <= cfg.linear_threshold
            and angular <= cfg.angular_threshold
            and joint <= cfg.joint_threshold
            else 0
        )
        if count >= cfg.consecutive_steps:
            return SceneCheckReport(
                (
                    SceneCheck(
                        "scene.stability",
                        "passed",
                        "Measured velocity convergence",
                        metrics,
                    ),
                )
            )
    return SceneCheckReport(
        (
            SceneCheck(
                "scene.stability", "failed", "Convergence budget exhausted", metrics
            ),
        )
    )
