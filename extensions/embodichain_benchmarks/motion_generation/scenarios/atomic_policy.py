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

"""Benchmark motion policy, attribution, and shared motion budgets."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from embodichain.lab.sim.atomic_actions import ActionInvocation, MotionPolicy

from ..config import ProtocolCfg

__all__ = [
    "benchmark_motion_policy_strategy",
    "benchmark_motion_policy",
    "benchmark_motion_attribution",
    "policy_motion_budget",
    "with_motion_budget",
    "with_plan_options",
]


EXACT_CARTESIAN_SKILLS = frozenset(
    {"open_door", "press", "push_object", "slide", "twist"}
)


def benchmark_motion_policy_strategy(
    skill_id: str,
) -> Literal["motion_gen", "ik_interp"]:
    """Return the auditable motion strategy required by one benchmark skill."""
    if skill_id in EXACT_CARTESIAN_SKILLS:
        return "ik_interp"
    return "motion_gen"


def benchmark_motion_policy(skill_id: str, sample_count: int) -> MotionPolicy:
    """Build the benchmark policy without hiding exact-path planner bypasses."""
    return MotionPolicy(
        strategy=benchmark_motion_policy_strategy(skill_id),
        sample_count=sample_count,
    )


def benchmark_motion_attribution(skill_id: str) -> dict[str, object]:
    """Describe whether the selected planner backend produced the motion."""
    strategy = benchmark_motion_policy_strategy(skill_id)
    if skill_id == "coordinated_pickment":
        # This primitive owns IK projection and samples on the fixed control
        # clock; its requested strategy does not call the selected backend.
        return {
            "motion_policy_strategy": strategy,
            "selected_planner_backend_used": False,
            "effective_motion_algorithm": "coordinated_ik_projection",
            "timing_is_solved": False,
        }
    selected_backend_used = strategy == "motion_gen"
    return {
        "motion_policy_strategy": strategy,
        "selected_planner_backend_used": selected_backend_used,
        "effective_motion_algorithm": (
            "selected_planner_backend"
            if selected_backend_used
            else "sequential_ik_interpolation"
        ),
    }


def policy_motion_budget(protocol: ProtocolCfg) -> dict[str, float]:
    """Map suite requests to the scalar interface of scenario-owned motion."""
    values: dict[str, float] = {}
    for source, target in (
        ("joint_velocity_limit_rad_s", "velocity_limit"),
        ("joint_acceleration_limit_rad_s2", "acceleration_limit"),
    ):
        requested = getattr(protocol, source)
        if requested is not None:
            values[target] = (
                float(requested)
                if isinstance(requested, (int, float))
                else min(float(value) for value in requested)
            )
    return values


def with_motion_budget(
    invocation: ActionInvocation, protocol: ProtocolCfg
) -> ActionInvocation:
    """Apply shared scalar budgets to IK paths without overriding backend vectors.

    MotionPolicy accepts scalars. Per-joint suite requests use their minimum
    for these scenario-owned paths, conservatively satisfying every request;
    the robot still caps each joint separately. Omitted requests retain the
    invocation defaults. Selected backends receive the full vector through
    their existing planner options.
    """
    if (
        invocation.motion_policy.strategy != "ik_interp"
        and invocation.skill_id != "coordinated_pickment"
    ):
        return invocation
    values = policy_motion_budget(protocol)
    if not values:
        return invocation
    return replace(
        invocation, motion_policy=replace(invocation.motion_policy, **values)
    )


def with_plan_options(
    invocation: ActionInvocation, plan_opts: object
) -> ActionInvocation:
    """Inject host options without replacing their vector budgets by defaults.

    Explicit invocation options remain authoritative. Selected-backend options
    own the suite's full per-joint budgets, so scenario scalar defaults must not
    override them. Scenario-owned IK retains its scalar policy limits.
    """
    policy = invocation.motion_policy
    if policy.strategy != "motion_gen" or policy.plan_opts is not None:
        return invocation
    limits = (
        {"velocity_limit": None, "acceleration_limit": None}
        if benchmark_motion_attribution(invocation.skill_id)[
            "selected_planner_backend_used"
        ]
        else {}
    )
    return replace(
        invocation, motion_policy=replace(policy, plan_opts=plan_opts, **limits)
    )
