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

"""Read EmbodiChain control parts from a World borrowed from DexSim Runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from embodichain.mcp._types import RobotState
from embodichain.robot_registry import resolve_robot_cfg
from embodichain.utils.string import resolve_matching_names

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext
    from dexsim.scene import JointDofLayout
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = ["read_robot_state", "run"]


def _select_dofs(
    layout: Sequence[JointDofLayout], selectors: Sequence[str]
) -> tuple[list[str], list[int], list[str]]:
    names = [joint.name for joint in layout]
    if len(set(names)) != len(names):
        raise ValueError("Ambiguous joint names in the articulation DOF layout")
    if not selectors:
        raise ValueError("The selected control part has no joint selectors")
    rows, selected = resolve_matching_names(selectors, names, preserve_order=True)
    indices = []
    units = []
    for row in rows:
        joint = layout[row]
        if joint.dof_count != 1 or joint.position_encoding not in (
            "angle",
            "translation",
        ):
            raise ValueError(
                f"Unsupported position encoding for joint {joint.name!r}: {joint.position_encoding}"
            )
        indices.append(joint.dof_start)
        units.append("rad" if joint.position_encoding == "angle" else "m")
    return selected, indices, units


def read_robot_state(
    context: SessionContext,
    *,
    object_ref: str,
    robot_cfg: RobotCfg,
    control_part: str,
) -> RobotState:
    """Read measured scalar joint positions on the existing World's owner thread.

    Args:
        context: Borrowed DexSim host context supplied by Session.invoke.
        object_ref: Path of the articulation in the finalized scene.
        robot_cfg: Resolved configuration declaring the required control parts.
        control_part: Configured group whose selectors specify output order.

    Returns:
        Current joint positions in radians or metres and their public DOF indices.
        Clock fields are null on backends without a public simulation clock.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("The attached World has no finalized scene")
    handle = scene.get_articulation(object_ref)
    if (
        scene.backend == "newton"
        and (scene.get_solver_diagnostics() or {}).get("solver_type") != "mujoco_warp"
    ):
        raise ValueError("Reading Newton joint state requires the mujoco_warp solver")
    parts = robot_cfg.control_parts or {}
    if control_part not in parts:
        raise ValueError(
            f"Unknown control part {control_part!r} for {robot_cfg.uid!r}. Available: {list(parts)}"
        )
    names, indices, units = _select_dofs(handle.joint_dof_layout, parts[control_part])
    positions = np.asarray(handle.get_current_qpos()).reshape(-1)[indices]
    if not np.isfinite(positions).all():
        raise ValueError(
            f"Non-finite measured positions for {object_ref!r}/{control_part}"
        )
    return RobotState(
        session_id=context.session_id,
        object_ref=handle.path,
        topology_revision=scene.topology_revision,
        control_part=control_part,
        joint_names=names,
        dof_ids=indices,
        qpos=positions.tolist(),
        units=units,
        step_index=scene.get_simulation_step() if scene.backend == "newton" else None,
        simulation_time_s=(
            scene.get_simulation_time() if scene.backend == "newton" else None
        ),
    )


def run(
    context: SessionContext,
    *,
    object_ref: str,
    robot_cfg: str,
    control_part: str,
    variant: str | None = None,
) -> RobotState:
    """DexSim Session.invoke entry point for a registered robot ID and variant."""
    cfg = resolve_robot_cfg(robot_cfg, variant=variant)
    return read_robot_state(
        context, object_ref=object_ref, robot_cfg=cfg, control_part=control_part
    )
