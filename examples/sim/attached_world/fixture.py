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

"""Create the two-robot attachment acceptance fixture in a dedicated DexSim host."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from dexsim.runtime import Session, SessionContext

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.dexsim_session.attachment import AttachedSimulationManager
from embodichain.lab.sim.cfg import RobotCfg, RobotPresetCfg, RigidObjectCfg
from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers
from embodichain.robot_registry import resolve_robot_cfg

TASK = (
    Path(__file__).resolve().parents[2]
    / "expansion/asset_generalization/configs/cube_platform_pick_place/task.ur5.yaml"
)
ARM = [f"joint{i}" for i in range(1, 7)]
PATHS = {"UR5": "robot_b", "cube": "moving_cube", "platform": "support"}


def robot_config(assembled: bool) -> RobotCfg:
    if assembled:
        cfg = config_to_cfg(yaml.safe_load(TASK.read_text()), source_path=TASK).robot
        if isinstance(cfg, RobotPresetCfg):
            cfg = cfg.resolve()
        prepare_robot_grippers(cfg)
    else:
        cfg = resolve_robot_cfg("URRobot", variant="ur5")
    cfg.uid = "UR5"
    cfg.fpath = cfg.urdf_cfg.assemble_urdf()
    return cfg


def native(session: Session, method: str, **params: Any) -> dict[str, Any]:
    """Use the existing DexSim MCP dispatcher for host-owned scene operations."""
    return session.invoke("dexsim.kit.mcp.dispatch", method=method, params=params)


def populate(session: Session, asset: Path) -> None:
    populate_with(lambda method, **params: native(session, method, **params), asset)


def populate_with(call: Callable[..., dict[str, Any]], asset: Path) -> None:
    """Construct the same baseline through either a local or a remote host."""
    for index, name in enumerate(("robot_a", "robot_b")):
        call(
            "articulation.import_urdf",
            name=name,
            asset_path=asset.name,
            position=[index * 2, 0, 0],
            fixed_base=True,
        )
    for name, kind, z in (("moving_cube", "dynamic", 1.0), ("support", "static", 0.2)):
        call(
            "scene.create_primitive",
            name=name,
            primitive="cube",
            size=[0.04] * 3,
            position=[1, 1, z],
            physics=kind,
        )
    call("scene.finalize")
    for sign, name in ((1, "robot_a"), (-1, "robot_b")):
        call(
            "articulation.set_state",
            object_ref=name,
            joint_names=ARM,
            joint_position=[sign * 0.05 * i for i in range(1, 7)],
        )
        call(
            "articulation.set_joint_state",
            object_ref=name,
            joint_names=ARM,
            joint_target_position=[sign * 0.01 * i for i in range(1, 7)],
        )
    call(
        "simulation.set_rigid_body_state",
        object_ref="moving_cube",
        linear_velocity=[0.1, 0.2, 0.3],
    )


def attach(
    context: SessionContext, assembled: bool, paths: dict[str, str]
) -> AttachedSimulationManager:
    return SimulationManager.attach(
        context,
        arena=context.scene.arenas.names[0],
        robots=[robot_config(assembled)],
        rigid_objects=[
            RigidObjectCfg(uid="cube", body_type="dynamic"),
            RigidObjectCfg(uid="platform", body_type="static"),
        ],
        paths=paths,
        logical_entities={
            "manipulator": "UR5",
            "payload": "cube",
            "destination": "platform",
        },
    )


def run(context: SessionContext, *, assembled: bool, paths: dict[str, str]) -> int:
    """Install a task-local manager only after all bindings succeed."""
    manager = attach(context, assembled, paths)
    context.state["attachment"] = manager
    return manager.instance_id
