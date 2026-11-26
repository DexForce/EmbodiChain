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

"""Robot assemblies and TCPs matched to the published NMG v1 policies."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from embodichain.data import get_data_path
from embodichain.extensions.robots import CobotMagicCfg, FrankaCfg
from embodichain.lab.sim.cfg import RobotCfg, URDFCfg
from embodichain.lab.sim.motion.solvers import OPWSolverCfg
from embodichain.lab.sim.robots.dual_arm import build_dual_arm_cfg, resolve_mounts
from embodichain.robot_registry import resolve_gripper_defaults


def make_robot_cfg(robot: str) -> RobotCfg:
    """Build a fixed-base arm with the policy's joint order and grasp TCP."""
    if robot == "franka":
        cfg = FrankaCfg.from_dict({"arm_model": "fr3"})
        hand = resolve_gripper_defaults("RobotiqArg2F140")
        cfg.urdf_cfg.fname = "nmg_example_franka_v1"
        cfg.urdf_cfg.add_component("hand", get_data_path(hand["fpath"]), np.eye(4))
        cfg.control_parts["eef"] = list(hand["control_parts"]["eef"])
        cfg.init_qpos = list(cfg.init_qpos) + list(hand["init_qpos"])
        for field in ("stiffness", "damping", "max_effort"):
            for joint in cfg.control_parts["eef"]:
                getattr(cfg.joint_drive_props, field)[joint] = hand[
                    "joint_drive_props"
                ][field]
        cfg.solver_cfg["arm"].root_link_name = "fr3_link0"
        cfg.solver_cfg["arm"].tcp = [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.20],
            [0.0, 0.0, 0.0, 1.0],
        ]
    elif robot == "piper":
        hand = resolve_gripper_defaults("CobotMagicV100Gripper")
        native = CobotMagicCfg.from_dict({})
        drives = deepcopy(native.joint_drive_props)
        for field in drives.__dataclass_fields__:
            values = getattr(drives, field)
            if isinstance(values, dict):
                setattr(
                    drives,
                    field,
                    {
                        key.removeprefix("left_"): value
                        for key, value in values.items()
                        if key.startswith("left_")
                    },
                )
        cfg = RobotCfg(
            urdf_cfg=URDFCfg(
                components=[
                    {
                        "component_type": "arm",
                        "urdf_path": get_data_path(hand["fpath"]),
                        "transform": np.eye(4),
                    }
                ]
            ),
            control_parts={
                "arm": [f"joint{i}" for i in range(1, 7)],
                "eef": list(hand["control_parts"]["eef"]),
            },
            init_qpos=[0.0, 1.0, -1.2, 0.0, 0.6, 0.0, 0.05, 0.05],
            joint_drive_props=drives,
            attrs=deepcopy(native.attrs),
            root_props=deepcopy(native.root_props),
            solver_cfg={
                "arm": OPWSolverCfg(
                    root_link_name="arm_base",
                    end_link_name="link6",
                    tcp=hand["default_root_from_tcp"],
                )
            },
        )
    else:
        raise ValueError(f"Unknown robot {robot!r}")
    cfg.uid = f"nmg_{robot}"
    cfg.root_props.fixed_base = True
    return cfg


def make_dual_franka_cfg() -> RobotCfg:
    """Mount two policy-matched FR3 arms with Robotiq grippers."""
    base = make_robot_cfg("franka")
    mounts = resolve_mounts({"preset": "mirrored_rz", "separation": 0.92, "rz": 0.15})
    cfg = build_dual_arm_cfg(base, mounts)
    cfg.uid = "nmg_dual_franka"
    cfg.urdf_cfg.fname = "nmg_example_dual_franka_v1"
    hand = base.urdf_cfg.components["hand"]
    for side in ("left", "right"):
        cfg.urdf_cfg.add_component(
            f"{side}_hand",
            hand["urdf_path"],
            hand["transform"],
            **hand.get("params", {}),
        )
    arm_home = base.init_qpos[: len(base.control_parts["arm"])]
    hand_home = base.init_qpos[len(arm_home) :]
    # The articulation visits the arm branches in interleaved joint order.
    cfg.init_qpos = [
        value for joint in arm_home for value in (joint, joint)
    ] + hand_home * 2
    return cfg
