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

"""Physical fixtures for the embodiment-switching Atomic Skills walkthrough.

These are simulation calibrations. Task recipes stay in the teaching script;
mesh mounting, tool frames, jaw commands and scene placement belong here.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim.atomic_actions import ControlPartCommandProfile
from embodichain.lab.sim.cfg import RigidObjectCfg, RobotCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.toolkits.graspkit import (
    ParallelJawGripperModelCfg,
    get_parallel_jaw_gripper_model,
)
from embodichain.utils import configclass, logger
from embodichain_robots import (
    ARX5Cfg,
    CobotMagicCfg,
    DexforceW1Cfg,
    FrankaPandaCfg,
    TianjiMarvinCfg,
    URRobotCfg,
)
from .support import (
    create_tutorial_argument_parser,
    create_tutorial_rigid_body_physics,
)

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import CompiledTrajectory
    from embodichain.lab.sim.objects import RigidObject, Robot


@configclass
class EmbodimentCfg:
    """Resolved construction and resource configuration for one walkthrough."""

    robot_cfg: RobotCfg = MISSING
    binding: dict[str, dict[str, str]] = {}
    models: dict[str, ParallelJawGripperModelCfg] = {}
    commands: dict[str, dict[str, tuple[float, ...]]] = {}
    object_size: tuple[float, float, float] = (0.04, 0.04, 0.04)
    # Per-robot task-frame placement, shared by both mounted-gripper choices.
    task_origin: tuple[float, float, float] = (-0.32, 0.0, 0.20)
    ready_clearance: float = 0.27
    task_yaw_degrees: float = 0.0
    handover_lift_height: float = 0.10
    initialize_above_object: bool = True
    ready_seeds: dict[str, tuple[float, ...]] = {}
    ready_yaws: dict[str, float] = {}
    destination_half_height: float = 0.01
    destination_lateral: float = -0.24
    preview_look_at: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None

    def command_profiles(
        self, device: torch.device
    ) -> dict[str, ControlPartCommandProfile]:
        """Materialize explicit jaw commands on the runtime device."""
        return {
            part: ControlPartCommandProfile.joint_positions(
                **{key: torch.tensor(q, device=device) for key, q in commands.items()}
            )
            for part, commands in self.commands.items()
        }


def parse_arguments() -> argparse.Namespace:
    """Validate the supported task/robot/tool matrix before creating simulation."""
    parser = create_tutorial_argument_parser(
        "Reuse Atomic Skills across robot and gripper configurations.",
        features=("grasp_sampling",),
        default_device="cpu",
        default_renderer="hybrid",
        robot_choices=("ur5", "franka", "arx5", "cobotmagic", "tianji", "w1_030"),
        default_robot=None,
    )
    parser.add_argument(
        "--task", choices=("pick_place", "handover"), default="pick_place"
    )
    parser.add_argument(
        "--gripper", choices=("auto", "pika", "dh_ag95"), default="auto"
    )
    args = parser.parse_args()
    if args.robot is None:
        args.robot = "ur5" if args.task == "pick_place" else "w1_030"
    robots = (
        ("ur5", "franka", "arx5")
        if args.task == "pick_place"
        else ("cobotmagic", "tianji", "w1_030")
    )
    if args.robot not in robots:
        parser.error(f"{args.task} supports robots {', '.join(robots)}")
    mounted = args.robot in ("ur5", "w1_030")
    if args.gripper != "auto" and not mounted:
        parser.error("--gripper pika/dh_ag95 is supported on ur5 and w1_030")
    if args.gripper == "auto":
        args.gripper = "pika" if mounted else "native"
    if args.num_envs != 1:
        parser.error("This walkthrough uses exactly one environment")
    return args


def _mounted_gripper(
    name: str,
) -> tuple[str, tuple[str, ...], dict, np.ndarray, ParallelJawGripperModelCfg]:
    """Return asset-local metadata; transforms map flange to canonical grasp TCP."""
    if name == "pika":
        # Mesh finger span is measured in GRIPPER_BASE; the root contains the
        # adapter rotation. At the grasp center TCP X opens and TCP Z approaches.
        tcp = np.array(
            [[0, 1, 0, 0.00064], [-1, 0, 0, 0], [0, 0, 1, 0.195], [0, 0, 0, 1]],
            dtype=float,
        )
        return (
            "PikaGripper/PikaGripper.urdf",
            ("FINGER1_JOINT", "FINGER2_JOINT"),
            {"open": (0.05, 0.05), "grasp": (0.0, 0.0)},
            tcp,
            ParallelJawGripperModelCfg(
                model_id="tutorial_pika",
                max_opening_width=0.1,
                finger_length=0.055,
                finger_width=0.025,
                finger_thickness=0.012,
                palm_depth=0.08,
            ),
        )
    tcp = np.eye(4)
    tcp[2, 3] = 0.19
    # All five followers are explicitly commanded in their source order; mimic
    # lowering remains the robot runtime's responsibility.
    joints = (
        "left_outer_knuckle_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    )
    return (
        "DH_AG95/dh_ag95.urdf",
        joints,
        {"open": (0.0,) * 6, "grasp": (0.90,) * 6},
        tcp,
        ParallelJawGripperModelCfg(
            model_id="tutorial_dh_ag95",
            max_opening_width=0.095,
            finger_length=0.055,
            finger_width=0.025,
            finger_thickness=0.012,
            palm_depth=0.14,
        ),
    )


def _mount_grippers(args: argparse.Namespace, cfg: EmbodimentCfg) -> None:
    asset, joints, commands, tcp, model = _mounted_gripper(args.gripper)
    path = get_data_path(asset)
    if args.robot == "ur5":
        cfg.robot_cfg = URRobotCfg.from_dict(
            {
                "robot_type": "ur5",
                "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
                + list(commands["open"]),
                "urdf_cfg": {
                    "fname": f"tutorial_ur5_{args.gripper}",
                    "components": [
                        {
                            "component_type": "hand",
                            "urdf_path": path,
                            "transform": np.eye(4),
                        }
                    ],
                },
                "control_parts": {"eef": list(joints)},
                "solver_cfg": {"arm": {"tcp": tcp}},
            }
        )
        cfg.commands = {"eef": commands}
        cfg.models = {"eef": model}
    else:
        mount = np.eye(4)
        cfg.robot_cfg = DexforceW1Cfg.from_dict(
            {
                "version": "v030",
                "with_default_eef": False,
                "urdf_cfg": {
                    "fname": f"tutorial_w1_030_{args.gripper}",
                    "components": [
                        {
                            "component_type": f"{side}_hand",
                            "urdf_path": path,
                            "transform": mount,
                        }
                        for side in ("left", "right")
                    ],
                },
                "control_parts": {
                    f"{side}_eef": [
                        (j if j.startswith(f"{side}_") else f"{side}_{j}").upper()
                        for j in joints
                    ]
                    for side in ("left", "right")
                },
                "solver_cfg": {
                    f"{side}_arm": {"tcp": tcp} for side in ("left", "right")
                },
            }
        )
        cfg.commands = {f"{side}_eef": commands for side in ("left", "right")}
        cfg.models = {f"{side}_eef": model for side in ("left", "right")}
    cfg.robot_cfg.urdf_cfg.fpath = str(
        Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
        / "assembled"
        / f"atomic_{args.robot}_{args.gripper}"
        / "robot.urdf"
    )


def create_embodiment(args: argparse.Namespace) -> EmbodimentCfg:
    """Resolve robot-specific physical configuration outside the task recipe."""
    cfg = EmbodimentCfg()
    if args.robot in ("ur5", "w1_030"):
        _mount_grippers(args, cfg)
    elif args.robot == "franka":
        cfg.robot_cfg = FrankaPandaCfg.from_dict({"init_rot": (0, 0, 180)})
        # Start in the sampled cube grasp's equivalent yaw, keeping the wrist
        # away from its limit and the redundant elbow on one continuous branch.
        cfg.ready_yaws = {"primary": 90.0}
        cfg.robot_cfg.root_props.self_collision_enabled = True
        cfg.models = {"eef": get_parallel_jaw_gripper_model("franka_panda_hand")}
        cfg.commands = {"eef": {"open": (0.04, 0.04), "grasp": (0.0, 0.0)}}
    elif args.robot == "arx5":
        cfg.robot_cfg = ARX5Cfg.from_dict({})
        # Inherit the preset's outward elbow, unflipped wrist and self-collision
        # settings. Keep this fixture on that branch. The former
        # high table required a folded wrist that intersected the forearm.
        # These fixture limits leave the reusable preset's limits intact.
        cfg.robot_cfg.solver_cfg["arm"].user_qpos_limits = [
            [-3.7, -2.0],
            [1.0, 3.0],
            [0.05, 2.8],
            [-2.5, -0.2],
            [-0.15, 0.15],
            [-2.5, -0.2],
        ]
        cfg.task_origin = (-0.32, 0.0, 0.05)
        cfg.ready_clearance = 0.10
        cfg.preview_look_at = (
            (-0.95, -0.75, 0.65),
            (-0.14, 0.0, 0.22),
            (0.0, 0.0, 1.0),
        )
        cfg.models = {
            "eef": ParallelJawGripperModelCfg(
                model_id="tutorial_arx5",
                max_opening_width=0.088,
                finger_length=0.05,
                finger_width=0.025,
                finger_thickness=0.01,
                palm_depth=0.065,
            )
        }
        cfg.commands = {"eef": {"open": (0.044, 0.044), "grasp": (0.0, 0.0)}}
    elif args.robot == "cobotmagic":
        # Inclined bench mounts keep the downward grasp path inside the
        # V100 wrist's authored range. Both source arm assets remain intact.
        cfg.robot_cfg = CobotMagicCfg.from_dict(
            {"init_pos": (0.0, 0.0, 0.50), "init_rot": (0.0, 35.0, 0.0)}
        )
        cfg.ready_clearance = 0.15
        cfg.handover_lift_height = 0.06
        # Equivalent jaw orientations put both wrist-camera housings on the
        # same side of the beam. Opposing housings intersect at exchange.
        # Rotate and lower this fixture to retain the source wrist limits
        # through release/retraction, then place within the receiver workspace.
        cfg.task_yaw_degrees = -35.0
        cfg.destination_lateral = -0.10
        cfg.ready_yaws = {"source": 180.0, "destination": 180.0}
        cfg.ready_seeds = {
            "left_arm": (-0.52, 1.40, -1.25, 0.35, 0.99, -1.60),
            "right_arm": (0.52, 1.40, -1.25, -0.35, 0.99, -0.50),
        }
        cfg.robot_cfg.root_props.self_collision_enabled = True
        from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

        # Follow the actual V100 source chain and align TCP X with its Y-axis
        # finger travel, instead of inheriting the catalog's source-tool axes.
        cfg.robot_cfg.solver_cfg = {
            part: PytorchSolverCfg(
                root_link_name=solver.root_link_name,
                end_link_name=solver.end_link_name,
                tcp=[[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                enable_seed_selection=True,
            )
            for part, solver in cfg.robot_cfg.solver_cfg.items()
        }
        cfg.models = {
            f"{s}_eef": get_parallel_jaw_gripper_model("cobotmagic_v100_gripper")
            for s in ("left", "right")
        }
        cfg.commands = {
            f"{s}_eef": {"open": (0.05, 0.05), "grasp": (0.0, 0.0)}
            for s in ("left", "right")
        }
    else:
        # Reuse the explicit opt-in simulation calibration, including its
        # model-derived grasp axes and source-inertia overlay. No cameras or
        # Task Program runtime are created by this direct Atomic caller.
        import yaml
        from importlib.resources import files

        path = files("embodichain_tasks.configs").joinpath(
            "components/embodiments/tianji_marvin.yaml"
        )
        data = yaml.safe_load(path.read_text())
        cfg.robot_cfg = TianjiMarvinCfg.from_dict(
            {k: v for k, v in data["simulation"].items() if k != "class_type"}
        )
        cfg.ready_clearance = 0.15
        cfg.models = {
            f"{s}_eef": get_parallel_jaw_gripper_model("tianji_marvin_gripper")
            for s in ("left", "right")
        }
        cfg.commands = {
            f"{s}_eef": {"open": (0.0, 0.0), "grasp": (-0.05, -0.05)}
            for s in ("left", "right")
        }
    if args.task == "pick_place":
        cfg.binding = {"primary": {"motion": "arm", "grasp": "eef"}}
    else:
        cfg.binding = {
            role: {"motion": f"{side}_arm", "grasp": f"{side}_eef"}
            for role, side in (("source", "left"), ("destination", "right"))
        }
        cfg.object_size = (0.035, 0.32, 0.035)
        cfg.destination_half_height = 0.06
        cfg.task_origin = {
            "cobotmagic": (0.52, 0.0, 0.16),
            "tianji": (0.42, 0.0, 0.75),
            "w1_030": (0.42, -0.06, 0.8),
        }[args.robot]
        if args.robot == "w1_030":
            cfg.destination_lateral = -0.10
        # Look along the robots' forward X axis so source and destination
        # remain visible on opposite sides, with the complete body in frame.
        cfg.preview_look_at = {
            "cobotmagic": (
                (1.65, 0.0, 0.85),
                (0.35, 0.0, 0.30),
                (0.0, 0.0, 1.0),
            ),
            "tianji": ((2.1, 0.0, 1.25), (0.12, 0.0, 0.62), (0.0, 0.0, 1.0)),
            "w1_030": (
                (2.7, cfg.task_origin[1], 1.65),
                (0.12, cfg.task_origin[1], 0.88),
                (0.0, 0.0, 1.0),
            ),
        }[args.robot]
    # Task-scoped, portable contact calibration; source inertias are retained.
    cfg.robot_cfg.root_props.fixed_base = True
    cfg.robot_cfg.root_props.min_position_iters = 32
    cfg.robot_cfg.root_props.min_velocity_iters = 8
    drive = cfg.robot_cfg.joint_drive_props
    arm_parts = [part["motion"] for part in cfg.binding.values()]
    for name, value in (("stiffness", 5e4), ("damping", 5e3), ("max_effort", 1e5)):
        values = getattr(drive, name)
        values = (
            dict(values)
            if isinstance(values, dict)
            else ({".*": values} if values is not None else {})
        )
        values.update({part: value for part in arm_parts})
        setattr(drive, name, values)
    for name, value in (
        ("stiffness", 500.0 if args.robot == "tianji" else 1e3),
        ("damping", 50.0 if args.robot == "tianji" else 1e2),
        ("max_effort", 3.0 if args.gripper == "dh_ag95" else 40.0),
        (
            "max_velocity",
            (
                0.1
                if args.robot == "tianji"
                else (0.25 if args.gripper != "dh_ag95" else 2.1)
            ),
        ),
    ):
        values = getattr(drive, name)
        values = (
            dict(values)
            if isinstance(values, dict)
            else ({".*": values} if values is not None else {})
        )
        values.update({part: value for part in cfg.commands})
        setattr(drive, name, values)
    cfg.validate()
    return cfg


def create_scene(
    sim: SimulationManager, cfg: EmbodimentCfg
) -> tuple[Robot, RigidObject, RigidObject]:
    """Construct one robot, a free object, and the destination support pad."""
    robot = sim.add_robot(cfg=cfg.robot_cfg)
    x, y, surface = cfg.task_origin
    angle = math.radians(cfg.task_yaw_degrees)
    sine, cosine = math.sin(angle), math.cos(angle)

    def position(lateral: float, height: float) -> tuple[float, float, float]:
        return (x - sine * lateral, y + cosine * lateral, height)

    rotation = (0.0, 0.0, cfg.task_yaw_degrees)
    dual = "source" in cfg.binding
    start_y = 0.14 if dual else -0.10
    target_y = cfg.destination_lateral if dual else 0.12
    sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="table",
            shape=CubeCfg(size=(0.18, 0.95 if dual else 0.5, 0.04)),
            body_type="static",
            init_pos=position(0.0, surface - 0.02 - (0.10 if dual else 0.0)),
            init_rot=rotation,
            attrs=create_tutorial_rigid_body_physics(
                static_friction=1.0, dynamic_friction=1.0
            ),
        )
    )
    if dual:
        # Leave air beneath the two grasp regions, including the long native
        # fingers. The shared beam rests freely on its central support.
        sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="pickup_support",
                shape=CubeCfg(size=(0.10, 0.06, 0.10)),
                body_type="static",
                init_pos=position(start_y, surface - 0.05),
                attrs=create_tutorial_rigid_body_physics(
                    static_friction=1.0, dynamic_friction=1.0
                ),
            )
        )
    pad = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="destination",
            shape=CubeCfg(
                size=(
                    0.10 if dual else 0.14,
                    0.06 if dual else 0.14,
                    2.0 * cfg.destination_half_height,
                )
            ),
            body_type="static",
            init_pos=position(target_y, surface + 0.02 - cfg.destination_half_height),
            init_rot=rotation,
            attrs=create_tutorial_rigid_body_physics(
                static_friction=1.0, dynamic_friction=1.0
            ),
        )
    )
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="transfer_object",
            shape=CubeCfg(size=cfg.object_size),
            init_pos=position(start_y, surface + cfg.object_size[2] * 0.5 + 0.002),
            init_rot=rotation,
            attrs=create_tutorial_rigid_body_physics(
                mass=0.03,
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                linear_damping=0.2,
                angular_damping=0.2,
            ),
        )
    )
    sim.prepare()
    for part, commands in cfg.commands.items():
        q = torch.tensor([commands["open"]], device=robot.device)
        robot.set_qpos(q, name=part, target=False)
        robot.set_qpos(q, name=part, target=True)
    # A deterministic above-object start keeps grasp selection comparable.
    for role, binding in (cfg.binding.items() if cfg.initialize_above_object else ()):
        pos = torch.tensor(
            position(
                (
                    (start_y + (0.12 if dual else 0.0))
                    if role != "destination"
                    else target_y
                ),
                surface + cfg.ready_clearance,
            ),
            device=robot.device,
        )
        ready = torch.eye(4, device=robot.device)
        ready_angle = angle + math.radians(cfg.ready_yaws.get(role, 0.0))
        ready_cosine, ready_sine = math.cos(ready_angle), math.sin(ready_angle)
        ready[:3, :3] = torch.tensor(
            [
                [ready_cosine, ready_sine, 0.0],
                [ready_sine, -ready_cosine, 0.0],
                [0.0, 0.0, -1.0],
            ],
            device=robot.device,
        )
        ready[:3, 3] = pos
        success, q = robot.compute_ik(
            pose=ready.unsqueeze(0),
            joint_seed=(
                torch.tensor([cfg.ready_seeds[binding["motion"]]], device=robot.device)
                if binding["motion"] in cfg.ready_seeds
                else robot.get_qpos(name=binding["motion"])
            ),
            name=binding["motion"],
        )
        if not bool(success.all()):
            raise RuntimeError(
                f"Unreachable ready pose for {binding['motion']}: {pos.tolist()}"
            )
        robot.set_qpos(q, name=binding["motion"], target=False)
        robot.set_qpos(q, name=binding["motion"], target=True)
    robot.clear_dynamics()
    sim.update(step=100)
    logger.log_info(
        "Embodiment: "
        + json.dumps(
            {
                "robot": robot.uid,
                "grippers": {
                    part: model.model_id for part, model in cfg.models.items()
                },
                "binding": cfg.binding,
                "tcp": {
                    p["motion"]: np.asarray(
                        robot.cfg.solver_cfg[p["motion"]].tcp
                    ).tolist()
                    for p in cfg.binding.values()
                },
                "commands": cfg.commands,
            }
        )
    )
    return robot, obj, pad


class MeasuredTaskCheck:
    """Read-only playback observations, separate from projected engine effects."""

    def __init__(
        self,
        cfg: EmbodimentCfg,
        robot: Robot,
        obj: RigidObject,
        compiled: CompiledTrajectory,
    ) -> None:
        self.cfg, self.robot, self.obj = cfg, robot, obj
        self.initial_z = float(obj.get_local_pose(to_matrix=True)[0, 2, 3])
        self.max_lift = 0.0
        self.pickup_lift = 0.0
        self.pickup_stop = (
            compiled.segment(
                0, "lift" if "primary" in cfg.binding else "pickup_transport"
            ).stop
            - 1
        )
        self.transfer_reference: torch.Tensor | None = None
        self.transfer_drift = float("inf")
        self.source_open = False
        self.release = self.retreat = None
        if "source" in cfg.binding:
            self.release = compiled.segment(0, "handover_release")
            self.retreat = compiled.segment(0, "source_retreat")

    def observe(self, step: int, count: int) -> None:
        """Observe the actual object and jaw state after each dispatched sample."""
        pose = self.obj.get_local_pose(to_matrix=True)
        self.max_lift = max(self.max_lift, float(pose[0, 2, 3]) - self.initial_z)
        if step == self.pickup_stop:
            self.pickup_lift = float(pose[0, 2, 3]) - self.initial_z
        if self.release is None or self.retreat is None:
            return
        destination = self.cfg.binding["destination"]["motion"]
        tcp = self.robot.compute_fk(
            qpos=self.robot.get_qpos(name=destination), name=destination, to_matrix=True
        )
        relative = torch.linalg.inv(tcp) @ pose
        if step == self.release.stop - 1:
            self.transfer_reference = relative.clone()
        if step == self.retreat.stop - 1 and self.transfer_reference is not None:
            self.transfer_drift = float(
                torch.linalg.vector_norm(
                    relative[0, :3, 3] - self.transfer_reference[0, :3, 3]
                )
            )
            part = self.cfg.binding["source"]["grasp"]
            opened = torch.tensor(
                self.cfg.commands[part]["open"], device=self.robot.device
            )
            self.source_open = bool(
                torch.allclose(
                    self.robot.get_qpos(name=part)[0], opened, atol=0.005, rtol=0
                )
            )
            if float(pose[0, 2, 3]) - self.initial_z < 0.04:
                self.transfer_drift = float("inf")

    def finish(self, sim: SimulationManager, target: torch.Tensor) -> None:
        """Require measured lift, stable placement, and receiver retention."""
        before = self.obj.get_local_pose(to_matrix=True).clone()
        sim.update(step=100)
        after = self.obj.get_local_pose(to_matrix=True)
        error = float(torch.linalg.vector_norm(after[0, :3, 3] - target[:3, 3]))
        drift = float(torch.linalg.vector_norm(after[0, :3, 3] - before[0, :3, 3]))
        transfer = self.release is None or (
            self.source_open and self.transfer_drift < 0.02
        )
        success = (
            self.pickup_lift > 0.04 and error < 0.03 and drift < 0.005 and transfer
        )
        result = {
            "planning_success": True,
            "playback_completed": True,
            "task_success": success,
            "lift_m": self.max_lift,
            "pickup_lift_m": self.pickup_lift,
            "placement_error_m": error,
            "settled_drift_m": drift,
            "object_xyz": after[0, :3, 3].tolist(),
            "target_xyz": target[:3, 3].tolist(),
        }
        if self.release is not None:
            result.update(
                source_open=self.source_open,
                receiver_drift_m=(
                    self.transfer_drift if math.isfinite(self.transfer_drift) else None
                ),
            )
        logger.log_info("Measured result: " + json.dumps(result))
        if not success:
            raise RuntimeError("Measured manipulation failed; see the result above")


__all__ = [
    "EmbodimentCfg",
    "MeasuredTaskCheck",
    "create_embodiment",
    "create_scene",
    "parse_arguments",
]
