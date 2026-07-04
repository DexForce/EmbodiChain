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

"""Demonstrate PickUp on an upright object with configurable approach."""

from __future__ import annotations

import argparse
import time

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    GraspTarget,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    format_tensor,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_ur5_gripper_robot_cfg,
    draw_axis_marker,
    get_tutorial_window_size,
)

GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040

OBJECT_MIN_HAND_CLOSE_QPOS = 0.024
OBJECT_XY = (-0.42, -0.08)

OBJECT_PRESETS = {
    "sugar_box": {
        "label": "sugar_box",
        "mesh_path": "SugarBox/sugar_box_usd/sugar_box.usda",
        "init_rot": (0.0, 0.0, 0.0),
        "body_scale": (0.8, 0.8, 0.8),
        "mass": 0.05,
        "use_usd_properties": False,
    },
}

PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240

APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


class PickUpDemo(DemoBase):
    """Demo that picks up an object using an antipodal grasp affordance."""

    def setup(self) -> None:
        """Create simulation, robot, object, motion generator and action engine."""
        width, height = get_tutorial_window_size(self.args)
        self.sim = create_default_sim(
            self.args,
            width=width,
            height=height,
            physics_dt=1.0 / 100.0,
            arena_space=2.5,
        )
        self.robot = self.sim.add_robot(
            cfg=create_ur5_gripper_robot_cfg(init_pos=(0.0, 0.0, 0.0))
        )
        self.obj = self._create_pick_object(self.args.object)

        motion_gen = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )

        hand_open, hand_close = self._get_hand_open_close_qpos()
        approach_direction = self._resolve_approach_direction()
        self._initialize_pre_pick_robot_pose(hand_open)
        pickup_cfg = PickUpCfg(
            control_part="arm",
            hand_control_part="hand",
            hand_open_qpos=hand_open,
            hand_close_qpos=hand_close,
            approach_direction=approach_direction,
            pre_grasp_distance=0.15,
            lift_height=0.16,
            sample_interval=PICK_SAMPLE_INTERVAL,
            hand_interp_steps=HAND_INTERP_STEPS,
        )

        self.atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
        self.atomic_engine.register(PickUp(motion_gen, cfg=pickup_cfg))

        self.semantics = self._create_object_semantics()
        self.approach_direction = approach_direction

        maybe_open_window(self.sim, self.args)
        if not self.args.no_vis_eef_axis:
            self._draw_pick_object_axis()

    def run(self) -> None:
        """Plan and replay the PickUp trajectory."""
        maybe_wait_for_user(
            self.args,
            f"Inspect the upright {self.args.object}, then press Enter to plan...",
        )

        logger.log_info(
            f"Planning pick_up for {self.args.object} with "
            f"approach_direction={format_tensor(self.approach_direction)}"
        )
        start_time = time.time()
        is_success, traj, _ = self.atomic_engine.run(
            steps=[("pick_up", GraspTarget(semantics=self.semantics))]
        )
        cost_time = time.time() - start_time
        logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
        if not is_success:
            logger.log_warning("Failed to plan pickup demo trajectory.")
            return

        maybe_wait_for_user(self.args, "Press Enter to replay the pickup demo...")

        with DemoRecording(
            self.sim, self.args, prefix=f"pickup_{self.args.object}_auto_play"
        ):
            self._replay_pickup_trajectory(traj)

        maybe_wait_for_user(self.args, "Press Enter to exit the simulation...")

    def _create_pick_object(self, object_name: str) -> RigidObject:
        preset = OBJECT_PRESETS[object_name]
        cfg = RigidObjectCfg(
            uid=preset["label"],
            shape=MeshCfg(fpath=get_data_path(preset["mesh_path"])),
            attrs=RigidBodyAttributesCfg(
                mass=preset["mass"],
                dynamic_friction=0.97,
                static_friction=0.99,
            ),
            max_convex_hull_num=16,
            init_pos=[OBJECT_XY[0], OBJECT_XY[1], 0.0],
            init_rot=preset["init_rot"],
            body_scale=preset["body_scale"],
            use_usd_properties=preset["use_usd_properties"],
        )
        obj = self.sim.add_rigid_object(cfg=cfg)

        # Settle the object to ensure it is resting on the ground before planning
        self.sim.update(step=10)
        return obj

    def _build_grasp_generator_cfg(self) -> GraspGeneratorCfg:
        return GraspGeneratorCfg(
            viser_port=11801,
            antipodal_sampler_cfg=AntipodalSamplerCfg(
                n_sample=self.args.n_sample,
                max_length=GRIPPER_MAX_OPEN_WIDTH,
                min_length=0.003,
            ),
            is_partial_annotate=False,
            is_filter_ground_collision=False,
        )

    def _build_gripper_collision_cfg(self) -> GripperCollisionCfg:
        return GripperCollisionCfg(
            max_open_length=GRIPPER_MAX_OPEN_WIDTH,
            finger_length=GRIPPER_FINGER_LENGTH,
            y_thickness=GRIPPER_Y_THICKNESS,
            root_z_width=GRIPPER_ROOT_Z_WIDTH,
            open_check_margin=0.002,
            point_sample_dense=0.012,
        )

    def _create_object_semantics(self) -> ObjectSemantics:
        label = OBJECT_PRESETS[self.args.object]["label"]
        return ObjectSemantics(
            label=label,
            geometry={
                "mesh_vertices": self.obj.get_vertices(env_ids=[0], scale=True)[0],
                "mesh_triangles": self.obj.get_triangles(env_ids=[0])[0],
            },
            affordance=AntipodalAffordance(
                mesh_vertices=self.obj.get_vertices(env_ids=[0], scale=True)[0],
                mesh_triangles=self.obj.get_triangles(env_ids=[0])[0],
                gripper_collision_cfg=self._build_gripper_collision_cfg(),
                generator_cfg=self._build_grasp_generator_cfg(),
                force_reannotate=self.args.force_reannotate,
            ),
            entity=self.obj,
        )

    def _get_hand_open_close_qpos(self) -> tuple[torch.Tensor, torch.Tensor]:
        hand_limits = self.robot.get_qpos_limits(name="hand")[0].to(
            device=self.sim.device, dtype=torch.float32
        )
        hand_open = hand_limits[:, 0]
        hand_close_limit = hand_limits[:, 1]
        hand_close = torch.minimum(
            hand_close_limit,
            torch.full_like(hand_close_limit, OBJECT_MIN_HAND_CLOSE_QPOS),
        )
        return hand_open, hand_close

    def _resolve_approach_direction(self) -> torch.Tensor:
        if self.args.approach == "custom":
            if self.args.custom_approach_direction is None:
                raise ValueError(
                    "--custom_approach_direction is required when --approach custom."
                )
            direction = self.args.custom_approach_direction
        else:
            direction = APPROACH_DIRECTIONS[self.args.approach]

        approach_direction = torch.tensor(
            direction, dtype=torch.float32, device=self.sim.device
        )
        norm = torch.linalg.norm(approach_direction)
        if norm < 1e-6:
            raise ValueError("approach_direction must be non-zero.")
        return approach_direction / norm

    def _make_pre_pick_eef_pose(self, position: torch.Tensor) -> torch.Tensor:
        pose = self.robot.compute_fk(
            qpos=self.robot.get_qpos(name="arm"),
            name="arm",
            to_matrix=True,
        ).clone()
        pose[:, :3, 3] = position
        return pose

    def _initialize_pre_pick_robot_pose(self, hand_open: torch.Tensor) -> None:
        obj_pose = self.obj.get_local_pose(to_matrix=True)
        move_position = obj_pose[:, :3, 3].clone()
        move_position[:, 2] = 0.36
        pre_pick_pose = self._make_pre_pick_eef_pose(move_position)
        ik_success, arm_qpos = self.robot.compute_ik(
            pose=pre_pick_pose,
            joint_seed=self.robot.get_qpos(name="arm"),
            name="arm",
        )
        if not torch.all(ik_success):
            raise RuntimeError("Failed to initialize the robot at the pre-pick pose.")

        n_envs = self.robot.get_qpos().shape[0]
        hand_qpos = hand_open.unsqueeze(0).repeat(n_envs, 1)
        for target in (False, True):
            self.robot.set_qpos(arm_qpos, name="arm", target=target)
            self.robot.set_qpos(hand_qpos, name="hand", target=target)
        self.robot.clear_dynamics()

    def _replay_pickup_trajectory(self, traj: torch.Tensor) -> None:
        post_grasp_clear_step = self._compute_pick_close_end_step()
        should_clear_object_dynamics = True
        for i in range(traj.shape[1]):
            self.robot.set_qpos(traj[:, i, :])
            self.sim.update(step=4)
            if should_clear_object_dynamics and i + 1 >= post_grasp_clear_step:
                self.obj.clear_dynamics()
                should_clear_object_dynamics = False
                logger.log_info(f"Object dynamics cleared after grasp at step={i}")
            time.sleep(1e-2)

        logger.log_info(
            f"PickUp keeps the upright {self.args.object} suspended in the gripper."
        )

        final_qpos = traj[:, -1, :]
        for _ in range(POST_TRAJECTORY_STEPS):
            self.robot.set_qpos(final_qpos)
            self.sim.update(step=2)
            time.sleep(1e-2)

    @staticmethod
    def _compute_pick_close_end_step() -> int:
        motion_waypoints = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
        n_approach = int(round(motion_waypoints) * 0.6)
        return n_approach + HAND_INTERP_STEPS

    def _draw_pick_object_axis(self) -> None:
        draw_axis_marker(
            self.sim,
            "pickup_object_axis",
            self.obj.get_local_pose(to_matrix=True),
        )


def main() -> None:
    """Entry point for the PickUp demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Demonstrate PickUp on an upright object."
    )
    parser = add_demo_args(parser)
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS.keys()),
        default="sugar_box",
        help="Object preset to pick.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )
    parser.add_argument(
        "--approach",
        choices=["top", "side", "side_y", "custom"],
        default="top",
        help="Pick approach direction preset.",
    )
    parser.add_argument(
        "--custom_approach_direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame approach direction used when --approach custom.",
    )
    args = parser.parse_args()
    PickUpDemo(args).main()


if __name__ == "__main__":
    main()
