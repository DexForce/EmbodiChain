#!/usr/bin/env python3
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

"""Functional atomic-actions tutorial.

This demo keeps the same UR10 + DH gripper + CoffeeCup scene as
``atomic_actions.py`` but uses the functional public API:
``pick_up`` -> ``place`` -> ``move`` -> ``gripper_close`` -> ``gripper_open``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from dexsim.utility import images_to_video

ROOT_DIR = Path(__file__).resolve().parents[3]
SIM_TUTORIAL_DIR = ROOT_DIR / "scripts/tutorials/sim"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SIM_TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_TUTORIAL_DIR))

from atomic_actions import create_mug, create_robot, initialize_simulation  # noqa: E402
from embodichain.lab.sim.atomic_actions import (  # noqa: E402
    AntipodalAffordance,
    ObjectSemantics,
    gripper_close,
    gripper_open,
    move,
    pick_up,
    place,
)
from embodichain.lab.sim.planners import (  # noqa: E402
    MotionGenCfg,
    MotionGenerator,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.sensors import Camera, CameraCfg  # noqa: E402
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (  # noqa: E402
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (  # noqa: E402
    GripperCollisionCfg,
)

FPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the atomic action tutorial through the functional API."
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--enable_rt", action="store_true")
    parser.add_argument("--open_window", action="store_true", default=False)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Defaults to outputs/atomic_actions_functional_YYYYMMDD_HHMM.",
    )
    parser.add_argument("--min_video_seconds", type=float, default=8.0)
    return parser.parse_args()


def timestamped_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return ROOT_DIR / "outputs" / f"atomic_actions_functional_{timestamp}"


def create_camera(sim) -> Camera:
    return sim.add_sensor(
        sensor_cfg=CameraCfg(
            uid="cam1",
            width=640,
            height=480,
            intrinsics=(600, 600, 320.0, 240.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(1.15, -0.75, 0.85),
                target=(0.45, 0.0, 0.12),
                up=(0.0, 0.0, 1.0),
            ),
            near=0.01,
            far=10.0,
            enable_color=True,
        )
    )


def capture_frame(camera: Camera):
    camera.update(fetch_only=camera.is_rt_enabled)
    rgb = camera.get_data()["color"][0, :, :, :3]
    return rgb.detach().cpu().numpy()


def apply_trajectory(robot, sim, camera, trajectory, joint_ids, frames: list) -> None:
    for i in range(trajectory.shape[1]):
        qpos = robot.get_qpos().clone()
        qpos[:, joint_ids] = trajectory[:, i]
        robot.set_qpos(qpos)
        sim.update(step=4)
        frames.append(capture_frame(camera))
        time.sleep(1e-3)


def hold(sim, camera, frames: list, seconds: float) -> None:
    for _ in range(max(1, int(round(seconds * FPS)))):
        sim.update(step=1)
        frames.append(capture_frame(camera))


def build_mug_semantics(mug) -> ObjectSemantics:
    mug_grasp_affordance = AntipodalAffordance(
        object_label="mug",
        force_reannotate=False,
        custom_config={
            "gripper_collision_cfg": GripperCollisionCfg(
                max_open_length=0.088,
                finger_length=0.078,
                point_sample_dense=0.012,
            ),
            "generator_cfg": GraspGeneratorCfg(
                viser_port=11801,
                antipodal_sampler_cfg=AntipodalSamplerCfg(
                    n_sample=20000,
                    max_length=0.088,
                    min_length=0.003,
                ),
            ),
        },
    )
    return ObjectSemantics(
        label="mug",
        geometry={
            "mesh_vertices": mug.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": mug.get_triangles(env_ids=[0])[0],
        },
        affordance=mug_grasp_affordance,
        entity=mug,
    )


def main() -> None:
    args = parse_args()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else timestamped_output_root()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    video_dir = output_root / "outputs/videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    mug = create_mug(sim)
    camera = create_camera(sim)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    sim.init_gpu_physics()
    if args.open_window:
        sim.open_window()

    hand_open = torch.tensor([0.00, 0.00], dtype=torch.float32, device=sim.device)
    hand_close = torch.tensor([0.025, 0.025], dtype=torch.float32, device=sim.device)
    mug_semantics = build_mug_semantics(mug)

    place_xpos = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022, 0.2489],
            [-0.9977, 0.0540, -0.0401, 0.3970],
            [0.0401, 0.0000, -0.9992, 0.2400],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    rest_xpos = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022, 0.5000],
            [-0.9977, 0.0540, -0.0401, 0.0000],
            [0.0401, 0.0000, -0.9992, 0.5000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=sim.device,
    )

    rows: list[dict[str, str]] = []
    frames: list = []
    try:
        hold(sim, camera, frames, 0.5)
        steps = [
            (
                "pick_up",
                lambda: pick_up(
                    motion_generator=motion_gen,
                    target=mug_semantics,
                    start_qpos=robot.get_qpos(name="arm"),
                    hand_open_qpos=hand_open,
                    hand_close_qpos=hand_close,
                    pre_grasp_distance=0.15,
                    lift_height=0.15,
                    approach_direction=torch.tensor(
                        [0.0, 0.0, -1.0],
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    sample_interval=80,
                ),
            ),
            (
                "place",
                lambda: place(
                    motion_generator=motion_gen,
                    target=place_xpos,
                    start_qpos=robot.get_qpos(name="arm"),
                    hand_open_qpos=hand_open,
                    hand_close_qpos=hand_close,
                    lift_height=0.15,
                    sample_interval=80,
                ),
            ),
            (
                "move",
                lambda: move(
                    motion_generator=motion_gen,
                    target=rest_xpos,
                    start_qpos=robot.get_qpos(name="arm"),
                    sample_interval=50,
                ),
            ),
            (
                "gripper_close",
                lambda: gripper_close(
                    motion_generator=motion_gen,
                    close_qpos=hand_close,
                    start_qpos=robot.get_qpos(name="hand"),
                    sample_interval=15,
                ),
            ),
            (
                "gripper_open",
                lambda: gripper_open(
                    motion_generator=motion_gen,
                    open_qpos=hand_open,
                    start_qpos=robot.get_qpos(name="hand"),
                    sample_interval=15,
                ),
            ),
        ]
        for step_name, step_fn in steps:
            is_success, trajectory, joint_ids = step_fn()
            rows.append(
                {
                    "step": step_name,
                    "success": str(bool(is_success)),
                    "waypoints": str(trajectory.shape[1] if trajectory.ndim == 3 else 0),
                    "joint_ids": ",".join(str(joint_id) for joint_id in joint_ids),
                }
            )
            if not is_success:
                raise RuntimeError(f"Functional atomic action failed: {step_name}")
            apply_trajectory(robot, sim, camera, trajectory, joint_ids, frames)

        target_frames = max(1, int(round(args.min_video_seconds * FPS)))
        if len(frames) < target_frames:
            hold(sim, camera, frames, (target_frames - len(frames)) / FPS)
        images_to_video(frames, str(video_dir), "episode_0_cam1", fps=FPS)
    finally:
        sim.destroy()

    with (output_root / "summary.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "success", "waypoints", "joint_ids"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Functional atomic actions demo finished: {output_root}")


if __name__ == "__main__":
    main()
