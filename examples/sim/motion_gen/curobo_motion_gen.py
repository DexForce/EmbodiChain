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

"""cuRobo V2 collision-aware motion-generation demo.

Builds a single-arm Franka Panda with a static cuboid obstacle (mirrored in
both DexSim and the cuRobo collision world), plans a collision-free
end-effector move through the EmbodiChain ``MotionGenerator`` API, and replays
the returned full-DoF trajectory in the simulator.

Requirements: an NVIDIA CUDA device and cuRobo V2 installed with matching
CUDA/PyTorch extras. See:
https://nvlabs.github.io/curobo/latest/getting-started/installation.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from embodichain import data as _data
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    PlanState,
)
from embodichain.lab.sim.planners.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlannerCfg,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
)

ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.40, 0.36]
DEMO_BLOCK_POS = [0.45, 0.0, 0.18]
CUROBO_INSTALL_URL = (
    "https://nvlabs.github.io/curobo/latest/getting-started/installation.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cuRobo V2 motion-gen demo")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the viewer window.",
    )
    parser.add_argument(
        "--step-repeat",
        type=int,
        default=4,
        help="Simulation updates per planned waypoint during playback.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=20,
        help="Simulation updates to hold before and after playback.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip cuRobo planner warmup.",
    )
    return parser.parse_args()


def _check_runtime() -> None:
    """Fail fast with actionable guidance when CUDA/cuRobo are unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuRobo V2 requires a CUDA-capable NVIDIA GPU, but none is "
            "available. This demo cannot run on CPU."
        )
    try:
        import curobo  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cuRobo V2 is not installed. Install it with NVIDIA's CUDA-matched "
            "extras, e.g. `pip install .[cu12]` or `pip install .[cu13]` "
            f"(also `.[cu12-torch]` / `.[cu13-torch]`). See {CUROBO_INSTALL_URL}."
        ) from exc


def _demo_world_path() -> str:
    return str(
        Path(_data.__file__).parent
        / "assets"
        / "curobo"
        / "collision_franka_demo.yml"
    )


def _franka_profile() -> CuroboRobotProfileCfg:
    """Explicit sim->cuRobo joint mapping; no index order is assumed."""
    sim_to_curobo = {f"fr3_joint{i}": f"panda_joint{i}" for i in range(1, 8)}
    return CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names=sim_to_curobo,
        fixed_joint_positions={
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        },
        base_link_name="panda_link0",
        tool_frame_name="panda_hand",
    )


def _build_scene(args: argparse.Namespace):
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            sim_device="cuda",
            num_envs=1,
            arena_space=2.0,
        )
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    # Mirror the cuRobo cuboid in DexSim so planner and simulator agree.
    sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=DEMO_BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=DEMO_BLOCK_POS,
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    return sim, robot


def _target_beyond_block(robot) -> torch.Tensor:
    """An end-effector target that requires routing around the cuboid."""
    qpos = robot.get_qpos(name=CONTROL_PART)
    fk = robot.compute_fk(qpos=qpos, name=CONTROL_PART, to_matrix=True)
    target = fk[0].clone()
    target[:3, 3] = torch.tensor([0.55, 0.20, 0.30], device=robot.device)
    return target


def _play(sim, robot, trajectory: torch.Tensor, step_repeat: int) -> None:
    all_ids = list(range(robot.dof))
    for w in range(trajectory.shape[1]):
        robot.set_qpos(qpos=trajectory[:, w], joint_ids=all_ids)
        sim.update(step=step_repeat)


def main() -> None:
    args = parse_args()
    _check_runtime()

    sim, robot = _build_scene(args)
    if not args.headless:
        sim.open_window()
    sim.update(step=args.hold_steps)

    cfg = CuroboPlannerCfg(
        robot_uid=ROBOT_UID,
        robot_profiles={CONTROL_PART: _franka_profile()},
        world=CuroboWorldCfg(world_config_path=_demo_world_path()),
        warmup=not args.no_warmup,
    )
    mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))

    start_qpos = robot.get_qpos(name=CONTROL_PART)
    target = _target_beyond_block(robot)

    result = mg.generate(
        [PlanState.from_xpos(target.unsqueeze(0))],
        MotionGenOptions(
            start_qpos=start_qpos,
            control_part=CONTROL_PART,
            plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
        ),
    )

    print(f"cuRobo success: {bool(result.success.item())}")
    print(f"positions shape: {tuple(result.positions.shape)}")
    print(f"duration: {float(result.duration[0]):.3f}s")

    if not bool(result.success.item()):
        sim.destroy()
        raise RuntimeError("cuRobo failed to find a collision-free trajectory.")

    # Replay the full-DoF trajectory.
    _play(sim, robot, result.positions, step_repeat=max(args.step_repeat, 1))
    sim.update(step=args.hold_steps)

    final_q = result.positions[0:1, -1, :].to(robot.device)
    fk = robot.compute_fk(qpos=final_q, name=CONTROL_PART, to_matrix=True)
    err = float(torch.norm(fk[0, :3, 3] - target[:3, 3]))
    print(f"final Cartesian position error: {err:.4f} m")

    sim.destroy()


if __name__ == "__main__":
    main()
