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

"""Tracking objectives, action mapping and native G1 reset behavior."""

from __future__ import annotations
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from embodichain.compute.kinematics import TreeKinematics
from embodichain.data_pipeline.datasets.motion import MotionLibrary
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg
from embodichain_tasks.locomotion.managers.actions import ReferenceJointPositionAction
from embodichain_tasks.locomotion.tracking._task import MotionTrackingTask


@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.requires_tasks
@pytest.mark.parametrize("update_data_interval", [0, 1])
def test_g1_state_injection_and_selected_reset(update_data_interval: int) -> None:
    """Written starts survive native stepping and preserve other arenas/materials."""
    data_root = os.environ.get("EMBODICHAIN_DATA_ROOT")
    if not data_root or not (Path(data_root) / "LAFAN1G1/motion.npz").is_file():
        pytest.skip("Prepare LAFAN1G1 under EMBODICHAIN_DATA_ROOT to run G1 physics")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(update_data_interval)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(__file__).resolve().parents[4],
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_g1_state_injection_and_selected_reset(update_data_interval: int) -> None:
    """Create the NoRender world in a process with no preinitialized renderer."""
    import numpy as np
    import yaml

    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim import SimulationManager
    from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv

    data_root = os.environ.get("EMBODICHAIN_DATA_ROOT")
    archive = Path(data_root or "") / "LAFAN1G1/motion.npz"
    if not data_root or not archive.is_file():
        pytest.skip("Prepare LAFAN1G1 under EMBODICHAIN_DATA_ROOT to run G1 physics")
    root = Path(__file__).resolve().parents[4]
    config_path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/locomotion/tracking/g1_motion/env.newton.yaml"
    )
    gym = yaml.safe_load(config_path.read_text())
    gym["num_envs"] = 2
    gym["max_episode_steps"] = 2
    gym["physics_config"]["solver_cfg"]["update_data_interval"] = update_data_interval
    options = gym["env"]["extensions"]["motion_tracking"]
    with np.load(archive) as data:
        options["clips"] = data["names"].tolist()[:2]
    options.update(
        observation_noise=False, random_start=False, fixed_clip_assignment=True
    )
    cfg = config_to_cfg(gym)
    cfg.init_rollout_buffer = False
    env = UnitreeG1MotionEnv(cfg)
    try:
        env.reset(seed=10000)
        env.sim.update(step=1, render_final_step=False)
        assert env.sim.physics.cuda_graph_status == "captured"
        ids = torch.tensor([1], device=env.device)
        floor = env.sim.get_rigid_object("floor")
        floor.set_friction(torch.tensor([0.73], device=env.device), env_ids=ids)
        com = env.robot.get_com_pose(["torso_link"])[ids].clone()
        com[:, :, 0] += 0.005
        env.robot.set_com_pose(com, link_names=["torso_link"], env_ids=ids)
        env.sim.update(step=1, render_final_step=False)
        friction_before = floor.get_friction().clone()
        com_before = env.robot.get_com_pose(["torso_link"]).clone()
        before = env.read_state()
        injected = {key: value[ids].clone() for key, value in before.items()}
        injected["pos"][:, 0] += 1.5
        injected["pos"][:, 2] += 0.4
        injected["v"].zero_()
        injected["w"].zero_()
        injected["qd"].zero_()
        env.write_state(ids, injected)
        after = env.read_state()
        for name in before:
            torch.testing.assert_close(
                after[name][[0]], before[name][[0]], rtol=0, atol=0
            )
        torch.testing.assert_close(
            floor.get_friction(), friction_before, rtol=0, atol=0
        )
        torch.testing.assert_close(
            env.robot.get_com_pose(["torso_link"]), com_before, rtol=0, atol=0
        )
        assert env.sim.physics.cuda_graph_status == "captured"
        env.sim.update(step=1, render_final_step=False)
        # A stale native start returns to the old root location, 1.5 m away.
        torch.testing.assert_close(
            env.read_state()["pos"][ids], injected["pos"], rtol=0, atol=0.05
        )
        assert env.sim.physics.cuda_graph_status == "captured"
        before_reset = env.read_state()
        env.reset(options={"reset_ids": ids})
        after_reset = env.read_state()
        torch.testing.assert_close(
            after_reset["pos"][ids],
            env.motion.frame(env.motion.starts[ids])["pos"],
            rtol=0,
            atol=2e-6,
        )
        for name in before_reset:
            torch.testing.assert_close(
                after_reset[name][[0]], before_reset[name][[0]], rtol=0, atol=0
            )
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(
                torch.zeros(2, env.task.dof, device=env.device)
            )
        assert bool((terminated | truncated).all())
        torch.testing.assert_close(env.motion_frame, env.motion.starts, rtol=0, atol=0)
        torch.testing.assert_close(
            env.read_state()["pos"],
            env.motion.frame(env.motion.starts)["pos"],
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            floor.get_friction(), friction_before, rtol=0, atol=0
        )
        torch.testing.assert_close(
            env.robot.get_com_pose(["torso_link"]), com_before, rtol=0, atol=0
        )
    finally:
        floor = None
        env.close(exit_process=False)
        del env
        SimulationManager.flush_cleanup_queue()


def task_and_state():
    urdf = """<robot name="one"><link name="root"/><link name="tip"/>
    <joint name="joint" type="revolute"><parent link="root"/><child link="tip"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint></robot>"""
    kin = TreeKinematics(urdf, ["joint"], dtype=torch.float64)
    scalar = torch.zeros(1, dtype=torch.float64)
    task = MotionTrackingTask(
        kin,
        scalar,
        torch.ones_like(scalar) * 0.5,
        torch.tensor([[-1.0, 1.0]], dtype=scalar.dtype),
        torch.ones_like(scalar),
        ["root", "tip"],
    )
    state = {
        "pos": torch.tensor([[0.0, 0.0, 1.0]], dtype=scalar.dtype),
        "rot": torch.eye(3, dtype=scalar.dtype)[None],
        "q": scalar.reshape(1, 1),
        "qd": scalar.reshape(1, 1),
        "v": torch.zeros(1, 3, dtype=scalar.dtype),
        "w": torch.zeros(1, 3, dtype=scalar.dtype),
    }
    return task, state


def test_reference_reward_and_gradient_point_toward_target():
    task, ref = task_and_state()
    action = torch.zeros(1, 1, dtype=torch.float64)
    reward, _ = task.reward(ref, ref, action, action)
    torch.testing.assert_close(reward, torch.tensor([0.1], dtype=reward.dtype))
    state = {k: v.clone() for k, v in ref.items()}
    state["pos"][:, 2] += 0.1
    state["pos"].requires_grad_()
    state["q"] += 0.2
    state["q"].requires_grad_()
    reward, _ = task.reward(state, ref, action, action)
    gradient = torch.autograd.grad(reward.sum(), (state["pos"], state["q"]))
    assert gradient[0][0, 2] < 0 and gradient[1][0, 0] < 0
    assert all(torch.isfinite(g).all() for g in gradient)
    state["pos"] = state["pos"].detach()
    state["pos"][:, 2] = 1.4
    hard, continuation = task.termination(state, ref)
    assert hard.item() and continuation.item() < 0.001


def test_reference_residual_bias_and_robot_relative_goal():
    task, ref = task_and_state()
    action = torch.tensor([[0.4]], dtype=torch.float64)
    normalized, target = task.control(
        action,
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.05]], dtype=torch.float64),
    )
    torch.testing.assert_close(normalized, torch.ones_like(action))
    torch.testing.assert_close(target, torch.full_like(action, 0.45))
    future = {
        k: v.unsqueeze(1).expand(-1, 10, *v.shape[1:]).clone() for k, v in ref.items()
    }
    future["pos"][..., 0] += 2.0
    goal = task.goal(ref, future).unflatten(-1, (10, -1))
    torch.testing.assert_close(
        goal[..., :3],
        torch.tensor([[[2.0, 0.0, 0.0]]], dtype=action.dtype).expand(1, 10, 3),
    )


def test_tracking_diagnostics_distinguish_alignment_and_failure_conditions():
    task, ref = task_and_state()
    state = {k: v.clone() for k, v in ref.items()}
    state["pos"] += torch.tensor([[2.0, 0.0, 0.3]], dtype=state["pos"].dtype)
    state["q"] += math.pi / 2
    errors = task.tracking_errors(state, ref)
    # Horizontal translation is removed only from the body-relative pose error.
    torch.testing.assert_close(
        errors["body_pos"], torch.tensor([0.09], dtype=torch.float64)
    )
    torch.testing.assert_close(
        errors["root_pos"], torch.tensor([4.09], dtype=torch.float64)
    )
    torch.testing.assert_close(
        errors["body_ori"], torch.tensor([math.pi**2 / 8], dtype=torch.float64)
    )
    reasons = task.failure_reasons(state, ref)
    assert reasons["height"].item() and not reasons["tilt"].item()
    state["pos"] = ref["pos"].clone()
    state["rot"] = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=torch.float64
    )
    reasons = task.failure_reasons(state, ref)
    assert not reasons["height"].item() and reasons["tilt"].item()
    failed, _ = task.termination(state, ref)
    assert torch.equal(failed, reasons["height"] | reasons["tilt"])

    # Equal headings with different root tilts must not introduce a yaw offset.
    from embodichain_models.dynamics.rwm.robot_state import rotation_exp

    task, ref = task_and_state()
    state = {k: v.clone() for k, v in ref.items()}
    state["rot"] = rotation_exp(
        torch.tensor([[math.pi / 4, 0.0, 0.0]], dtype=torch.float64)
    )
    ref["rot"] = rotation_exp(
        torch.tensor([[0.0, math.pi / 4, 0.0]], dtype=torch.float64)
    )
    state["pos"].requires_grad_()
    error = task.tracking_errors(state, ref)["body_pos"]
    torch.testing.assert_close(
        error, torch.tensor([1 - math.cos(math.pi / 4)], dtype=torch.float64)
    )
    gradient = torch.autograd.grad(error.sum(), state["pos"])[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[0.0, 0.0, math.sin(math.pi / 4)]], dtype=torch.float64)
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_rotation_error_has_correct_values_and_gradients_near_zero_and_pi(compiled):
    from embodichain_models.dynamics.rwm.robot_state import rotation_exp

    task, original = task_and_state()
    angles = torch.tensor(
        [0.0, 1e-9, -1e-6, 0.3, math.pi - 1e-4, -math.pi + 1e-4],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = {
        name: value.expand(len(angles), *value.shape[1:])
        for name, value in original.items()
    }
    axis = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    axis = axis / axis.norm()
    rotation = rotation_exp(angles[:, None] * axis)

    def error(rotation):
        state = {**reference, "rot": rotation}
        return task.tracking_errors(state, reference)["root_ori"]

    fn = (
        torch.compile(error, backend="aot_eager", fullgraph=True) if compiled else error
    )
    values = fn(rotation)
    torch.testing.assert_close(values, angles.square(), atol=1e-12, rtol=1e-10)
    (gradient,) = torch.autograd.grad(values.sum(), (angles,))
    torch.testing.assert_close(gradient, 2 * angles.detach(), atol=1e-10, rtol=1e-10)
    half_turn = (
        2 * axis[:, None] * axis[None, :] - torch.eye(3, dtype=axis.dtype)
    ).expand(len(angles), -1, -1)
    torch.testing.assert_close(
        fn(half_turn), torch.full_like(angles, math.pi**2), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("future_offsets", [(1, 2), (5, 10)])
def test_control_uses_next_frame_independent_of_goal_offsets(future_offsets):
    task, _ = task_and_state()
    task.future_offsets = torch.tensor(future_offsets)
    qpos = torch.zeros(8, 8)
    qpos[:, 6] = 1
    qpos[:, 7] = torch.arange(8)
    motion = MotionLibrary(qpos, torch.zeros(8, 7), [8], ["move"], ["joint"], 50)
    env = SimpleNamespace(
        task=task,
        motion=motion,
        motion_frame=torch.tensor([0, 7]),
        num_envs=2,
        device=torch.device("cpu"),
        robot=SimpleNamespace(
            joint_names=["joint"], cfg=SimpleNamespace(init_qpos=[0.0]), set_qpos=Mock()
        ),
    )
    action = ReferenceJointPositionAction(
        ActionTermCfg(params={"joint_names": ["joint"]}), env
    )
    action.process_actions(torch.zeros(2, 1))
    torch.testing.assert_close(action.processed_actions, torch.tensor([[1.0], [7.0]]))
    torch.testing.assert_close(action.setpoint_actions, torch.tensor([[2.0], [14.0]]))
    action.apply_actions()
    call = env.robot.set_qpos.call_args
    torch.testing.assert_close(call.args[0], torch.tensor([[1.0], [7.0]]))
    assert call.kwargs["joint_ids"] == [0]
    assert call.kwargs["clamp_target_qpos"] is False


if __name__ == "__main__":
    _run_g1_state_injection_and_selected_reset(int(sys.argv[1]))
