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

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest


def _probe(source: str = "task_program") -> None:
    import torch
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config
    from embodichain.lab.expansion.program import (
        collect_program_batch,
        collect_vector_program,
    )

    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    if source == "task_program_newton":
        path = root / "examples/expansion/engine/task.newton.yaml"
    if source == "real2sim":
        path = Path(os.environ["EXPANSION_REAL2SIM_DEPLOYMENT"])
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 2
    cfg.seed = 0
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.light.direct = []
    cfg.light.indirect = None
    cfg.sensor = []
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    if source.startswith("task_program"):
        cfg.task_program.program.count = 1
    env = None
    try:
        env = REGISTERED_ENVS[load_config(path)["id"]].make(cfg=cfg)
        env.reset(seed=0)
        if source == "task_program":
            from embodichain.lab.expansion.scene import apply_joint_scene_candidates
            from embodichain.lab.sim.scene import SceneView
            from embodichain.lab.sim.scene.feasibility import (
                JointSceneCandidate,
                RobotDeploymentCandidate,
            )
            from embodichain.lab.sim.scene.expansion import (
                SceneVariant,
                ScenePoseChange,
            )

            old_fk = env.robot.compute_batch_fk(
                qpos=env.robot.get_qpos(name="arm").unsqueeze(1),
                name="arm",
                to_matrix=True,
            ).clone()
            proposals = {}
            for row, dx in enumerate((0.01, -0.01)):
                root_pose = env.robot.get_local_pose(to_matrix=True)[row].clone()
                cube_pose = (
                    env.sim.get_rigid_object("cube")
                    .get_local_pose(to_matrix=True)[row]
                    .clone()
                )
                root_pose[0, 3] += dx
                cube_pose[0, 3] += dx
                proposals[row] = JointSceneCandidate(
                    "joint-request",
                    SceneVariant(
                        "scene", 0, row, (ScenePoseChange("cube", cube_pose.tolist()),)
                    ),
                    RobotDeploymentCandidate({env.robot.uid: root_pose.tolist()}),
                    "runtime",
                    (),
                )
            apply_joint_scene_candidates(SceneView(env.sim, env_ids=(0, 1)), proposals)
            new_fk = env.robot.compute_batch_fk(
                qpos=env.robot.get_qpos(name="arm").unsqueeze(1),
                name="arm",
                to_matrix=True,
            )
            torch.testing.assert_close(
                new_fk[:, 0, 0, 3] - old_fk[:, 0, 0, 3],
                torch.tensor([0.01, -0.01]),
                atol=1e-5,
                rtol=0,
            )
        program = env.compile_task_program(cfg.task_program)
        rows = None
        if source == "real2sim":
            import tempfile
            from embodichain.lab.expansion.real2sim import _Real2SimRow
            from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                load_real2sim_tabletop_profile,
            )

            profile = load_real2sim_tabletop_profile(
                os.environ["EXPANSION_REAL2SIM_PROFILE"]
            )
            rows = {
                row: _Real2SimRow(
                    env,
                    row,
                    profile,
                    output=Path(tempfile.mkdtemp(prefix="expansion-source-")),
                    ordinal=row,
                    attempt=0,
                    initial_state_id=f"initial-{row}",
                )
                for row in (0, 1)
            }
            bridges = {row: service.bridge for row, service in rows.items()}
        else:
            bridges = {
                row: env.task_program_adapter.for_env_ids((row,)).create_bridge(program)
                for row in (0, 1)
            }
        object_uid = profile.object_entity_id if source == "real2sim" else "cube"
        positions = [[env.robot.get_qpos()[row].cpu().clone()] for row in (0, 1)]
        objects = [
            [env.sim.get_rigid_object(object_uid).get_local_pose()[row].cpu().clone()]
            for row in (0, 1)
        ]
        frames = [0, 0]

        def observe(ids, action, obs):
            command = (
                action.value
                if isinstance(action.value, torch.Tensor)
                else action.value["qpos"]
            )
            assert command.shape[0] == 2
            for row in ids:
                frames[row] += 1
                positions[row].append(env.robot.get_qpos()[row].cpu().clone())
                objects[row].append(
                    env.sim.get_rigid_object(object_uid)
                    .get_local_pose()[row]
                    .cpu()
                    .clone()
                )

        if source == "task_program_newton":
            # The native Task Program host executes the canonical vector runtime.
            result = collect_vector_program(
                env, program, (0, 1), max_steps=2500, observe_step=observe
            )
        else:
            result = collect_program_batch(
                env,
                bridges,
                max_steps=2500,
                observe_step=observe,
                reject_row=(
                    (lambda row: row == 0 and frames[row] >= 10)
                    if source == "task_program_abort"
                    else None
                ),
            )
        assert result.accepted == (
            (False, True) if source == "task_program_abort" else (True, True)
        ), result
        if source == "task_program_abort":
            assert frames[0] == 11 and frames[1] > frames[0]
        assert tuple(frames) == result.lengths
        if rows is not None:
            for row, service in rows.items():
                request = service.qualify(
                    qpos=torch.stack(positions[row]),
                    object_pose=torch.stack(objects[row]),
                )
                assert request.validation.accepted
        if source.startswith("task_program"):
            from dataclasses import replace
            from embodichain.lab.sim.scene import (
                SceneView,
                capture_state,
                restore_render_state,
            )

            view = SceneView(env.sim, env_ids=(0, 1))
            from embodichain.lab.sim.scene.physics import capture_physics_properties
            from embodichain.lab.expansion.parameters import _apply_auxiliary_physics

            parameters = capture_physics_properties(view)
            first_joint = env.robot.joint_names[0]
            _apply_auxiliary_physics(
                view,
                parameters,
                robot_uid=env.robot.uid,
                com_offsets={"cube": torch.tensor([[0.001, 0, 0], [-0.001, 0, 0]])},
                stiffness_scale={first_joint: torch.tensor([0.9, 1.1])},
                damping_scale={first_joint: torch.tensor([0.8, 1.2])},
            )
            changed = capture_physics_properties(view)
            torch.testing.assert_close(
                changed["cube.com_pose"][:, 0] - parameters["cube.com_pose"][:, 0],
                torch.tensor([0.001, -0.001]),
                atol=1e-7,
                rtol=0,
            )
            _apply_auxiliary_physics(
                view,
                parameters,
                robot_uid=env.robot.uid,
                com_offsets={"cube": torch.zeros(2, 3)},
                stiffness_scale={first_joint: torch.ones(2)},
                damping_scale={first_joint: torch.ones(2)},
            )
            restored = capture_physics_properties(view)
            for key, value in parameters.items():
                torch.testing.assert_close(restored[key], value)
            before = capture_state(view)
            limits = env.robot.get_qpos_limits().clone()
            batches = []
            for row in before.batches:
                entities = []
                for entity in row:
                    if entity.kind == "robot":
                        positions = list(entity.qpos)
                        positions[-1] = float(limits[0, -1, 0]) - 1e-3
                        entity = replace(entity, qpos=tuple(positions))
                    entities.append(entity)
                batches.append(tuple(entities))
            measured = replace(before, batches=tuple(batches))
            clock = env.sim._visualization_sim_step
            env.sim.enable_physics(False)
            restore_render_state(view, measured)
            actual = capture_state(view)
            for expected, observed in zip(measured.batches, actual.batches):
                for a, b in zip(expected, observed):
                    torch.testing.assert_close(
                        torch.tensor(a.qpos), torch.tensor(b.qpos), atol=1e-6, rtol=0
                    )
            torch.testing.assert_close(
                env.robot.get_qpos_limits(), limits, atol=0, rtol=0
            )
            restore_render_state(view, before)
            assert env.sim._visualization_sim_step == clock
            env.sim.enable_physics(True)
        print("qualified:", result.accepted, result.lengths, flush=True)
    finally:
        if env is not None:
            env.close(exit_process=False)
        env = None
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "source", ["task_program", "task_program_abort", "task_program_newton", "real2sim"]
)
def test_native_task_program_two_rows_one_world(source):
    if source == "real2sim" and not all(
        os.environ.get(key)
        for key in ("EXPANSION_REAL2SIM_DEPLOYMENT", "EXPANSION_REAL2SIM_PROFILE")
    ):
        pytest.skip(
            "prepare a pinned Real2Sim fixture and set EXPANSION_REAL2SIM_DEPLOYMENT/PROFILE"
        )
    root = Path(__file__).resolve().parents[3]
    command = "import os,runpy,sys;runpy.run_path(sys.argv[1])['_probe'](sys.argv[2]);sys.stdout.flush();sys.stderr.flush();os._exit(0)"
    result = subprocess.run(
        [sys.executable, "-c", command, str(Path(__file__).resolve()), source],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(root),
                    str(root / "extensions"),
                    str(root / "extensions/embodichain_tasks"),
                )
            ),
            "EMBODICHAIN_SIM_EXIT_PROCESS": "0",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
