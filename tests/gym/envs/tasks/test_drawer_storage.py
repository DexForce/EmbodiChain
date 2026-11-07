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

"""Measured containment, release and registration of the drawer storage task."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain_tasks.manipulation.drawer_storage import (
    DrawerStorageEnv,
    _FrankaDrawerStorageEnv,
)


def _observed_task(
    env_cls=DrawerStorageEnv,
    hand_joints=("gripper_finger1_joint_1",),
    open_hand=(0.0,),
    finger_links=("gripper_finger1_link_1", "gripper_finger2_link_1"),
):
    n = 7
    tray = torch.eye(4).repeat(n, 1, 1)
    tray[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tray[:, :3, 3] = torch.tensor([-1.0, 0.0, 0.445])
    local = torch.tensor([[0.0, 0.105, -0.032]]).repeat(n, 1)
    local[1, 2] += 0.3  # XY-aligned floating cube must fail.
    local[2, 0] = 0.19  # Centre inside, but a corner protrudes through the side.
    cube = torch.eye(4).repeat(n, 1, 1)
    cube[:, :3, 3] = torch.einsum("bij,bj->bi", tray[:, :3, :3], local) + tray[:, :3, 3]
    rail = torch.zeros(n, 1)
    rail[3] = 0.1
    hand = torch.tensor(open_hand).repeat(n, 1)
    hand[4, -1] += 0.01  # A single unopen joint must fail.
    finger = cube.clone()
    finger[:, 0, 3] += 0.2
    second_finger = finger.clone()
    second_finger[5, 0, 3] -= 0.18
    second_finger[6, 0, 3] = torch.inf
    fingers = dict(zip(finger_links, (finger, second_finger), strict=True))
    obj = SimpleNamespace(get_local_pose=lambda **kw: cube)
    drawer = SimpleNamespace(
        joint_names=["cabinet_to_drawer"],
        get_qpos=lambda **kw: rail,
        get_link_pose=lambda *a, **kw: tray,
    )
    env = object.__new__(env_cls)
    env._num_envs = n
    env.cfg = SimpleNamespace(task_program=True)
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True, completion_mask=torch.ones(n, dtype=torch.bool)
    )
    env.sim = SimpleNamespace(
        device=torch.device("cpu"),
        get_rigid_object=lambda uid: obj,
        get_articulation=lambda uid: drawer,
    )
    env.robot = SimpleNamespace(
        joint_names=hand_joints,
        get_qpos=lambda **kw: hand,
        get_link_pose=lambda name, **kw: fingers[name],
    )
    return env


@pytest.mark.parametrize(
    ("env_cls", "hand_joints", "open_hand", "finger_links"),
    [
        (
            DrawerStorageEnv,
            ("gripper_finger1_joint_1",),
            (0.0,),
            ("gripper_finger1_link_1", "gripper_finger2_link_1"),
        ),
        (
            _FrankaDrawerStorageEnv,
            ("fr3_finger_joint1", "fr3_finger_joint2"),
            (0.04, 0.04),
            ("fr3_leftfinger", "fr3_rightfinger"),
        ),
    ],
    ids=["ur5", "franka"],
)
def test_storage_requires_corners_closed_rail_open_hand_and_clearance(
    env_cls, hand_joints, open_hand, finger_links
):
    env = _observed_task(env_cls, hand_joints, open_hand, finger_links)
    assert env.is_task_success().tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_physical_storage_does_not_bypass_program_failure():
    env = _observed_task()
    env._active_task_program_bridge.program_completed = False
    assert not env.is_task_success().any()


def test_configured_deployment_preserves_its_explicit_acceptance_class():
    from pathlib import Path
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import get_env_spec
    from embodichain.utils.utility import load_config

    config = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage/task.ur5.yaml"
    )
    raw = load_config(config)
    cfg = config_to_cfg(raw, source_path=config)
    from argparse import Namespace
    from embodichain.lab.scripts.run_task import resolve_collection_plan

    assert resolve_collection_plan(Namespace(), raw).max_attempts == 1
    assert cfg.task_program is not None
    assert get_env_spec(raw["id"]).cls is DrawerStorageEnv


def _run_franka_contact_episode() -> None:
    """Check intended contact pairs against native impulse records by segment."""
    from pathlib import Path

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import make
    from embodichain.utils.utility import load_config

    source = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage/task.franka.yaml"
    )
    payload = load_config(source)
    payload.update(
        headless=True, device="cpu", renderer="no-render", enable_sensor=False, seed=0
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.filter_dataset_saving = True

    fingers = ("fr3_leftfinger", "fr3_rightfinger")
    allowed = {
        "open_drawer": [
            ("cube", "pickup_table"),
            *((f, "large_handle_bar") for f in fingers),
        ],
        "pick_cube": [("cube", "pickup_table"), *((f, "cube") for f in fingers)],
        "store_cube": [("cube", "drawer_bottom"), *((f, "cube") for f in fingers)],
        "close_drawer": [
            ("cube", "drawer_bottom"),
            *((f, "large_handle_bar") for f in fingers),
        ],
    }
    allowed = {
        stage: {frozenset(pair) for pair in pairs} for stage, pairs in allowed.items()
    }
    env = make(payload["id"], cfg=cfg)
    try:
        env.reset(seed=0, options={"save_data": False})
        spawn = env.sim.spawn_result
        targets = [
            path for path in spawn.handles if path.endswith(("/FrankaPanda", "/cube"))
        ]
        assert len(targets) == 2
        query = spawn.create_contact_query(
            targets, match="any", capacity=4096, device="cpu"
        )
        names = {
            actor.actor_id: actor.link_name or actor.path.rsplit("/", 1)[-1]
            for actor in query.actors
        }
        observed = {stage: set() for stage in allowed}
        stage = None
        sample_count = 0
        begin = env._begin_demo_segment_recording

        def begin_segment(segment_id, segment):
            nonlocal stage
            stage = segment.name
            return begin(segment_id, segment)

        def sample_substep(_dt):
            nonlocal sample_count
            sample_count += 1
            buffer = query.fetch()
            assert not buffer.dropped_count
            # Column 9 contains the normal contact impulse.
            impulses = buffer.data[: buffer.count, 9]
            for left, right in buffer.actor_ids[: buffer.count][impulses > 0].tolist():
                observed[stage].add(frozenset((names[left], names[right])))

        env._begin_demo_segment_recording = begin_segment
        env._advance_physics = lambda: env.sim.update(
            env.physics_dt,
            env.cfg.sim_steps_per_control,
            after_substep=sample_substep,
            render_final_step=False,
        )
        result = execute_demo_episode(env)
        assert result.completed and result.all_success, result.to_metadata()
        assert sample_count == result.length * env.cfg.sim_steps_per_control
        for stage, pairs in observed.items():
            assert pairs <= allowed[stage], (stage, pairs - allowed[stage])
            assert any(pair.intersection(fingers) for pair in pairs), stage
    finally:
        env.close(exit_process=False)


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
def test_franka_drawer_storage_contact_pairs() -> None:
    """Reject a successful episode that hits unintended scene geometry in transit."""
    from pathlib import Path
    import subprocess
    import sys

    path = Path(__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_run_franka_contact_episode']()",
            str(path),
        ],
        cwd=path.parents[4],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
