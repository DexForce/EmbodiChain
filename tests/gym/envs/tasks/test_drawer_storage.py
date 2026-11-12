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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.task_program._final_acceptance import (
    _deployment_final_rules,
)
from embodichain.lab.gym.utils.trajectory_state import _evaluation_observation_cfgs
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import get_env_spec
from embodichain.utils.utility import load_config

_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage"
)


def _load_deployment(embodiment):
    path = _CONFIG / f"task.{embodiment}.yaml"
    raw = load_config(path)
    return config_to_cfg(raw, source_path=path)


def _observed_task(embodiment="ur5"):
    cfg = _load_deployment(embodiment)
    open_hand, finger_links = {
        "ur5": ((0.0,), ("gripper_finger1_link_1", "gripper_finger2_link_1")),
        "franka": ((0.04, 0.04), ("fr3_leftfinger", "fr3_rightfinger")),
    }[embodiment]
    n = 7
    tray = torch.tensor([-1.0, 0.0, 0.445, 0.0, 0.0, 0.5**0.5, 0.5**0.5]).repeat(n, 1)
    tray_rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    local = torch.tensor([[0.0, 0.105, -0.032]]).repeat(n, 1)
    local[1, 2] += 0.3  # XY-aligned floating cube must fail.
    local[2, 0] = 0.19  # Centre inside, but a corner protrudes through the side.
    cube = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(n, 1)
    cube[:, :3] = local @ tray_rotation.T + tray[:, :3]
    rail = torch.zeros(n, 1)
    rail[3] = 0.1
    hand = torch.tensor(open_hand).repeat(n, 1)
    hand[4, -1] += 0.01  # A single unopen joint must fail.
    finger = cube.clone()
    finger[:, 0] += 0.2
    second_finger = finger.clone()
    second_finger[5, 0] -= 0.18
    second_finger[6, 0] = torch.inf
    fingers = dict(zip(finger_links, (finger, second_finger), strict=True))
    obj = SimpleNamespace(get_local_pose=lambda **kw: cube)
    drawer = SimpleNamespace(
        joint_names=["cabinet_to_drawer"],
        get_qpos=lambda **kw: rail,
        get_link_pose=lambda *a, **kw: tray,
    )
    env = object.__new__(EmbodiedEnv)
    env._num_envs = n
    env.cfg = cfg
    env.episode_context = EpisodeContext(n)
    env.episode_context._begin_episode(range(n))
    registration = get_env_spec(
        f"TaskProgramDrawerStorage-{'UR5' if embodiment == 'ur5' else 'Franka'}-v1"
    ).task_program_registration
    env._final_acceptance_rules = _deployment_final_rules(registration)
    env._evaluation_recording = _evaluation_observation_cfgs(registration)
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True, completion_mask=torch.ones(n, dtype=torch.bool)
    )
    env.sim = SimpleNamespace(
        device=torch.device("cpu"),
        asset_uids=("cube", "drawer"),
        get_asset={"cube": obj, "drawer": drawer}.get,
        get_articulation=lambda uid: drawer,
    )

    def read_hand(*, name, target):
        assert name == "eef"
        assert target is False
        return hand

    env.robot = SimpleNamespace(
        get_qpos=read_hand,
        joint_names=tuple(f"finger_{index}" for index in range(len(open_hand))),
        get_joint_ids=lambda **kw: tuple(range(len(open_hand))),
        get_link_pose=lambda name, **kw: fingers[name],
    )
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        _bind_evaluation_joint_names,
    )

    env.robot.control_parts = {"eef": list(env.robot.joint_names)}
    bound = _bind_evaluation_joint_names(registration.evaluation_conditions, env.robot)
    env._final_acceptance_rules = tuple(
        (ref, bound[ref]) for ref in registration.evaluation_final_refs
    )
    return env


@pytest.mark.parametrize("embodiment", ["ur5", "franka"])
def test_storage_requires_corners_closed_rail_open_hand_and_clearance(embodiment):
    env = _observed_task(embodiment)
    assert env.is_task_success().tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_final_acceptance_remeasures_after_completion():
    env = _observed_task()
    bridge = env._active_task_program_bridge
    env._active_task_program_bridge = None
    assert not env.is_task_success().any()
    env._active_task_program_bridge = bridge
    bridge.program_completed = False
    assert not env.is_task_success().any()
    bridge.program_completed = True

    # The drawer moves after the program completed. A final-segment snapshot
    # would keep accepting row 0. A newly closed drawer can also recover row 3.
    assert env.is_task_success()[0]
    rail = env.sim.get_articulation("drawer").get_qpos(target=False)
    rail[0] = 0.01001
    rail[3] = 0.010
    assert env.is_task_success().tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    # An episode-level program replacement still cannot bypass the host gate.
    env.cfg.task_program = None
    bridge.completion_mask[3] = False
    assert not env.is_task_success().any()
    bridge.completion_mask[3] = True
    rail[3] = -0.001
    assert env.is_task_success()[3]
    rail[3] = torch.nan
    assert not env.is_task_success()[3]


def test_configured_acceptance_is_part_of_registration_and_construction(tmp_path):
    from argparse import Namespace
    from embodichain.lab.scripts.run_task import resolve_collection_plan

    source = _CONFIG / "task.ur5.yaml"
    raw = load_config(source)
    config_to_cfg(raw, source_path=source)
    assert resolve_collection_plan(Namespace(), raw).max_attempts == 1
    spec = get_env_spec(raw["id"])
    assert spec.cls is EmbodiedEnv
    config_to_cfg(raw, source_path=source)
    assert get_env_spec(raw["id"]) is spec
    assert spec.task_program_registration.evaluation_final_refs == (
        "cube_contained",
        "rail_closed",
        "hand_open",
        "fingers_clear",
    )

    weaker = tmp_path / "evaluation.yaml"
    declaration = load_config(_CONFIG / "task_program/evaluation.yaml")
    declaration["graph"]["rules"]["rail_closed"]["params"]["upper"] = 0.02
    weaker.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="different environment or integration"):
        config_to_cfg(
            {**raw, "evaluation": {"component": str(weaker)}}, source_path=source
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("tolerence", 0.001, ValueError, "declared contract"),
        ("tolerance", True, ValueError, "must be finite"),
        ("tolerance", float("nan"), ValueError, "must be finite"),
        ("corners", [[0.025, 0.025]], ValueError, "XYZ triples"),
        ("lower", [0.3, 0.31, 0.05], ValueError, "ordered XYZ triples"),
    ],
)
def test_evaluation_component_rejects_invalid_predicates(
    tmp_path, field, value, error, message
):
    source = _CONFIG / "task.franka.yaml"
    raw = load_config(source)
    component = tmp_path / "evaluation.yaml"
    payload = load_config(_CONFIG / "task_program/evaluation.yaml")
    payload["graph"]["rules"]["cube_contained"]["params"][field] = value
    component.write_text(yaml.safe_dump(payload), encoding="utf-8")
    raw["evaluation"]["component"] = str(component)
    with pytest.raises(error, match=message):
        config_to_cfg(raw, source_path=source)


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
            buffer = query.fetch_solver_report()
            assert not buffer.dropped_count
            assert bool(buffer.impulse_valid[: buffer.count].all())
            # Column 9 contains the measured normal solver impulse. Geometry
            # detection has no impulses and cannot qualify this contact audit.
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
