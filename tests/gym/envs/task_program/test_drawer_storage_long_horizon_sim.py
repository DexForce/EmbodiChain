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

"""Physical qualification of supported release and contact-driven drawer closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PARTS = ("drink_can", "water_bottle", "coffee_cup", "paper_cup")


def _qualify_drawer_episode(env: Any) -> tuple[Any, dict[str, Any]]:
    """Observe floor support at release and hand contact throughout drawer closing."""
    from embodichain.lab.gym.envs.demo import (
        execute_demo_episode,
        resolve_demo_segments,
    )
    from embodichain.lab.gym.envs.types import ControllerAction
    import torch
    from embodichain.lab.sim.sensors import (
        ArticulationContactFilterCfg,
        ContactSensorCfg,
    )

    robot = env.sim.get_robot("UR5")
    drawer = env.sim.get_articulation("drawer")
    sensor = env.sim.add_sensor(
        ContactSensorCfg(
            uid="drawer_qualification_contacts",
            rigid_uid_list=list(_PARTS),
            articulation_cfg_list=[
                ArticulationContactFilterCfg(
                    articulation_uid="drawer",
                    link_name_list=["drawer_bottom", "large_handle_bar"],
                ),
                ArticulationContactFilterCfg(
                    articulation_uid="UR5",
                    link_name_list=["gripper_finger1_link_1", "gripper_finger2_link_1"],
                ),
            ],
            max_contacts_per_env=256,
        )
    )
    floor_id, handle_id = sensor.get_actor_ids(
        "drawer", ["drawer_bottom", "large_handle_bar"]
    )[0].tolist()
    finger_ids = set(
        sensor.get_actor_ids(
            "UR5", ["gripper_finger1_link_1", "gripper_finger2_link_1"]
        )[0].tolist()
    )
    object_ids = {part: int(sensor.get_actor_ids(part)[0, 0]) for part in _PARTS}
    current_segment = ""
    releases: dict[str, dict[str, Any]] = {}
    previous_command = 0.024
    free_closing_distance = 0.0
    contact_closing_distance = 0.0
    original_step = env.step
    hand_joint = robot.get_joint_ids(name="eef", remove_mimic=True)[0]

    def contacts() -> set[frozenset[int]]:
        sensor.update()
        data = sensor.get_data()
        assert sensor.dropped_contacts == 0, "Contact qualification overflowed."
        return {
            frozenset(pair)
            for pair in data["user_ids"][0][data["is_valid"][0]].cpu().tolist()
        }

    def tracked_step(action: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal previous_command, free_closing_distance, contact_closing_distance
        value = action.value if isinstance(action, ControllerAction) else action
        qpos = value if isinstance(value, torch.Tensor) else value["qpos"]
        command = float(qpos[0, hand_joint])
        if current_segment.startswith("store_"):
            part = current_segment.removeprefix("store_")
            if command < previous_command - 1.0e-5 and part not in releases:
                pairs = contacts()
                floor = (
                    drawer.get_link_pose("drawer_bottom", to_matrix=True)[0]
                    .cpu()
                    .numpy()
                )
                obj = (
                    env.sim.get_rigid_object(part)
                    .get_local_pose(to_matrix=True)[0]
                    .cpu()
                    .numpy()
                )
                relative = floor[:3, :3].T @ (obj[:3, 3] - floor[:3, 3])
                releases[part] = {
                    "supported_before_open": frozenset((object_ids[part], floor_id))
                    in pairs,
                    "relative_position_before_open": relative.tolist(),
                    "drawer_position": float(drawer.get_qpos()[0, 0]),
                }
        previous_command = command
        if current_segment != "close_drawer":
            return original_step(action, *args, **kwargs)
        before = float(drawer.get_qpos()[0, 0])
        result = original_step(action, *args, **kwargs)
        distance = max(0.0, before - float(drawer.get_qpos()[0, 0]))
        if distance:
            pairs = contacts()
            if any(frozenset((handle_id, finger)) in pairs for finger in finger_ids):
                contact_closing_distance += distance
            else:
                free_closing_distance += distance
        return result

    def segments() -> Any:
        nonlocal current_segment, previous_command
        for segment in resolve_demo_segments(env):
            current_segment = segment.name
            previous_command = 0.024
            yield segment

    env.step = tracked_step
    try:
        result = execute_demo_episode(env, segments=segments())
    finally:
        env.step = original_step
    evidence = {
        "releases": releases,
        "contact_closing_distance_m": contact_closing_distance,
        "free_closing_distance_m": free_closing_distance,
    }
    evidence["accepted"] = (
        set(releases) == set(_PARTS)
        and all(entry["supported_before_open"] for entry in releases.values())
        and contact_closing_distance > 0.20
        and free_closing_distance < 0.003
    )
    return result, evidence


def _run_episode() -> dict[str, Any]:
    """Exercise the official deployment with ordinary physics and demo execution."""
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
        make,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    discover_task_packages()
    execute_init_hooks()
    source = (
        _REPOSITORY_ROOT
        / "extensions/embodichain_tasks/configs/tasks/manipulation"
        / "drawer_storage"
        / "task.ur5.long_horizon.yaml"
    )
    payload = load_config(source)
    payload.update(
        num_envs=1,
        headless=True,
        renderer="no-render",
        device="cpu",
        enable_sensor=False,
        seed=0,
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    env = None
    try:
        env = make(payload["id"], cfg=cfg)
        env.reset(seed=0, options={"save_data": False})
        result, physical = _qualify_drawer_episode(env)
        positions = np.array(
            [
                env.sim.get_rigid_object(part)
                .get_local_pose(to_matrix=True)[0, :3, 3]
                .cpu()
                .numpy()
                for part in _PARTS
            ]
        )
        drawer = env.sim.get_articulation("drawer")
        reference = (
            drawer.get_link_pose("drawer_bottom", to_matrix=True)[0].cpu().numpy()
        )
        joint_position = float(drawer.get_qpos()[0, 0])
        relative = (positions - reference[:3, 3]) @ reference[:3, :3]
        return {
            "physical_acceptance": physical,
            "completed": result.completed,
            "all_success": result.all_success,
            "episode": result.to_metadata(),
            "relative_positions": relative.tolist(),
            "drawer_position": joint_position,
        }
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
def test_long_horizon_drawer_storage_completes_with_supported_release_and_contact_closure(
    tmp_path: Path,
) -> None:
    """Catch contact, reachability, ordering, and loaded-drawer regressions."""
    destination = tmp_path / "drawer_storage_long_horizon.json"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(destination)],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(destination.read_text())
    assert report["completed"] and report["all_success"], report["episode"]
    assert len(report["episode"]["segments"]) == 24
    assert all(segment["success"] for segment in report["episode"]["segments"])
    # Mesh-origin heights above the floor: floor half-thickness plus each
    # asset's scaled lower Z bound (can radius, bottle base, and cup bases).
    expected = np.array(
        [
            [-0.09, -0.04, 0.0267],
            [-0.03, -0.04, 0.0391],
            [0.03, -0.04, 0.0256],
            [0.09, -0.04, 0.0371],
        ]
    )
    assert report["physical_acceptance"]["accepted"], report["physical_acceptance"]
    assert -0.003 <= report["drawer_position"] <= 0.005
    assert np.all(
        np.abs(np.array(report["relative_positions"]) - expected)
        < [0.025, 0.025, 0.015]
    )


if __name__ == "__main__":
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    status = 1
    try:
        report = _run_episode()
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        status = 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
