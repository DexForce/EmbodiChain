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

"""Physical outcome regressions for configured production manipulation experts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SUBPROCESS_TIMEOUT_SECONDS = 180
# Keep room for the configured contact offsets on both scene assets.
_MIN_STAND_CLEARANCE_M = 0.015
_RUN_EPISODE = (
    "import runpy, sys; "
    "module = runpy.run_path(sys.argv[1]); "
    "module['_run_child'](sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])"
)


def _physical_state(env: Any, task_name: str) -> dict[str, object]:
    """Read object poses and speeds independently of semantic acceptance."""
    import torch

    uids = ("handover_object",) if task_name == "hand_over" else ("block_1", "block_2")
    if task_name == "blocks_ranking_rgb":
        uids += ("block_3",)
    result = {}
    for uid in uids:
        obj = env.sim.get_rigid_object(uid)
        result[uid] = {
            "pose": obj.get_local_pose(to_matrix=True)[0].cpu().tolist(),
            "linear_speed": torch.linalg.vector_norm(obj.body_data.lin_vel[0]).item(),
            "angular_speed": torch.linalg.vector_norm(obj.body_data.ang_vel[0]).item(),
        }
    return result


def _tianji_stand_clearance(env: Any) -> float:
    """Measure the fixed stand against the tabletop's actual mesh height band."""
    import xml.etree.ElementTree as ET

    import numpy as np
    from scipy.spatial.transform import Rotation
    import trimesh

    from embodichain.data import get_data_path

    assert env.robot.cfg.root_props.fixed_base
    urdf = Path(get_data_path(env.robot.cfg.fpath))
    collision = ET.parse(urdf).getroot().find("link[@name='base_link']/collision")
    assert collision is not None
    geometry = collision.find("geometry/mesh")
    assert geometry is not None
    stand = trimesh.load(
        urdf.parent / geometry.attrib["filename"], force="mesh", process=False
    )
    stand.apply_scale(np.fromstring(geometry.attrib.get("scale", "1 1 1"), sep=" "))
    origin = collision.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    root_pose = env.robot.get_link_pose("base_link", to_matrix=True)[0].cpu().numpy()
    stand.apply_transform(root_pose @ transform)

    table = env.sim.get_rigid_object("table")
    tabletop = trimesh.load(
        get_data_path(table.cfg.shape.fpath), force="mesh", process=False
    )
    tabletop.apply_scale(table.get_body_scale()[0].cpu().numpy())
    tabletop.apply_transform(table.get_local_pose(to_matrix=True)[0].cpu().numpy())
    # Feet may extend below the table. Only the stand's geometry at tabletop
    # height can penetrate the slab in this fixed-base, rear-facing layout.
    column = stand.slice_plane([0, 0, tabletop.bounds[0, 2]], [0, 0, 1])
    column = column.slice_plane([0, 0, tabletop.bounds[1, 2]], [0, 0, -1])
    assert not column.is_empty
    return float(tabletop.bounds[0, 0] - column.bounds[1, 0])


def _run_real_sim_episode(
    task_name: str, seed: int, output_dir: Path, variant: str = "cobotmagic"
) -> dict[str, Any]:
    """Execute the production deployment and then replay only recorded actions."""
    import torch

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.envs.wrapper import ReplayWrapper
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import discover_task_packages, make
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    discover_task_packages()
    source = (
        _REPOSITORY_ROOT
        / "extensions/embodichain_tasks/configs/tasks/manipulation"
        / (task_name if task_name == "hand_over" else f"tableware/{task_name}")
        / f"task.{variant}.yaml"
    )
    payload = load_config(source)
    payload.update(
        num_envs=1,
        headless=True,
        renderer="no-render",
        device="cpu",
        enable_sensor=False,
        seed=seed,
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    cfg.filter_visual_rand = True
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    env = None
    try:
        env = make(payload["id"], cfg=cfg)
        env.reset(seed=seed, options={"save_data": False})
        stand_clearance = None
        if variant == "tianji_marvin" and task_name == "stack_blocks_two":
            stand_clearance = _tianji_stand_clearance(env)
            assert stand_clearance >= _MIN_STAND_CLEARANCE_M
        result = execute_demo_episode(env)
        report = {
            "episode": result.to_metadata(),
            "expert_final": _physical_state(env, task_name),
            "stand_table_clearance_m": stand_clearance,
        }
        (output_dir / "outcome.json").write_text(
            json.dumps(report, allow_nan=False), encoding="utf-8"
        )
        assert result.completed and result.all_success, report["episode"]
        trajectory_path = output_dir / "trajectory.pt"
        env.save_trajectory(str(trajectory_path))
        trajectory = torch.load(trajectory_path, weights_only=False)
        replay = ReplayWrapper(env, trajectory, mode="dynamic")
        replay.reset(seed=seed, options={"save_data": False})
        length = int(trajectory["meta"]["lengths"][0])
        assert length == result.length and length > 0
        for _ in range(length):
            _, _, _, truncated, _ = replay.step(None)
        assert truncated.tolist() == [True]
        report["replay_final"] = _physical_state(env, task_name)
        return report
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _run_child(
    task_name: str, seed: int, output_directory: str, variant: str = "cobotmagic"
) -> None:
    """Keep native interpreter teardown from masking a checked simulation result."""
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    exit_code = 1
    try:
        output_dir = Path(output_directory)
        report = _run_real_sim_episode(task_name, seed, output_dir, variant)
        (output_dir / "outcome.json").write_text(
            json.dumps(report, allow_nan=False), encoding="utf-8"
        )
        exit_code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def _assert_physical_goal(task_name: str, state: dict[str, Any]) -> None:
    """Check measured placement, upright poses, and resting speeds directly."""
    poses = {uid: value["pose"] for uid, value in state.items()}
    positions = {
        uid: [pose[axis][3] for axis in range(3)] for uid, pose in poses.items()
    }
    for uid, value in state.items():
        assert all(math.isfinite(item) for row in poses[uid] for item in row)
        if task_name == "stack_blocks_two":
            assert poses[uid][2][2] > math.cos(math.pi / 4), (uid, state)
        elif task_name == "blocks_ranking_rgb":
            # A colored cube may rest on any face; RGB ranking has no up axis.
            assert max(abs(value) for value in poses[uid][2][:3]) > math.cos(
                math.pi / 4
            ), (uid, state)
        assert value["linear_speed"] < 0.03, (uid, state)
        assert value["angular_speed"] < 0.20, (uid, state)
    if task_name == "hand_over":
        # Authored delivery_rest_pose in the shared tabletop hand-over program.
        destination = (0.45, -0.18, 0.782)
        error = math.dist(positions["handover_object"], destination)
        assert error < 0.05, state
    elif task_name == "stack_blocks_two":
        separation = [a - b for a, b in zip(positions["block_2"], positions["block_1"])]
        assert abs(separation[0]) < 0.025, state
        assert abs(separation[1]) < 0.025, state
        assert abs(separation[2] - 0.05) < 0.012, state
    else:
        for uid, offset in (("block_1", -0.08), ("block_3", 0.08)):
            assert (
                abs(positions[uid][0] - positions["block_2"][0] - offset) < 0.035
            ), state
            assert abs(positions[uid][1] - positions["block_2"][1]) < 0.03, state


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    ("task_name", "variant"),
    [
        (task, variant)
        for task in ("stack_blocks_two", "blocks_ranking_rgb")
        for variant in (
            "cobotmagic",
            "dual_ur5_dh_pgi_140_80",
            "dual_ur10_dh_pgi_140_80",
            "tianji_marvin",
        )
    ]
    + [
        ("hand_over", "dual_ur10_dh_pgi_140_80"),
        ("hand_over", "tianji_marvin"),
    ],
)
def test_production_expert_and_dynamic_replay_satisfy_physical_goal(
    task_name: str, variant: str, tmp_path: Path
) -> None:
    """The shared Task Program expert and action replay reach the same physical goal."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_EPISODE,
            str(Path(__file__).resolve()),
            task_name,
            "0",
            str(tmp_path),
            variant,
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "outcome.json").read_text(encoding="utf-8"))
    episode = report["episode"]
    assert episode["completed"] and episode["success"] == [True], episode
    segments = episode["segments"]
    assert (
        len(segments)
        == {
            "stack_blocks_two": 2,
            "blocks_ranking_rgb": 4,
            "hand_over": 1,
        }[task_name]
    )
    for segment in segments:
        assert segment["success"], segment
        validation = segment["metadata"]["validation"]
        assert validation["accepted_mask"] == [True], segment
        assert validation["validators"], segment
        for trace in validation["validators"]:
            assert (
                trace["result_mask"] == trace["result"]["accepted_mask"] == [True]
            ), trace
        for trace in segment["metadata"]["post_policies"]:
            assert trace["result_mask"] == [True], trace
            assert trace["result"]["state"]["settled_mask"] == [True], trace
    _assert_physical_goal(task_name, report["expert_final"])
    _assert_physical_goal(task_name, report["replay_final"])
