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

"""Production workspace-demo evidence and renderer/physics separation."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.scripts._robot_workspace_scene import CUBE_Z, GOAL, _fragment
from embodichain.lab.scripts._robot_workspace_task import _Evidence, _PinnedGrasp
from embodichain.lab.scripts._robot_workspace_view import _Display
from embodichain.lab.scripts.robot_workspace import run_demo

pytestmark = pytest.mark.no_sim


def test_shared_scene_has_static_surface_and_dynamic_cube():
    import numpy as np

    entities = {cfg.uid: cfg for cfg in _fragment().entities}
    assert entities["support_surface"].body_type == "static"
    table = entities["support_surface"]
    assert table.shape.shape_type == "Mesh"
    assert table.shape.collision.approximation == "triangle_mesh"
    vertices = np.asarray(table.shape.vertices) + np.asarray(table.init_pos)
    assert vertices[:, 2].min() == pytest.approx(0.0)
    assert vertices[:, 2].max() == pytest.approx(0.75)
    top = vertices[np.isclose(vertices[:, 2], 0.75)]
    assert np.linalg.norm(top[:, :2], axis=-1).max() == pytest.approx(1.2)
    assert entities["cube"].body_type == "dynamic"
    assert entities["cube"].init_pos[2] == CUBE_Z


def test_pinned_grasp_follows_actual_object_pose_without_resampling():
    local = torch.eye(4)
    local[2, 3] = 0.01
    generator = _PinnedGrasp({"pose": local})
    objects = torch.eye(4).repeat(2, 1, 1)
    objects[0, 0, 3] = -0.42
    objects[1, 1, 3] = 0.48
    rows = generator.get_valid_grasp_poses(
        mesh_vertices=None,
        mesh_triangles=None,
        obj_poses=objects,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    for index, (poses, costs) in enumerate(rows):
        torch.testing.assert_close(poses[0], objects[index] @ local)
        assert len(poses) == 1
    assert local[2, 3] == pytest.approx(0.01)


def test_evidence_requires_lift_release_and_ten_consecutive_terminal_steps():
    body = torch.zeros(1, 13)
    body[0, :3] = torch.tensor(GOAL)
    finger = torch.tensor([[0.024]])
    cube = SimpleNamespace(body_state=body, get_local_pose=lambda: body[:, :7])
    env = SimpleNamespace(
        sim=SimpleNamespace(get_rigid_object=lambda uid: cube),
        robot=SimpleNamespace(get_qpos=lambda **k: finger),
    )
    evidence = _Evidence()
    assert not evidence.validation().accepted
    evidence.reset(initial_height=CUBE_Z)
    body[0, 2] = CUBE_Z + 0.06
    evidence.observe(env)
    assert evidence.lift >= 0.05
    body[0, :3] = torch.tensor(GOAL)
    for _ in range(12):
        evidence.observe(env)
    assert not evidence.validation().accepted  # Still held/closed.
    finger[:] = 0.0
    for _ in range(9):
        evidence.observe(env)
    assert not evidence.validation().accepted
    evidence.observe(env)
    assert evidence.validation().accepted
    body[0, 0] += 0.1
    evidence.observe(env)
    assert not evidence.validation().accepted  # Cannot inherit earlier success.


def test_render_only_capture_never_advances_physics(tmp_path):
    calls = []

    @contextmanager
    def frame(**kwargs):
        calls.append("frame")
        yield

    sim = SimpleNamespace(
        render_frame=frame, update=lambda **kwargs: calls.append("physics")
    )
    display = _Display("window", tmp_path)
    display.frame(sim, "capture")
    assert calls == ["frame"]
    display.view = "none"
    display.advance(sim, 7)
    assert calls.count("physics") == 7
    assert display.steps == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid"},
        {"view": "invalid"},
        {"seed": -1},
        {"candidates": 0},
        {"candidates": True},
    ],
)
def test_invalid_demo_arguments_fail_before_creating_output(tmp_path, kwargs):
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        run_demo(output=output, **kwargs)
    assert not output.exists()


def test_lift_evidence_uses_measured_initial_height_instead_of_authored_height() -> (
    None
):
    initial = CUBE_Z + 0.004
    body = torch.zeros(1, 13)
    body[0, :3] = torch.tensor(GOAL)
    body[0, 2] = initial + 0.049
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: SimpleNamespace(
                body_state=body, get_local_pose=lambda: body[:, :7]
            )
        ),
        robot=SimpleNamespace(get_qpos=lambda **kwargs: torch.tensor([[0.024]])),
    )
    evidence = _Evidence()
    evidence.reset(initial_height=initial)
    evidence.observe(env)
    assert evidence.validation().checks[0].status == "failed"
    assert evidence.lift == pytest.approx(0.049, abs=1e-6)


def test_physical_input_loading_does_not_auto_register_task_packages() -> None:
    import os, subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    code = """
import sys
from embodichain.lab.scripts._robot_workspace_scene import _embodiment, _fragment
assert _embodiment()['simulation']['robot_type'] == 'ur5'
assert len(_fragment().entities) == 2
assert not any(n.startswith('embodichain.lab.gym') or n.startswith('embodichain.lab.task_program') for n in sys.modules)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_red_green_cloud_freshly_checks_every_upright_object_pose(tmp_path) -> None:
    import numpy as np
    from embodichain.lab.scripts._robot_workspace_scene import (
        _make_reachability_map,
        NEGATIVE,
    )
    from embodichain.lab.sim.scene import SceneCheck, SceneCheckReport

    checked = []
    grasp = torch.eye(4)
    grasp[2, 3] = 0.03

    class Workspace:
        def check_object_pose(self, change, local_grasp, *, joint_seed):
            assert local_grasp is grasp
            pose = np.asarray(change.arena_pose)
            np.testing.assert_array_equal(pose[:3, :3], np.eye(3))
            checked.append((pose[:3, 3], joint_seed))
            passed = bool(pose[0, 3] > -0.8)
            return SceneCheckReport(
                (
                    SceneCheck(
                        "workspace.pose_ik",
                        "passed" if passed else "failed",
                        "actual pose check",
                        (
                            {"position_error_m": 0.0, "orientation_error_rad": 0.0}
                            if passed
                            else {}
                        ),
                    ),
                )
            )

    pose = np.eye(4)
    pose[:3, 3] = (-0.5, 0, CUBE_Z)
    candidate = SimpleNamespace(
        variant=SimpleNamespace(pose_changes=[SimpleNamespace(arena_pose=pose)]),
        joint_seed=(0.4, 0.6),
    )
    info = _make_reachability_map(
        Workspace(), grasp, (candidate,), torch.zeros(2), tmp_path
    )
    with np.load(info["cache"], allow_pickle=False) as data:
        assert len(checked) == info["point_count"] == len(data["points"])
        np.testing.assert_allclose(data["points"][:, 2], CUBE_Z)
        expected = data["points"][:, 0] > -0.8
        np.testing.assert_array_equal(data["reachable"], expected)
        assert np.all(data["colors"][expected] == (40, 220, 100))
        assert np.all(data["colors"][~expected] == (245, 65, 65))
        np.testing.assert_allclose(data["points"][info["negative_index"]], NEGATIVE)
        assert not data["reachable"][info["negative_index"]]
        assert data["reachable"][info["candidate_indices"]].all()
        assert np.isnan(data["position_error_m"][~expected]).all()
        np.testing.assert_allclose(
            data["joint_seeds"][info["candidate_indices"][0]], candidate.joint_seed
        )
        corners = np.asarray(
            [[-0.025, -0.025], [-0.025, 0.025], [0.025, -0.025], [0.025, 0.025]]
        )
        radii = np.linalg.norm(data["points"][:, None, :2] + corners, axis=-1)
        assert (radii <= 1.2 * np.cos(np.pi / 96) + 1e-6).all()


@pytest.mark.parametrize(
    "center, supported",
    [
        ((0.0, 0.0), True),
        ((1.0, 0.9), False),
        ((1.19, 0.0), False),
        ((1.14, 0.0), True),
    ],
)
def test_initial_support_checks_whole_cube_against_round_table(
    center, supported
) -> None:
    from embodichain.lab.scripts._robot_workspace_scene import _initial_checks
    from embodichain.lab.sim.scene import SceneCheck, SceneCheckReport

    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([*center, CUBE_Z])
    cube = SimpleNamespace(
        get_local_pose=lambda **kwargs: pose, body_state=torch.zeros(1, 13)
    )
    sim = SimpleNamespace(get_rigid_object=lambda uid: cube)
    workspace = SimpleNamespace(
        check_object_pose=lambda *args, **kwargs: SceneCheckReport(
            (SceneCheck("workspace.pose_ik", "passed"),)
        )
    )
    result = _initial_checks(sim, workspace, torch.eye(4))
    assert (result.checks[0].status == "passed") == supported
