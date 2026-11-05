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
"""CPU contracts for S03 topology, timed control and material-node acceptance."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np

TECH = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, TECH / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("s03_common", "physics/experiments/dexuni/s03_common.py")
geometry = _load(
    "s03_geometry",
    "physics/experiments/dexuni/s03_cloth_corner_lift/tools/prepare_inputs.py",
)
metrics = _load("s03_metrics", "reporting/dexuni/s03/analyze.py")
CONFIG = json.loads((TECH / "assets/physics/dexuni/s03/s03_config.json").read_text())


def test_flat_sheet_preserves_physical_nodes_and_corner_group() -> None:
    vertices, triangles, corner = geometry.cloth_mesh(CONFIG)
    assert vertices.shape == (961, 3) and triangles.shape == (1800, 3)
    assert set(corner) == {0, 1, 31, 32}
    x = vertices[triangles]
    normals = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    assert (normals[:, 2] > 0).all()
    assert abs(np.linalg.norm(normals, axis=1).sum() / 2 - 0.09) < 1e-6
    assert np.ptp(vertices[:, 2]) == 0


def test_timed_controller_holds_and_fully_opens_without_particle_control() -> None:
    controller = common.CornerLiftController(
        CONFIG, np.array([0, 0.18, 0.26, 0.43, 0.51])
    )
    for frame in range(1, controller.max_frames() + 1):
        controller.step(frame / 60, 0.861)
        if controller.finished:
            break
    assert controller.finished
    assert abs(controller.events["transport"] - controller.events["hold"] - 2) < 1e-9
    assert abs(controller.finger - 0.039) < 1e-9
    assert abs(controller.s - 0.51) < 1e-9


def _run() -> dict:
    phase = np.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9])
    t = np.array([0, 0.9, 1, 2, 4, 5, 5.9, 6, 7.5, 9.5, 10.5, 11.2, 13.2])
    p = np.zeros((len(t), 8, 3), np.float32)
    p[:, :, 0] = 0.48
    p[:, :, 2] = 0.861
    p[(phase == 3) | (phase == 4), :4, 2] += 0.18
    return {
        "config": CONFIG,
        "manifest": {
            "run_id": "fixture",
            "platform": "fixture",
            "repeat": 0,
            "state": {"id": "s00"},
            "variant": "nominal",
            "kind": "full",
            "controller": {"events": {"hold": 4.0, "transport": 6.0}, "finished": True},
        },
        "step_seconds": np.full(len(t), 0.03),
        "states": {
            "phase": phase,
            "time": t,
            "particles": p,
            "corner_ids": np.arange(4),
            "tcp": np.zeros((len(t), 3)),
        },
    }


def test_success_uses_fixed_material_corner_and_release_not_global_peak() -> None:
    run = _run()
    row = metrics.task_metrics(run)
    assert row["success"]
    assert abs(row["corner_hold_min_mm"] - 180) < 0.01
    run["states"]["particles"][:, 0:4, 2] = 0.861
    run["states"]["particles"][:, 4:, 2] = 1.1
    assert not metrics.task_metrics(run)["pass_lift"]


def test_high_release_and_nonfinite_nodes_fail_acceptance() -> None:
    run = _run()
    run["states"]["particles"][-1, :4, 2] = 0.91
    assert not metrics.task_metrics(run)["pass_release"]
    run = _run()
    run["states"]["particles"][0, 7, 0] = np.nan
    assert not metrics.task_metrics(run)["finite"]


def test_performance_eligibility_ignores_fold_but_requires_finite_full_episode() -> (
    None
):
    runner = _load("s03_runner", "physics/experiments/dexuni/run_s03.py")
    run = {
        "config": {"trajectory": {"frame_rate": 60}},
        "manifest": {
            "status": "completed",
            "controller": {"finished": True, "events": {"done": 0.5}},
        },
        "step_seconds": np.full(30, 0.03),
        "states": {
            "time": np.arange(31) / 60,
            "particles": np.full((31, 4, 3), 1.2),
            "tcp": np.zeros((31, 3)),
            "qpos": np.zeros((31, 9)),
            "target_q": np.zeros((31, 9)),
        },
    }
    assert runner.performance_eligibility(run)["success"]
    run["states"]["particles"][-1, :, 2] = 10.0
    assert runner.performance_eligibility(run)["success"]
    run["states"]["qpos"][-1, 0] = np.nan
    assert not runner.performance_eligibility(run)["success"]
    run["states"]["qpos"][-1, 0] = 0.0
    run["step_seconds"] = run["step_seconds"][:-1]
    assert not runner.performance_eligibility(run)["success"]


def test_performance_report_identifies_frozen_budget_and_approved_preview(
    tmp_path: Path,
) -> None:
    """A reduced-budget batch must report its own settings and preview."""
    runner = _load("s03_report_runner", "physics/experiments/dexuni/run_s03.py")
    inputs = tmp_path / "inputs/s03"
    inputs.mkdir(parents=True)
    config = json.loads(json.dumps(CONFIG))
    config["solver"].update(num_substeps=8, vbd_iterations=12)
    config["solver"]["vbd_self_contact"] = {"particle_enable_self_contact": True}
    (inputs / "s03_config.json").write_text(json.dumps(config))
    preview = tmp_path / "accepted/video.mp4"
    (tmp_path / "protocol.json").write_text(
        json.dumps({"scope": "performance_only", "approved_preview": str(preview)})
    )
    runner._performance_report(tmp_path, [])
    report = (tmp_path / "REPORT.md").read_text()
    assert "8 substeps, 12 VBD iterations" in report
    assert "[Approved scene preview](accepted/video.mp4)" in report
    assert "40 VBD iterations" not in report


def test_premature_drop_during_descent_fails_even_if_final_release_is_low() -> None:
    run = _run()
    run["config"] = json.loads(json.dumps(run["config"]))
    run["config"]["acceptance"]["descent_grasp_drift_max"] = 0.02
    hold = run["states"]["phase"] == 3
    lower = run["states"]["phase"] == 5
    # A successful hold followed by an empty closed gripper descending high
    # above a dropped cloth must fail even though the final cloth is on-table.
    run["states"]["tcp"][hold, 2] = 1.041
    run["states"]["tcp"][lower, 2] = 0.98
    row = metrics.task_metrics(run)
    assert row["pass_release"] and not row["pass_descent"] and not row["success"]


def test_table_contact_allows_overhang_but_rejects_interior_penetration() -> None:
    run = _run()
    # The table begins at X=0.2 m; cloth hanging outside it can be lower.
    run["states"]["particles"][-1, 7] = [0.16, 0.0, 0.84]
    row = metrics.task_metrics(run)
    assert row["pass_table"] and row["success"]
    assert row["global_below_table_plane_mm"] > 19
    run["states"]["particles"][-1, 7, 0] = 0.3
    row = metrics.task_metrics(run)
    assert not row["pass_table"] and not row["success"]


def test_adapter_allows_repeat_fold_variation_but_rejects_grasp_mismatch() -> None:
    adapter = _load("s03_adapter", "physics/experiments/dexuni/check_s03_adapter.py")
    reference = _run()["states"]
    reference["joint_names"] = np.array(["joint"])
    reference["corner_height"] = reference["particles"][:, :4, 2].mean(axis=1)
    repeated = {key: value.copy() for key, value in reference.items()}
    native = {key: value.copy() for key, value in reference.items()}
    # Free-sheet folds can differ while the grasped corner remains equivalent.
    repeated["particles"][:, 4:, 0] += 0.004
    native["particles"][:, 4:, 0] += 0.005
    adapter.compare_states(reference, native, [reference, repeated])
    native["corner_height"][native["phase"] == 3] += 0.003
    import pytest

    with pytest.raises(AssertionError, match="phases"):
        adapter.compare_states(reference, native, repeated)


def test_texture_loading_preserves_node_identity_and_material_uv(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    import pytest

    config = json.loads(json.dumps(CONFIG))
    config["cloth"]["visual"] = {"texture_file": "fabric.png", "uv_repeat": 2.0}
    (tmp_path / "s03_config.json").write_text(json.dumps(config))
    vertices, triangles, corner = geometry.cloth_mesh(config)
    np.savez(
        tmp_path / "s03_cloth_mesh.npz",
        vertices=vertices,
        triangles=triangles,
        corner_ids=corner,
    )
    np.savez(
        tmp_path / "s03_ik_path.npz",
        s=[0.0, 1.0],
        arm_q=np.zeros((2, 7)),
        waypoint_s=[0.0, 1.0],
    )
    args = SimpleNamespace(set=None, state="s00", variant="nominal")
    with pytest.raises(FileNotFoundError):
        common.read_inputs(args, tmp_path)
    (tmp_path / "fabric.png").write_bytes(b"texture fixture")
    loaded = common.read_inputs(args, tmp_path)
    np.testing.assert_array_equal(loaded["initial"], vertices)
    np.testing.assert_array_equal(loaded["corner_ids"], corner)
    np.testing.assert_allclose(loaded["cloth_uv"].min(axis=0), [0, 0])
    np.testing.assert_allclose(loaded["cloth_uv"].max(axis=0), [2, 2])
    np.testing.assert_allclose(loaded["cloth_uv"][0], [0, 2])
    assert Path(loaded["cloth_texture"]).read_bytes() == b"texture fixture"
    config["cloth"].pop("visual")
    (tmp_path / "s03_config.json").write_text(json.dumps(config))
    assert common.read_inputs(args, tmp_path)["cloth_texture"] is None


def test_relocated_snapshot_resolves_robot_only_with_matching_frozen_hash(
    tmp_path: Path,
) -> None:
    """Moving a result keeps the original config and verifies the bundled robot."""
    import hashlib
    from types import SimpleNamespace
    import pytest

    root = tmp_path / "inputs/s03"
    robot = root / "franka_emika_panda/robot.urdf"
    robot.parent.mkdir(parents=True)
    robot.write_bytes(b"frozen robot")
    config = json.loads(json.dumps(CONFIG))
    config["robot"]["urdf"] = str(tmp_path / "deleted/robot.urdf")
    original = json.dumps(config)
    (root / "s03_config.json").write_text(original)
    vertices, triangles, corner = geometry.cloth_mesh(config)
    np.savez(
        root / "s03_cloth_mesh.npz",
        vertices=vertices,
        triangles=triangles,
        corner_ids=corner,
    )
    np.savez(
        root / "s03_ik_path.npz",
        s=[0.0, 1.0],
        arm_q=np.zeros((2, 7)),
        waypoint_s=[0.0, 1.0],
    )
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {
                "inputs_sha256": {
                    "s03/franka_emika_panda/robot.urdf": hashlib.sha256(
                        robot.read_bytes()
                    ).hexdigest()
                }
            }
        )
    )
    args = SimpleNamespace(set=None, state="s00", variant="nominal")
    assert common.read_inputs(args, root)["config"]["robot"]["urdf"] == str(robot)
    assert (root / "s03_config.json").read_text() == original
    robot.write_bytes(b"different robot")
    with pytest.raises(ValueError, match="frozen asset hash"):
        common.read_inputs(args, root)
