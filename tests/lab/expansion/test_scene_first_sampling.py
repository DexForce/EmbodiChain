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

"""All layout sources share conservative screening and own selected poses."""

from __future__ import annotations
from types import SimpleNamespace
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from scene_layout_support import make_source
from embodichain.lab.expansion._scene_first_sampling import _LayoutPoseSource
from embodichain.lab.expansion._pose_sampling_cfg import validate_sampling
from embodichain.lab.sim.scene.feasibility import FeasibilityBudgetExhausted


def setup_source(
    monkeypatch,
    strategy="scene_first",
    accepted=True,
    member_yaw=False,
    workspace=None,
    bounded_fixed_root=False,
):
    import embodichain.lab.expansion._scene_first_sampling as module

    source = make_source()

    class Object:
        def __init__(self, uid):
            self.uid = uid
            self.cfg = SimpleNamespace(to_dict=lambda: {"uid": uid})

        def get_local_pose(self, to_matrix=True):
            return torch.tensor(np.array(source.poses[self.uid]))[None]

    objects = {uid: Object(uid) for uid in source.poses}
    robot = SimpleNamespace(
        uid="robot",
        device=torch.device("cpu"),
        control_parts={"custom_arm": [0]},
        cfg=SimpleNamespace(to_dict=lambda: {"uid": "robot"}),
        get_local_pose=lambda to_matrix=True: torch.eye(4)[None],
        get_solver=lambda name: object(),
        get_qpos_limits=lambda name: torch.tensor([[[0.0, 1.0]]]),
        compute_batch_fk=lambda **kw: workspace[None],
    )
    env = SimpleNamespace(
        num_envs=1,
        robot=robot,
        sim=SimpleNamespace(
            num_envs=1,
            get_rigid_object_uid_list=lambda: tuple(objects),
            get_rigid_object=objects.get,
            sim_config=SimpleNamespace(
                physics_cfg=SimpleNamespace(to_dict=lambda: {"backend": "default"})
            ),
        ),
    )
    checked = []
    closed = []
    monkeypatch.setattr(module, "scene_layout_input_from_sim", lambda *a, **kw: source)
    monkeypatch.setattr(
        "embodichain.lab.task_program.integrations.simulation.feasibility_geometry.collision_bodies_from_simulation",
        lambda env: (),
    )
    monkeypatch.setattr(
        "embodichain.lab.expansion.acceptance._GeometryRow.contact_permissions",
        lambda *a: ((), ()),
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.scene.feasibility.NewtonCollisionChecker",
        lambda *a, **kw: SimpleNamespace(close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.scene.feasibility.RobotFeasibilityBackend",
        lambda *a, **kw: None,
    )

    def screen(mode, backend, poses, budget):
        checked.append(deepcopy(poses))
        return SimpleNamespace(
            accepted=accepted,
            checks=(
                SimpleNamespace(
                    stage="motion",
                    status="passed" if accepted else "failed",
                    reason="checked task path",
                    phase_id="reach",
                ),
            ),
        )

    monkeypatch.setattr("embodichain.lab.sim.scene.feasibility.plan_task_mode", screen)
    opts = {
        "engine": {
            "max_attempts": 8,
            "max_wall_time_s": 120.0,
            "episodes_per_operator": 1,
        },
        "scene": {
            "tray": {
                "translation_m": [[-0.1, 0.1], [-0.1, 0.1], [0, 0]],
                "yaw_rad": [0, 0],
            },
            "mug": {"follow": "tray"},
        },
        "validation": {},
        "scene_sampling": {
            "strategy": strategy,
            "control_part": "custom_arm",
            "max_candidates": 64,
            "max_rollouts": 8,
            "layout": {
                "support_uid": "workbench",
                "parent_by_uid": dict(source.parent_by_uid),
                "policy": {
                    "group_targets": ["tray"],
                    "group_xy_offset_bounds_m": {"tray": [[-0.1, -0.1], [0.1, 0.1]]},
                },
                "conditions": {"tray": np.array(source.poses["tray"]).tolist()},
            },
            "task_mode": {
                "mode_id": "runtime",
                "phases": [
                    {
                        "phase_id": "reach",
                        "targets": [
                            {
                                "control_part": "custom_arm",
                                "frame_uid": "mug",
                                "pose": np.eye(4).tolist(),
                            }
                        ],
                    }
                ],
            },
        },
        "scheduling": {"policy": "fifo", "pool_size": 4},
    }
    if strategy in ("robot_first", "bidirectional"):
        sampling = opts["scene_sampling"]
        sampling["sample_count"] = len(workspace)
        sampling["robot_anchors"] = [
            {"phase_id": "reach", "control_part": "custom_arm"}
        ]
        sampling["layout"]["conditions"] = {
            "workbench": source.poses["workbench"],
        }
        sampling["task_mode"]["phases"][0]["targets"][0]["pose"][0][3] = 0.04
        opts["scene"]["z_target"] = {
            "translation_m": [[-0.1, 0.1], [-0.1, 0.1], [0, 0]],
            "yaw_rad": [0, 0],
        }
        sampling["layout"]["policy"]["group_targets"].append("z_target")
        sampling["layout"]["policy"]["group_xy_offset_bounds_m"]["z_target"] = [
            [-0.1, -0.1],
            [0.1, 0.1],
        ]
    if bounded_fixed_root:
        opts["scene_sampling"]["layout"]["conditions"] = {}
        opts["scene"]["z_target"] = {
            "translation_m": [[-0.1, 0.1], [-0.1, 0.1], [0, 0]],
            "yaw_rad": [0, 0],
        }
    if member_yaw:
        opts["scene"]["mug"] = {
            "translation_m": [[-0.15, 0.15], [-0.15, 0.15], [0, 0]],
            "yaw_rad": [0, 0],
        }
        opts["scene_sampling"]["layout"]["policy"]["intra_group"] = {
            "targets": {"mug": {"yaw_delta_range_deg": [30, 30]}}
        }
    validate_sampling(opts)
    return (
        _LayoutPoseSource(env, opts, seed=7, revision="test-source"),
        checked,
        closed,
    )


def test_scene_first_anchors_and_workspace_task_checks_are_owned_and_cached(
    monkeypatch, tmp_path
):
    source, checked, closed = setup_source(monkeypatch)
    source.output = tmp_path
    candidate = source.candidate(0)
    assert source.validate(0).accepted and len(checked) == 1
    poses = {c.entity_uid: np.array(c.arena_pose) for c in candidate.scene.pose_changes}
    np.testing.assert_allclose(poses["tray"], source._conditions.poses["tray"])
    np.testing.assert_allclose(poses["mug"], checked[0]["mug"])
    checked[0]["mug"][0, 3] = 99
    assert candidate is source.candidate(0)
    assert source.budget.counts["candidates"] == 1
    assert (
        json.loads((tmp_path / "scene-first-checks.json").read_text())["phase_checks"][
            "0"
        ][0]["stage"]
        == "motion"
    )
    source.close()
    source.close()
    assert closed == [True]


@pytest.mark.parametrize("strategy", ["scene_first", "robot_first", "bidirectional"])
def test_failed_task_screening_never_becomes_a_successful_fallback(
    monkeypatch, strategy
):
    source, checked, _ = setup_source(
        monkeypatch, strategy=strategy, accepted=False, workspace=_workspace()
    )
    source.candidate(0)
    assert not source.validate(0).accepted and len(checked) == 4
    assert all(not e["screened"] for e in source._events if e["event"] == "proposal")


def _workspace() -> torch.Tensor:
    # Tool XY is +4 cm along the support group's X. The first three TCP samples
    # share one cell, the fourth is in a different cell, and the last is illegal.
    poses = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    poses[:, :2, 3] = torch.tensor(
        [[-0.309, -0.04], [-0.310, -0.04], [-0.311, -0.04], [-0.19, 0.05], [0.9, 0.0]],
        dtype=poses.dtype,
    )
    # Keep authored root height and orientation despite the projected FK pose.
    poses[:, 2, 3] = 0.7
    poses[:, 1, 1] = poses[:, 2, 2] = -1
    return poses


def test_robot_first_balances_prior_cells_and_completes_other_groups(monkeypatch):
    workspace = _workspace()
    source, checked, _ = setup_source(monkeypatch, "robot_first", workspace=workspace)
    workspace.fill_(77)
    first = source.candidate(0)
    second = source.candidate(1)
    assert source.validate(0).accepted and source.validate(1).accepted
    assert source.candidate(0) is first
    x_coords, other_poses = [], []
    for index, candidate in enumerate((first, second)):
        poses = {
            c.entity_uid: np.array(c.arena_pose) for c in candidate.scene.pose_changes
        }
        x_coords.append(poses["tray"][0, 3])
        other_poses.append(poses["z_target"])
        np.testing.assert_allclose(poses["tray"][:3, :3], np.eye(3))
        assert poses["tray"][2, 3] == pytest.approx(0.061)
        np.testing.assert_allclose(poses["mug"][:2, 3], poses["tray"][:2, 3])
        np.testing.assert_allclose(checked[index]["tray"], poses["tray"])
    assert min(x_coords) == pytest.approx(-0.35, abs=0.0011)
    assert max(x_coords) == pytest.approx(-0.23)
    assert not np.allclose(*other_poses)


@pytest.mark.parametrize("strategy", ["robot_first", "bidirectional"])
def test_empty_workspace_draw_costs_a_candidate_and_never_runs_task_screen(
    monkeypatch, strategy
):
    workspace = _workspace()
    workspace[:, 0, 3] = 99
    source, checked, _ = setup_source(monkeypatch, strategy, workspace=workspace)
    source.candidate(0)
    proposals = [e for e in source._events if e["event"] == "proposal"]
    assert not proposals[0]["screened"]
    assert proposals[0]["reason"] == "no_workspace_anchor"
    if strategy == "robot_first":
        assert not source.validate(0).accepted and checked == []
    else:
        assert source.validate(0).accepted and len(checked) == 1
        assert [p["origin"] for p in proposals] == ["robot_first", "scene_first"]
        assert source._events[-1]["raw_ordinal"] == 1
    assert source.budget.counts["candidates"] == len(proposals)


def test_mixed_draws_keep_direction_and_layout_after_checkpoint_restore(monkeypatch):
    source, _, _ = setup_source(monkeypatch, "bidirectional", workspace=_workspace())
    first = source.candidate(0)
    restored, _, _ = setup_source(monkeypatch, "bidirectional", workspace=_workspace())
    restored.load_state_dict(source.state_dict())
    assert restored.candidate(0) == first
    assert restored.candidate(1) == source.candidate(1)
    proposals = [e for e in restored._events if e["event"] == "proposal"]
    assert [(e["raw_ordinal"], e["origin"]) for e in proposals] == [
        (0, "robot_first"),
        (1, "scene_first"),
    ]


def test_simple_baseline_keeps_identical_task_screening_and_conditions(monkeypatch):
    source, checked, _ = setup_source(monkeypatch, strategy="bounded_pose")
    # Its simple random draw does not satisfy the exact anchor, and must be
    # rejected before any workspace/path computation.
    source.candidate(0)
    assert not source.validate(0).accepted and checked == []
    assert any(e.get("reason") == "condition_mismatch" for e in source._events)


def test_source_obeys_hard_raw_candidate_budget(monkeypatch):
    source, checked, _ = setup_source(monkeypatch)
    source.budget.charge("candidates", source.cfg["max_candidates"])
    source.candidate(0)
    assert not source.validate(0).accepted and checked == []
    assert source.budget.counts["candidates"] == 64


def test_scene_first_config_rejects_batch_and_unknown_phase_fields_before_runtime():
    # Reuse a valid source's owned config, without constructing a new World.
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/workspace_pose_pick_place/acceptance/scene_first/expansion.yaml"
    )
    cfg = yaml.safe_load(path.read_text())
    cfg["engine"]["batch_size"] = 4
    with pytest.raises(ValueError, match="B=1"):
        validate_sampling(cfg)
    cfg["engine"]["batch_size"] = 1
    cfg["scene_sampling"]["task_mode"]["phases"][0]["unknown"] = True
    with pytest.raises(ValueError, match="Unknown TaskPhase"):
        validate_sampling(cfg)


@pytest.mark.parametrize(
    "invalid, error",
    [
        ("unknown_stage", "declared phase targets"),
        ("moved_frame", "initial frame"),
        ("pinned_root", "authored root condition"),
    ],
)
def test_robot_anchors_reject_invalid_frame_provenance(invalid, error):
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = root / (
        "extensions/embodichain_tasks/configs/tasks/manipulation/workspace_pose_pick_place/"
        "acceptance/scene_first/expansion.yaml"
    )
    cfg = yaml.safe_load(path.read_text())
    sampling = cfg["scene_sampling"]
    sampling["strategy"] = "robot_first"
    if invalid == "unknown_stage":
        sampling["robot_anchors"][0]["phase_id"] = "unknown"
    elif invalid == "moved_frame":
        sampling["task_mode"]["phases"][0]["held_objects"] = [
            {"entity_uid": "cube", "control_part": "arm"}
        ]
    else:
        sampling["layout"]["conditions"] = {"cube": np.eye(4).tolist()}
    with pytest.raises(ValueError, match=error):
        validate_sampling(cfg)


def test_source_rejects_stale_geometry_and_does_not_refresh_silently(monkeypatch):
    source, checked, _ = setup_source(monkeypatch)
    source.env.sim.spawn_result = SimpleNamespace(topology_revision=2)
    source._scene_identity = (id(source.env.sim.spawn_result), 1)
    with pytest.raises(RuntimeError, match="stale"):
        source.candidate(0)
    assert checked == []


def test_fixed_layout_geometry_cannot_be_moved_by_a_bounded_draw(monkeypatch):
    source, checked, _ = setup_source(
        monkeypatch, strategy="bounded_pose", bounded_fixed_root=True
    )
    source.candidate(0)
    assert not source.validate(0).accepted and not checked
    assert source._events[0]["reason"] == "layout_moves_fixed_entity"


def test_member_layout_yaw_cannot_expand_the_task_domain(monkeypatch):
    source, checked, _ = setup_source(monkeypatch, member_yaw=True)
    source.candidate(0)
    assert not source.validate(0).accepted and checked == []
    reasons = [e["reason"] for e in source._events if e["event"] == "proposal"]
    assert reasons and set(reasons) == {"layout_outside_task_domain"}


def test_screening_must_not_step_the_owning_world(monkeypatch):
    source, _, _ = setup_source(monkeypatch)
    source.env.sim.physics_step_index = 0

    def bad_query(*args):
        source.env.sim.physics_step_index += 1
        return SimpleNamespace(accepted=True, checks=())

    monkeypatch.setattr(
        "embodichain.lab.sim.scene.feasibility.plan_task_mode", bad_query
    )
    with pytest.raises(RuntimeError, match="advanced"):
        source.candidate(0)


def test_legacy_workspace_strategy_cannot_silently_ignore_new_layout():
    import yaml
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/workspace_pose_pick_place/acceptance/scene_first/expansion.yaml"
    )
    value = yaml.safe_load(path.read_text())
    value["scene_sampling"]["strategy"] = "workspace_joint"
    value["scene_sampling"]["targets"] = {"cube": np.eye(4).tolist()}
    with pytest.raises(ValueError, match="does not consume layout"):
        validate_sampling(value)
