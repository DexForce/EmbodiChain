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

"""Grasp metric contracts for fixed attempt budgets and physical retention."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np
import polars as pl
import torch
import yaml
from embodichain.extensions.benchmarks.grasp_pose_generator import cli
from embodichain.extensions.benchmarks.grasp_pose_generator.metrics import summarize
from embodichain.lab.sim import grasp_validation
from embodichain.lab.sim.cfg import MimicComplianceCfg, RigidObjectCfg
from embodichain.lab.sim.grasp_validation import GraspValidator, _result
from embodichain.lab.sim.utility import gripper_geometry
from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.utils.string import resolve_matching_names_values


def _trace(length=361):
    row = np.zeros((length, 19))
    row[:, 3:12] = np.eye(3).reshape(9)
    row[:, 12:15] = [0.09, 1, 1]
    return row


def test_rotating_retained_target_and_directional_drop():
    row = _trace()
    row[320:, 3:12] = np.diag([-1, -1, 1]).reshape(9)
    kept = _result(row, "x", False)
    assert kept["success"] and kept["disturbance"]["rotation_deg"] == 180
    row[335:, 13:15] = 0
    lost = _result(row, "y", False)
    assert lost["hold"]["success"] and not lost["success"]
    assert np.isclose(lost["disturbance"]["drop_s"], 0.35)
    assert _result(row, "x", True)["status"] == "missing"


def test_ranking_missing_evidence_and_zero_padding():
    rows = []
    for rank in (1, 2, 3):
        for direction in ("base", "x", "y"):
            trace = _trace()
            if rank != 3:
                trace[:, 13:15] = 0
            rows.append(
                dict(_result(trace, direction, False), rank=rank, condition=direction)
            )
    case = dict(
        count=3,
        trials=rows,
        object="cube",
        category="boxes",
        gripper="hand",
        algorithm="test",
    )

    def scores():
        return summarize([case])["overall"]["algorithms"]["test"]["metrics"]

    assert scores() == {
        "Success@1": 0,
        "Hit@5": 1,
        "Hit@10": 1,
        "Precision@10": 0.1,
        "Robust@1": 0,
    }
    rows[0].update(_result(_trace(), "base", False))
    rows[1].update(_result(_trace(), "x", False))
    assert scores()["Success@1"] == 1 and scores()["Robust@1"] == 0
    rows.pop()
    incomplete = summarize([case])
    assert incomplete["missing_cases"] == 1
    assert incomplete["by_category"]["boxes"]["algorithms"]["test"]["metrics"] is None


def test_gripper_and_category_scores_weight_objects_equally():
    records = []
    good = [
        dict(_result(_trace(), d, False), rank=1, condition=d)
        for d in ("base", "x", "y")
    ]
    for method in ("first", "second"):
        for object_id, category, seeds in (
            ("easy", "book", 3),
            ("hard", "book", 1),
            ("round", "bowl", 1),
        ):
            for gripper in ("A", "B"):
                for seed in range(seeds):
                    success = object_id == "round" or (object_id, gripper) == (
                        "easy",
                        "A",
                    )
                    rows = good if success else []
                    records.append(
                        dict(
                            object=object_id,
                            category=category,
                            gripper=gripper,
                            algorithm=method,
                            seed=seed,
                            count=int(bool(rows)),
                            trials=rows,
                        )
                    )
    report = summarize(records)

    def success_at_one(group: dict, algorithm: str = "first") -> float:
        return group["algorithms"][algorithm]["metrics"]["Success@1"]

    assert "folds" not in report
    assert success_at_one(report["by_gripper"]["A"]) == pytest.approx(2 / 3)
    assert success_at_one(report["by_category"]["book"]) == 0.25
    books = report["by_gripper_category"]["A"]["book"]
    assert success_at_one(books) == 0.5
    assert books["paired_difference_ci95"]["first minus second"]["Success@1"] == [
        0.0,
        0.0,
    ]
    incomplete = summarize(records[:-1])
    assert not incomplete["complete"]
    assert incomplete["overall"]["algorithms"]["second"]["metrics"] is None
    assert success_at_one(incomplete["by_category"]["book"], "second") == 0.25


def test_inline_suite_reports_gripper_category_and_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = "FrankaPandaHand"
    shift = [25, 0, -45]
    suite = {
        "grippers": [hand],
        "objects": [
            {
                "id": "block",
                "family": "boxes",
                "holdout": True,
                "rigid": {},
                "rotation_shift_deg": shift,
            }
        ],
    }
    config = tmp_path / "suite.yaml"
    config.write_text(yaml.safe_dump(suite))

    def completed_case(spec, method, source, root, out, args):
        assert spec["rotation_shift_deg"] == shift
        trials = [
            dict(_result(_trace(), d, False), rank=1, condition=d)
            for d in ("base", "x", "y")
        ]
        return dict(
            spec,
            algorithm=method,
            physics=args.physics,
            trial_id_prefix="case1",
            count=1,
            finished=True,
            trials=trials,
        )

    monkeypatch.setattr(cli, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(1))
    monkeypatch.setattr(cli, "_run_case", completed_case)
    cli.main(["run", "--suite", str(config), "--output", str(tmp_path)])
    report = json.loads((tmp_path / "report.json").read_text())
    assert (
        report["by_gripper_category"][hand]["boxes"]["algorithms"]["antipodal"][
            "metrics"
        ]["Robust@1"]
        == 1
    )
    trials = pl.read_parquet(tmp_path / "trials" / "case1.parquet")
    assert trials.select("gripper", "category", "object").unique().rows() == [
        (hand, "boxes", "block")
    ]
    assert "object_holdout" not in trials.columns
    markdown = (tmp_path / "report.md").read_text()
    assert hand in markdown and "boxes" in markdown and "both_new" not in markdown

    case = tmp_path / "block.yaml"
    case.write_text(
        yaml.safe_dump(
            {
                "object": "block",
                "gripper": hand,
                "rigid": {},
                "rotation_shift_deg": shift,
            }
        )
    )
    poses = tmp_path / "candidates.jsonl"
    poses.write_text(json.dumps({"T_object_tcp": np.eye(4).tolist()}))
    cli.main(
        [
            "validate",
            "--case",
            str(case),
            "--poses",
            str(poses),
            "--output",
            str(tmp_path),
        ]
    )


def test_visual_preset_dispatches_one_case_per_gripper_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = files("embodichain.extensions.benchmarks.grasp_pose_generator")
    visual = yaml.safe_load((fixtures / "visual_check.yaml").read_text())[
        "visual_check"
    ]
    suite = yaml.safe_load((fixtures / visual["suite"]).read_text())
    suite_dir = tmp_path / "config"
    suite_dir.mkdir()
    bundle = tmp_path / "extracted" / "GraspBenchmark"
    directories = {
        "GraspBenchmark/objects_subset": bundle / "objects_subset",
        "GraspBenchmark/grippers": bundle / "grippers",
    }
    resolve_asset = Mock(side_effect=lambda reference: str(directories[reference]))
    monkeypatch.setattr("embodichain.data.get_data_path", resolve_asset)
    suite.update(
        seeds=[17, 23],
        algorithms={"external": "poses.jsonl", "other": None},
    )
    for hand in suite["grippers"]:
        hand_dir = bundle / "grippers" / hand
        hand_dir.mkdir(parents=True)
        (hand_dir / f"{hand}.urdf").write_text('<robot name="hand"/>')
    for entry in suite["objects"]:
        category = entry["category"]
        entry["placements"] = {"nominal": {}, "other": {"uid": "unused"}}
        directory = bundle / "objects_subset" / category
        model = visual["objects"][category]
        package = directory / model
        package.mkdir(parents=True)
        (package / "root.usda").write_text("#usda 1.0\n")
    full = cli.suite_objects(suite, suite_dir)
    assert {obj["category"]: obj["id"] for obj in full} == visual["objects"]
    for directory in (bundle / "objects_subset").iterdir():
        (directory / "unused-uuid/unselected/simready_usd").mkdir(parents=True)
    candidate = {"T_object_tcp": np.eye(4).tolist()}
    (suite_dir / "poses.jsonl").write_text(json.dumps(candidate))
    (suite_dir / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    config = tmp_path / "visual.yaml"
    config.write_text(
        yaml.safe_dump({"visual_check": visual | {"suite": "config/suite.yaml"}})
    )
    expected = {
        (hand, category, model)
        for hand in suite["grippers"]
        for category, model in visual["objects"].items()
    }
    assert len(expected) == 35
    attempts = []

    def run_case(spec, method, source, root, out, args):
        attempts.append((spec, method, source, args.num_envs, args.headless))
        missing = args.headless and len(attempts) == 1
        return dict(
            spec,
            physics=args.physics,
            algorithm=method,
            case_config=spec,
            trial_id_prefix=str(len(attempts)),
            count=int(missing),
            trials=(
                [dict(rank=1, condition="base", status="missing", reason="sensor lost")]
                if missing
                else []
            ),
            finished=not missing,
        )

    monkeypatch.setattr(cli, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(1))
    monkeypatch.setattr(cli, "_run_case", run_case)
    command = ["run", "--suite", str(config), "--output", str(tmp_path / "results")]
    with pytest.raises(SystemExit):
        cli.main(command + ["--num-envs", "3"])
    assert not attempts
    for headless in (False, True):
        attempts.clear()
        if headless:
            with pytest.raises(SystemExit):
                cli.main(command + ["--headless"])
        else:
            cli.main(command)
        assert len(attempts) == 35
        assert {
            (s["gripper"], s["category"], s["object"]) for s, *_ in attempts
        } == expected
        for spec, method, source, num_envs, actual_headless in attempts:
            assert num_envs == 1 and actual_headless is headless
            hand = spec["gripper"]
            assert (
                Path(spec["gripper_urdf"])
                == bundle / "grippers" / hand / f"{hand}.urdf"
            )
            assert (spec["seed"], spec["placement"], method) == (
                17,
                "nominal",
                "external",
            )
            assert source == [candidate]
            assert spec["visual_check"] == {"sample_count": visual["sample_count"]}
            assert (
                spec["root_from_tcp"]
                == suite["grippers"][spec["gripper"]]["root_from_tcp"]
            )
            assert Path(spec["rigid"]["source"]["file_path"]).is_file()
    assert {call.args[0] for call in resolve_asset.call_args_list} == set(directories)
    with pytest.raises(ValueError, match="book/absent"):
        cli.suite_objects(suite, suite_dir, representatives={"book": "absent"})


def test_visual_worker_executes_only_top_base_and_isolates_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = dict(
        object="block",
        category="blocks",
        gripper="FrankaPandaHand",
        seed=0,
        placement="default",
        rigid={"uid": "target"},
    )
    visual = spec | {"visual_check": {"sample_count": 2000}}
    options = SimpleNamespace(
        physics="default", num_envs=1, resume=True, headless=True, command="run"
    )
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.02
    candidates = [{"T_object_tcp": pose.tolist()} for pose in (first, second)]
    validator = Mock()
    validator.initial = torch.eye(4)
    validator.validate.return_value = [{"status": "ok", "success": True}]
    monkeypatch.setattr(
        grasp_validation, "GraspValidator", lambda *args, **kwargs: validator
    )
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)

    def run(case_spec, source):
        return cli._run_case(case_spec, "external", source, tmp_path, tmp_path, options)

    result = run(visual, candidates)
    assert result["finished"] and result["count"] == 1
    poses, conditions = validator.validate.call_args.args
    np.testing.assert_array_equal(poses, first[None])
    assert conditions == ["base"]
    assert [(row["rank"], row["condition"]) for row in result["trials"]] == [
        (1, "base")
    ]
    assert run(visual, candidates)["trial_id_prefix"] == result["trial_id_prefix"]
    empty = run(visual, [])
    assert empty["finished"] and not empty["trials"] and empty["count"] == 0
    validator.validate.assert_called_once()
    full = run(spec, candidates)
    assert full["trial_id_prefix"] != result["trial_id_prefix"]
    assert len(full["trials"]) == 6
    assert {row["condition"] for row in full["trials"]} == {"base", "x", "y"}
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        '<robot><link><collision><geometry><mesh filename="pad.obj"/>'
        "</geometry></collision></link></robot>"
    )
    pad = tmp_path / "pad.obj"
    packaged = visual | {"gripper_urdf": "hand.urdf"}
    pad.write_text("v 0 0 0\n")
    before = run(packaged, candidates)
    pad.write_text("v 0.001 0 0\n")
    after = run(packaged, candidates)
    assert before["trial_id_prefix"] != after["trial_id_prefix"]


def test_dataset_categories_expand_extracted_usd_packages(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    root = tmp_path / "desk_subset"
    source_paths = {}
    for category, model in (("book", "atlas"), ("book", "novel"), ("cup", "mug")):
        path = (
            root / category / f"{model}-uuid" / model / "simready_usd" / f"{model}.usda"
        )
        if category == "cup":
            path = root / category / model / f"{model}.usda"
        path.parent.mkdir(parents=True)
        stage = Usd.Stage.CreateNew(str(path))
        cube = UsdGeom.Cube.Define(stage, "/Target")
        cube.CreateSizeAttr(0.04)
        body = cube.GetPrim()
        stage.SetDefaultPrim(body)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(body)
        if model == "mug":
            profiles = {}
            for backend in ("PhysX", "Newton"):
                variants = body.GetVariantSets().AddVariantSet(backend)
                for choice in ("Enabled", "Disabled"):
                    variants.AddVariant(choice)
                profiles[backend] = {
                    "prim": "/Target",
                    "variantSetName": backend,
                    "activateOption": "Enabled",
                }
            stage.GetRootLayer().customLayerData = {
                "SimReady_Metadata": {"Variants": {"Physics": profiles}}
            }
        stage.GetRootLayer().Save()
        source_paths[model] = (category, path)

    suite = {
        "dataset_root": "desk_subset",
        "objects": [
            {
                "category": category,
                "rotation_shift_deg": [0, 90, 0] if category == "book" else [0, 0, 0],
                "rigid": {
                    "body_type": "dynamic",
                    "asset_physics_mode": "overlay",
                    "attrs": {"mass_props": {"mass": mass}},
                },
            }
            for category, mass in (("book", 0.12), ("cup", 0.08))
        ],
    }
    objects = cli.suite_objects(suite, tmp_path, physics="newton")
    assert (
        cli.suite_objects(
            suite | {"dataset_root": str(root)},
            tmp_path / "elsewhere",
            physics="newton",
        )
        == objects
    )
    assert {obj["id"] for obj in objects} == set(source_paths)
    for obj in objects:
        category, path = source_paths[obj["id"]]
        assert obj["category"] == category
        assert obj["rotation_shift_deg"] == (
            [0, 90, 0] if category == "book" else [0, 0, 0]
        )
        cfg = RigidObjectCfg.from_dict(obj["rigid"])
        assert Path(cfg.source.file_path) == path
        assert cfg.attrs.mass_props.mass == (0.12 if category == "book" else 0.08)
        if obj["id"] == "mug":
            assert cfg.source.variants == {
                "/Target": {"Newton": "Enabled", "PhysX": "Disabled"}
            }


def test_dexuni_contact_evidence_uses_distance_only_for_unmeasured_rows():
    validator = GraspValidator.__new__(GraspValidator)
    validator._geometry_contacts = True
    validator.target_id = torch.tensor([[1]])
    validator.pad_ids = torch.tensor([[2, 3]])
    validator.support_id = torch.tensor([[4]])
    data = dict(
        user_ids=torch.tensor([[[1, 2], [3, 1], [1, 4]]]),
        is_valid=torch.ones(1, 3, dtype=torch.bool),
        impulse_valid=torch.tensor([[True, False, False]]),
        impulse=torch.tensor([[0.0, 0.0, 0.0]]),
        distance=torch.tensor([[-0.001, 0.0, 0.001]]),
    )
    validator.sensor = SimpleNamespace(
        update=lambda: None,
        get_data=lambda: data,
        get_latest_contact_loss_mask=lambda: torch.tensor([False]),
    )
    fingers, support, lost = validator._contacts()
    assert fingers.tolist() == [[False, True]]
    assert not support.item() and not lost.item()
    data["distance"][0, 1] = float("nan")
    assert validator._contacts()[2].item()


@pytest.mark.parametrize("keeps_tipping", [False, True])
def test_targets_settle_in_the_same_half_second(keeps_tipping: bool) -> None:
    validator = GraspValidator.__new__(GraspValidator)
    elapsed = 0

    def advance(step: int) -> None:
        nonlocal elapsed
        elapsed += step

    def poses(*, to_matrix: bool) -> torch.Tensor:
        result = torch.eye(4).repeat(2, 1, 1)
        # Each arena moves and returns within a different observation window.
        if elapsed == 120:
            result[0, 0, 3] = 0.01
        if elapsed == 170 or (keeps_tipping and elapsed % 50 == 20):
            result[1, 1:3, 1:3] = torch.tensor([[0, -1], [1, 0]])
        return result

    validator.sim = SimpleNamespace(num_envs=2, device="cpu", update=advance)
    validator.obj = SimpleNamespace(get_local_pose=poses)
    if keeps_tipping:
        with pytest.raises(ValueError, match="did not settle"):
            validator._settled_pose()
        assert elapsed == 500
    else:
        torch.testing.assert_close(validator._settled_pose(), torch.eye(4))
        assert 220 <= elapsed <= 250


def test_rotation_shift_composes_in_arena_axes_and_clears_table() -> None:
    validator = GraspValidator.__new__(GraspValidator)
    validator.sim = SimpleNamespace(device="cpu", num_envs=2, update=Mock())
    poses = torch.eye(4).repeat(2, 1, 1)
    # Arena 0 starts at Rz(90°) and arena 1 at Ry(90°).
    poses[0, :3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    poses[1, :3, :3] = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    poses[:, :3, 3] = torch.tensor([[0.3, 0.1, 0.4], [-0.2, 0.2, 0.6]])
    validator.obj = Mock()
    validator.obj.get_local_pose.return_value = poses.clone()
    shape_pose = torch.eye(4)
    shape_pose[:2, 3] = torch.tensor([0.01, -0.03])
    validator.shapes = [
        SimpleNamespace(
            shape_type=SimpleNamespace(name="BOX"),
            local_pose=shape_pose,
            half_extents=torch.tensor([0.02, 0.04, 0.07]),
        )
    ]
    rotation = grasp_validation._rotation_shift_matrix([90, 0, 90])
    validator._place_rotated_target(rotation)
    placed = validator.obj.set_local_pose.call_args.args[0]
    torch.testing.assert_close(placed[:, :2, 3], poses[:, :2, 3])
    # After turns about arena X then Z, local +X and +Y point up, respectively.
    expected_rot = torch.tensor(
        [[[0, 0, 1], [0, -1, 0], [1, 0, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(placed[:, :3, :3], expected_rot)
    torch.testing.assert_close(placed[:, 2, 3], torch.tensor([0.01, 0.07]))
    separated = validator.obj.set_local_pose.call_args_list[0].args[0]
    assert (validator._clearance(separated) > 0).all()
    validator.sim.update.assert_called_once()
    linear, angular = validator.obj.set_velocity.call_args.args
    assert not linear.any() and not angular.any()


def test_rotation_shift_requires_three_finite_angles() -> None:
    for invalid in ([0, 90], [0, float("nan"), 0]):
        with pytest.raises(ValueError, match="rotation_shift_deg"):
            grasp_validation._rotation_shift_matrix(invalid)


@pytest.fixture
def trial_validator() -> GraspValidator:
    validator = GraspValidator.__new__(GraspValidator)
    validator._show_trials = False
    validator.close_wait_steps = 20
    validator.initial = validator.root_tcp = validator.root_contact = torch.eye(4)
    validator.axes = [0, 1, 2]
    validator.opened = torch.zeros(4)
    validator.closed = torch.tensor([0.0, 0.0, 0.0, 1.0])
    validator.hand, validator.obj, validator.sensor = Mock(), Mock(), Mock()
    validator.cfg = SimpleNamespace(root_link="hand")
    validator.com = torch.zeros(3)
    validator.hand.get_link_pose.return_value = validator.initial[None]
    validator.obj.get_local_pose.return_value = validator.initial[None]
    validator._clearance = lambda pose: torch.ones(1)
    no_contact = torch.zeros(1, dtype=torch.bool)
    validator._contacts = lambda: (
        torch.ones(1, 2, dtype=torch.bool),
        no_contact,
        no_contact,
    )
    validator.sim = Mock(
        num_envs=1,
        device="cpu",
        viewer=None,
        arena_offsets=torch.zeros(1, 3),
        is_viewer_open=False,
    )
    return validator


def test_close_wait_holds_closed_target_before_lift_and_shifts_disturbance(
    trial_validator: GraspValidator,
) -> None:
    validator = trial_validator
    commands = []
    validator.sim.update.side_effect = lambda step: commands.append(
        validator.hand.set_qpos.call_args.args[0][0]
    )

    (trial,) = validator.validate(np.eye(4)[None], ["x"])
    # Each row is the three stage positions followed by the finger command.
    assert commands[80 - 1].tolist() == pytest.approx([0, 0, 0.08, 0.5])
    np.testing.assert_allclose(
        torch.stack(commands[100 - 1 : 120]), np.tile([0, 0, 0.08, 1], (21, 1))
    )
    assert commands[160 - 1].tolist() == pytest.approx([0, 0, 0.13, 1])
    assert commands[530 - 1].tolist() == pytest.approx([0.03, 0, 0.18, 1])
    assert len(commands) == 560
    assert trial["hold"]["duration_s"] == pytest.approx(3.0)
    assert trial["disturbance"]["duration_s"] == pytest.approx(0.6)
    validator.sim.open_viewer.assert_not_called()


def test_window_opens_at_first_pregrasp_and_is_reused(
    trial_validator: GraspValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = trial_validator
    validator._show_trials = True
    viewer = Mock()

    def open_viewer() -> bool:
        # The first visible frame has the hand open, 8 cm before the candidate.
        (hand_pose,) = validator.hand.set_local_pose.call_args.args
        torch.testing.assert_close(hand_pose[0, :3, 3], torch.tensor([0, 0, -0.08]))
        (joints,) = validator.hand.set_qpos.call_args.args
        torch.testing.assert_close(joints[0], validator.opened)
        (object_pose,) = validator.obj.set_local_pose.call_args.args
        torch.testing.assert_close(object_pose[0], validator.initial)
        validator.sim.viewer = viewer
        validator.sim.is_viewer_open = True
        return True

    validator.sim.open_viewer.side_effect = open_viewer
    monkeypatch.setattr(grasp_validation.time, "sleep", lambda seconds: None)
    for _ in range(2):
        validator.validate(np.eye(4)[None], ["base"])
    validator.sim.open_viewer.assert_called_once()
    viewer.set_look_at.assert_called_once()


@pytest.mark.parametrize(
    ("options", "renderer"), [({}, "no-render"), ({"headless": False}, "fast-rt")]
)
def test_validator_starts_without_a_window_and_selects_the_requested_renderer(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, bool], renderer: str
) -> None:
    manager = Mock(side_effect=RuntimeError("stop before native startup"))
    monkeypatch.setattr(grasp_validation, "SimulationManager", manager)
    with pytest.raises(RuntimeError, match="stop before native startup"):
        GraspValidator(RigidObjectCfg(uid="target"), "FrankaPandaHand", **options)
    (cfg,) = manager.call_args.args
    assert cfg.headless
    assert cfg.render_cfg.renderer == renderer


def test_suite_gripper_ids_and_invalid_calibration(tmp_path: Path) -> None:
    hand = "FrankaPandaHand"
    assert cli._suite_grippers([hand]) == {hand: {}}
    reflection = np.diag([-1, 1, 1, 1]).tolist()
    with pytest.raises(ValueError, match="root_from_tcp"):
        cli._suite_grippers({hand: {"root_from_tcp": reflection}})
    with pytest.raises(ValueError):
        cli._suite_grippers({hand: {"root_from_contact_frame": np.eye(4).tolist()}})
    for invalid in (-1, float("inf"), True):
        with pytest.raises(ValueError, match="stiffness"):
            cli._suite_grippers({hand: {"stiffness": invalid}})
    names = resolve_gripper_defaults(hand)["control_parts"]["eef"]
    for invalid in (
        {"unknown_joint": 0.0},
        dict.fromkeys(names, True),
        dict.fromkeys(names, float("nan")),
    ):
        with pytest.raises(ValueError, match="close_qpos"):
            cli._suite_grippers({hand: {"close_qpos": invalid}})
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"grippers:\n  {hand}: {{}}\n  {hand}: {{}}\n")
    with pytest.raises(ValueError, match=f"Duplicate suite key '{hand}'"):
        cli.main(["run", "--suite", str(suite), "--output", str(tmp_path)])


def test_suite_close_wait_steps_defaults_to_zero_and_rejects_invalid_counts() -> None:
    assert cli._close_wait_steps({}) == 0
    for invalid in (-1, 2.5, True):
        with pytest.raises(ValueError, match="close_wait_steps"):
            cli._close_wait_steps({"close_wait_steps": invalid})


def test_centered_block_example_keeps_palm_above_block() -> None:
    fixtures = files("embodichain.extensions.benchmarks.grasp_pose_generator")
    case = yaml.safe_load((fixtures / "block.yaml").read_text())
    centered = json.loads((fixtures / "candidates.jsonl").read_text().splitlines()[2])
    hand = resolve_gripper_defaults(case["gripper"])
    geometry = gripper_geometry.build_gripper_closing_geometry(hand, sample_count=2)
    object_from_contact = (
        centered["T_object_tcp"]
        @ np.linalg.inv(hand["default_root_from_tcp"])
        @ hand["root_from_contact_frame"]
    )
    palm_z = (
        geometry.body[0].numpy() @ object_from_contact[2, :3]
        + object_from_contact[2, 3]
    )
    assert palm_z.min() > case["rigid"]["shape"]["size"][2] / 2


def test_case_uses_tcp_for_geometry_and_drive_settings_for_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand = "TianjiMarvinGripper"
    tcp = np.eye(4)
    tcp[0, 3] = 0.03
    calibration = {
        "variant": "right",
        "root_from_tcp": tcp.tolist(),
        "stiffness": 0.0,
        "close_qpos": resolve_gripper_defaults(hand, variant="right")["joint_commands"][
            "grasp"
        ],
    }
    urdf = tmp_path / "hand.urdf"
    urdf.write_text('<robot name="local"/>')
    spec = dict(
        gripper=hand,
        object="block",
        category="block",
        seed=0,
        placement="default",
        rigid={"uid": "target"},
        gripper_urdf="hand.urdf",
        close_wait_steps=25,
        rotation_shift_deg=[0, 90, 0],
        **cli._suite_grippers({hand: calibration})[hand],
    )
    options = SimpleNamespace(
        physics="default", num_envs=1, resume=False, headless=True, command="run"
    )
    validator = Mock()
    geometry = Mock(side_effect=RuntimeError("stop before mesh sampling"))
    monkeypatch.setattr(grasp_validation, "GraspValidator", validator)
    monkeypatch.setattr(gripper_geometry, "build_gripper_collision_geometry", geometry)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    with pytest.raises(RuntimeError, match="stop before mesh sampling"):
        cli._run_case(spec, "antipodal", None, tmp_path, tmp_path, options)
    assert validator.call_args.kwargs["close_wait_steps"] == 25
    assert validator.call_args.kwargs["rotation_shift_deg"] == [0, 90, 0]
    assert validator.call_args.kwargs["stiffness"] == 0.0
    assert validator.call_args.kwargs["close_qpos"] == calibration["close_qpos"]
    assert validator.call_args.kwargs["variant"] == "right"
    assert validator.call_args.kwargs["gripper_urdf"] == urdf
    np.testing.assert_allclose(validator.call_args.kwargs["root_from_tcp"], tcp)
    assert geometry.call_args.args[0]["root_link"] == "right_hand_base_link"
    assert geometry.call_args.args[0]["fpath"] == str(urdf)
    assert geometry.call_args.args[0]["mesh_paths"] == {}
    assert geometry.call_args.kwargs.keys() == {"root_from_tcp"}
    np.testing.assert_allclose(geometry.call_args.kwargs["root_from_tcp"], tcp)


@pytest.mark.parametrize(
    ("hand", "physics", "master"),
    [
        ("FrankaPandaHand", "default", "fr3_finger_joint1"),
        ("DHPGI14080", "default", "gripper_finger1_joint_1"),
        ("RobotiqArg2F85", "default", "finger_joint"),
        ("RobotiqArg2F140", "default", "finger_joint"),
        ("CobotMagicV100Gripper", "default", "joint7"),
        ("PikaGripper", "default", "FINGER1_JOINT"),
        ("TianjiMarvinGripper", "default", "LEFT_HAND_FINGER_1"),
        ("RobotiqArg2F85", "newton", None),
    ],
)
def test_suite_stages_hand_with_backend_coupling_and_separate_drive_gains(
    monkeypatch: pytest.MonkeyPatch,
    hand: str,
    physics: str,
    master: str | None,
) -> None:
    fixtures = files("embodichain.extensions.benchmarks.grasp_pose_generator")
    suite = yaml.load((fixtures / "suite.yaml").read_text(), Loader=cli._SuiteLoader)
    options = cli._suite_grippers(suite["grippers"])[hand]
    defaults = resolve_gripper_defaults(hand, variant=options.get("variant"))
    manager = Mock()
    add_robot = manager.return_value.add_robot
    add_robot.side_effect = RuntimeError("stop before spawning robot")
    monkeypatch.setattr(grasp_validation, "SimulationManager", manager)
    with pytest.raises(RuntimeError, match="stop before spawning robot"):
        GraspValidator(RigidObjectCfg(uid="target"), hand, physics=physics, **options)
    (cfg,) = add_robot.call_args.args
    assert (cfg.mimic_compliance or {}) == (
        {master: MimicComplianceCfg(0.0, 0.0)} if master else {}
    )
    fingers = defaults["control_parts"]["eef"]
    if hand == "DHPGI14080":
        assert cfg.joint_commands["grasp"] == dict.fromkeys(fingers, 0.04)
        assert defaults["joint_commands"]["grasp"] == dict.fromkeys(fingers, 0.024)
    stage = ["stage_0", "stage_1", "stage_2"]
    for prop, hand_value, stage_value in (
        ("stiffness", suite["grippers"][hand]["stiffness"], 30000.0),
        ("damping", defaults["joint_drive_props"]["damping"], 350.0),
        ("max_effort", defaults["joint_drive_props"]["max_effort"], 2000.0),
    ):
        _, names, values = resolve_matching_names_values(
            getattr(cfg.joint_drive_props, prop), [*fingers, *stage]
        )
        assert names == [*fingers, *stage]
        assert values == [hand_value] * len(fingers) + [stage_value] * len(stage)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ((0.04, 0.03), "mimic"),
        ((0.041, 0.041), "limits"),
    ],
)
def test_dh_close_qpos_respects_mimic_and_travel(
    monkeypatch: pytest.MonkeyPatch, targets: tuple[float, float], message: str
) -> None:
    hand = "DHPGI14080"
    names = resolve_gripper_defaults(hand)["control_parts"]["eef"]
    monkeypatch.setattr(grasp_validation, "SimulationManager", Mock())
    with pytest.raises(ValueError, match=message):
        GraspValidator(
            RigidObjectCfg(uid="target"), hand, close_qpos=dict(zip(names, targets))
        )


@pytest.mark.gpu
@pytest.mark.subprocess_sim
@pytest.mark.parametrize("physics", ["default", "newton"])
def test_gpu_cli_isolates_missed_and_held_grasps(tmp_path: Path, physics: str) -> None:
    """Two misses and a centred grasp share a native nine-arena batch."""
    fixtures = files("embodichain.extensions.benchmarks.grasp_pose_generator")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "embodichain",
            "benchmark",
            "grasp-pose-generator",
            "validate",
            "--physics",
            physics,
            "--case",
            str(fixtures / "block.yaml"),
            "--poses",
            str(fixtures / "candidates.jsonl"),
            "--num-envs",
            "9",
            "--output",
            str(tmp_path),
        ],
        check=True,
        timeout=120,
    )
    (checkpoint,) = tmp_path.glob("*.json")
    case = json.loads(checkpoint.read_text())
    assert case["protocol"]["physics"] == physics
    assert case["protocol"].get("mimic_coupling") == (
        "hard" if physics == "default" else None
    )
    assert case["protocol"]["close_wait_steps"] == 0
    assert case["protocol"]["solver"] == (
        "dexuni" if physics == "newton" else "default"
    )
    assert case["finished"]
    trials = {(r["rank"], r["condition"]): r for r in case["trials"]}
    assert len(trials) == 9
    for direction in ("base", "x", "y"):
        assert not trials[1, direction]["success"]
        assert not trials[2, direction]["success"]
        centred = trials[3, direction]
        assert centred["hold"]["entered"]
        if physics == "default":
            assert centred["success"]
    # The coupled hand can slip after pickup. Each condition runs from reset.
    assert any(trials[3, direction]["success"] for direction in ("base", "x", "y"))
    for direction in ("x", "y"):
        trial = trials[3, direction]
        assert np.isfinite(trial["peak_acceleration_m_s2"])
        assert (
            1.5
            < trial["peak_axis_acceleration_m_s2"]
            <= trial["peak_acceleration_m_s2"]
        )
        if physics == "default":
            assert trial["peak_acceleration_m_s2"] < 2.5


def test_visual_report_distinguishes_observed_and_unexecuted_grasps(
    tmp_path: Path,
) -> None:
    from embodichain.extensions.benchmarks.grasp_pose_generator import _reporting

    case = dict(category="book", object="atlas", count=1, finished=True)
    base = dict(rank=1, condition="base", status="ok")
    cases = [
        dict(case, gripper="A", trials=[dict(base, success=True)]),
        dict(
            case, gripper="B", trials=[dict(base, success=False, failure_stage="lift")]
        ),
        dict(case, gripper="C", count=0, trials=[]),
    ]
    protocol = {"physics": "default"}
    assert _reporting._visual_report(cases, protocol)["complete"]
    cases.extend(
        [
            dict(
                case,
                gripper="D",
                finished=False,
                trials=[dict(base, status="missing", reason="unavailable contact")],
            ),
            dict(case, gripper="E", count=0, trials=[], error="asset missing"),
        ]
    )
    report = _reporting._visual_report(cases, protocol)
    assert not report["complete"]
    assert report["counts"] == {
        "success": 1,
        "failed": 1,
        "no_candidate": 1,
        "missing": 1,
        "error": 1,
    }
    assert report["attempts"] == 3
    assert [row["attempts"] for row in report["cases"]] == [1, 1, 0, 1, 0]
    _reporting._write_visual_report(tmp_path, report)
    markdown = (tmp_path / "report.md").read_text()
    for gripper, outcome in (
        ("A", "success | —"),
        ("B", "failed | lift"),
        ("C", "no candidate | —"),
        ("D", "missing | unavailable contact"),
        ("E", "error | asset missing"),
    ):
        assert f"| {gripper} | book | atlas | {outcome} |" in markdown
