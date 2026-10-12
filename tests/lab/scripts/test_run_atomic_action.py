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

import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch
import yaml

from embodichain.lab.scripts import run_atomic_action as launcher
from embodichain.lab.scripts._atomic_action.config import (
    SCENES,
    _load_config,
    _recipes,
    _scene_args,
)
from embodichain.lab.scripts._atomic_action.expansion import _augment, _template
from embodichain.lab.scripts._atomic_action.support import (
    create_affordance_sampling_context,
)
from embodichain.lab.sim.atomic_actions import TimedTrajectory

CONFIG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "embodichain/lab/scripts/_atomic_action/configs"
)


@pytest.mark.parametrize("skill", SCENES)
def test_packaged_scene_configs_use_production_decoder(skill: str) -> None:
    cfg = _load_config(CONFIG_ROOT / f"{skill}.yaml")
    recipes = _recipes(cfg)
    expected = 4 if skill == "move_joint" else 16
    assert len(recipes) == expected
    assert recipes[0] == (0, 0)
    assert cfg["expansion"]["trajectory"]["spatial"]["operators"] == [
        "joint_residual",
        "via_points",
    ]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"unknown": True}, "unsupported fields"),
        ({"physics": None}, "declare physics"),
        ({"runtime": {"num_envs": True}}, "positive integer"),
        ({"runtime": {"unknown": 1}}, "unsupported fields"),
        (
            {"options": {"custom_approach_direction": [0, 0, 0], "approach": "custom"}},
            "nonzero",
        ),
        (
            {
                "expansion": {
                    "trajectory": {
                        "variants_per_family": 4,
                        "spatial": {"enabled": False},
                    }
                }
            },
            "spatial.enabled",
        ),
        (
            {
                "expansion": {
                    "trajectory": {
                        "spatial": {"enabled": True, "dedup_tolerance": 0.01}
                    }
                }
            },
            "below joint_offset_scale",
        ),
        ({"output": {"save_video": "yes"}}, "boolean"),
    ],
)
def test_config_rejects_invalid_contract_before_simulation(
    tmp_path: Path, patch: dict, message: str
) -> None:
    raw = yaml.safe_load((CONFIG_ROOT / "pickup.yaml").read_text())
    raw.update(patch)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        _load_config(path)


def test_relative_output_path_and_cli_override_precedence(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_ROOT / "pickup.yaml").read_text())
    raw["output"]["directory"] = "outputs"
    raw["runtime"]["headless"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    parser = launcher._create_parser()
    cfg, _ = launcher._resolve_config(parser.parse_args(["--config", str(path)]))
    assert cfg["runtime"]["headless"] is True
    assert cfg["output"]["directory"] == str(tmp_path / "outputs")
    cfg, indices = launcher._resolve_config(
        parser.parse_args(
            [
                "--config",
                str(path),
                "--no-headless",
                "--num-envs",
                "2",
                "--recipe-indices",
                "2",
                "4",
            ]
        )
    )
    assert cfg["runtime"]["headless"] is False
    assert cfg["runtime"]["num_envs"] == 2
    assert indices == (2, 4)


def test_cli_physics_cannot_switch_file_owned_backend() -> None:
    args = launcher._create_parser().parse_args(
        ["--config", str(CONFIG_ROOT / "pickup.yaml"), "--physics", "newton"]
    )
    with pytest.raises(ValueError, match="confirm"):
        launcher._resolve_config(args)


@pytest.mark.parametrize("planner", ("trapezoidal", "toppra", "curobo"))
def test_configured_pour_retains_native_planner_sampling(planner: str) -> None:
    from embodichain.lab.scripts._atomic_action.pour import _create_pick_motion_policy

    policy = _create_pick_motion_policy(planner, fixed_sample_count=False)
    assert policy.plan_opts is None
    assert policy.strategy == "motion_gen"


def test_affordance_branch_identity_is_independent_of_slots() -> None:
    cfg = _load_config(CONFIG_ROOT / "pickup.yaml")
    args = _scene_args(cfg, (8, 2, 4), 7)
    sampling = create_affordance_sampling_context(args)
    assert [sampling.branch_for(row, row_index=row) for row in range(3)] == [2, 0, 1]
    other = create_affordance_sampling_context(_scene_args(cfg, (8,), 0))
    assert torch.equal(
        sampling.generator("grasp").get_state(), other.generator("grasp").get_state()
    )


def _compiled(rows: int = 3) -> SimpleNamespace:
    # Two actions repeat their join at sample 40 with zero elapsed time.
    horizon = 71
    positions = torch.zeros(rows, horizon, 3)
    positions[:, :40, 0] = torch.linspace(0, 0.5, 40)
    positions[:, 40:, 0] = torch.linspace(0.5, 0.0, horizon - 40)
    positions[:, :, 2] = 0.04  # gripper is never an editable arm joint
    intervals = torch.full((rows, horizon), 0.01)
    intervals[:, 0] = 0
    intervals[:, 40] = 0
    segments = (
        (
            SimpleNamespace(name="approach", start=0, stop=25),
            SimpleNamespace(name="close", start=25, stop=40),
        ),
        (
            SimpleNamespace(name="approach", start=0, stop=20),
            SimpleNamespace(name="release", start=20, stop=31),
        ),
    )
    plans = tuple(
        SimpleNamespace(
            skill_id=skill, segments=group, diagnostics=SimpleNamespace(metadata={})
        )
        for skill, group in zip(("pick_up", "place"), segments)
    )

    def segment(index: int, name: str) -> SimpleNamespace:
        value = next(value for value in segments[index] if value.name == name)
        offset = 40 if index else 0
        return SimpleNamespace(
            name=name, start=value.start + offset, stop=value.stop + offset
        )

    return SimpleNamespace(
        trajectory=TimedTrajectory(
            positions, None, None, intervals, torch.arange(rows)
        ),
        action_plans=plans,
        segment=segment,
    )


class _Robot:
    joint_names = ("arm_0", "arm_1", "hand")

    def get_joint_ids(self, name: str) -> list[int]:
        assert name == "arm"
        return [0, 1]

    def get_qpos_limits(self) -> torch.Tensor:
        return torch.tensor([[-3.0, 3.0], [-3.0, 3.0], [0.0, 0.05]]).expand(3, -1, -1)


def test_both_spatial_operators_keep_contacts_hands_clock_and_action_joins() -> None:
    cfg = _load_config(CONFIG_ROOT / "place.yaml")
    args = _scene_args(cfg, (0, 1, 2), 0)
    compiled = _compiled()
    expanded, records = _augment(compiled, _Robot(), args)
    assert records[0]["operator"] == "none"
    assert {record["operator"] for record in records[1:]} == {
        "joint_residual",
        "via_points",
    }
    assert torch.equal(expanded.dt, compiled.trajectory.dt)
    assert torch.equal(expanded.positions[0], compiled.trajectory.positions[0])
    assert torch.equal(
        expanded.positions[:, :, 2], compiled.trajectory.positions[:, :, 2]
    )
    for left, right in ((25, 40), (60, 71)):
        assert torch.equal(
            expanded.positions[:, left:right],
            compiled.trajectory.positions[:, left:right],
        )
    for endpoint in (0, 24, 39, 40, 59, 60, 70):
        assert torch.equal(
            expanded.positions[:, endpoint], compiled.trajectory.positions[:, endpoint]
        )
    assert not torch.equal(
        expanded.positions[1, :25, :2], compiled.trajectory.positions[1, :25, :2]
    )
    assert not torch.equal(
        expanded.positions[2, :25, :2], compiled.trajectory.positions[2, :25, :2]
    )


def test_trajectory_variant_is_stable_when_recipe_moves_to_another_batch() -> None:
    cfg = _load_config(CONFIG_ROOT / "place.yaml")
    compiled = _compiled()
    together, _ = _augment(compiled, _Robot(), _scene_args(cfg, (0, 1, 2), 0))
    isolated, _ = _augment(_compiled(1), _Robot(), _scene_args(cfg, (2,), 9))
    assert torch.equal(together.positions[2], isolated.positions[0])


def test_spatial_variants_share_one_grasp_reference_per_branch() -> None:
    cfg = _load_config(CONFIG_ROOT / "place.yaml")
    compiled = _compiled()
    # Model different feasible IK/grasp rankings on otherwise equal slots.
    compiled.trajectory.positions[1, :, 1] = 0.3
    compiled.trajectory.positions[2, :, 1] = 0.6
    expanded, records = _augment(compiled, _Robot(), _scene_args(cfg, (0, 1, 2), 0))
    assert [record["reference_row"] for record in records] == [0, 0, 0]
    assert torch.equal(
        expanded.positions[:, 25:40],
        compiled.trajectory.positions[0:1, 25:40].expand(3, -1, -1),
    )


def test_zero_time_position_change_is_rejected() -> None:
    compiled = _compiled()
    compiled.trajectory.positions[0, 40, 0] += 0.1
    with pytest.raises(ValueError, match="zero time"):
        _template(compiled, 0, _Robot.joint_names, (0, 1), "place", 0)


def test_missing_distinct_variants_fail_without_reusing_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.scripts._atomic_action import expansion

    cfg = _load_config(CONFIG_ROOT / "place.yaml")
    fake = SimpleNamespace(
        variants=(SimpleNamespace(spatial_operator="none"),), rejected={"duplicates": 4}
    )
    monkeypatch.setattr(
        expansion, "expand_trajectory_variants", lambda *args, **kwargs: fake
    )
    with pytest.raises(RuntimeError, match="only 1 survived"):
        _augment(_compiled(1), _Robot(), _scene_args(cfg, (1,), 0))


def test_cli_batches_all_recipes_and_records_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = []

    def run_batch(cfg, indices, batch_id, cli_args, *, last):
        batches.append((indices, last))
        return [{"status": "executed", "batch_id": batch_id}]

    monkeypatch.setattr(launcher, "_run_batch", run_batch)
    launcher.cli(
        [
            "--config",
            str(CONFIG_ROOT / "place.yaml"),
            "--num-envs",
            "3",
            "--recipe-indices",
            "0",
            "1",
            "2",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert batches == [((0, 1, 2), False), ((4,), True)]
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["status"] == "executed"
    assert len(manifest["replays"]) == 2
    assert manifest["measured_task_success"] is None


def test_cli_persists_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("IK failure")

    monkeypatch.setattr(launcher, "_run_batch", fail)
    with pytest.raises(RuntimeError, match="IK failure"):
        launcher.cli(
            [
                "--config",
                str(CONFIG_ROOT / "place.yaml"),
                "--no-expansion",
                "--output-dir",
                str(tmp_path),
            ]
        )
    manifest = json.loads((tmp_path / "expansion_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "IK failure" in manifest["error"]
