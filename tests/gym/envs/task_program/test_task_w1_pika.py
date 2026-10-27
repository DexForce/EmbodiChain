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

"""Physical and semantic contracts of the nine W1/Pika deployments."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RobotCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.gym.utils._component_composition import (
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config

_CONFIGS = (
    Path(__file__).parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation"
)
_TASKS = (
    ("repeated_pick_place", 6),
    ("hand_over", 1),
    ("tableware/blocks_ranking_rgb", 4),
    ("tableware/stack_blocks_two", 2),
    ("tableware/pour_water", 3),
    ("tableware/blocks_ranking_size", None),
    ("tableware/match_object_container", None),
    ("tableware/place_object_drawer", None),
    ("tableware/stack_cups", None),
)


@pytest.mark.parametrize("task,segments", _TASKS)
def test_w1_pika_deployment_composes_and_preserves_task_route(
    task: str, segments: int | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import embodichain.lab.sim.cfg as sim_cfg

    def asset(value: str) -> str:
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<robot name='fixture'><link name='base_link'/></robot>")
        return str(path)

    monkeypatch.setattr(sim_cfg, "get_data_path", asset)
    path = _CONFIGS / task / "task.dexforce_w1_030_pika.yaml"
    payload = load_config(path)
    physical = _resolve_gym_components(payload, base_dir=path.parent)
    if segments is None:
        importlib.import_module(
            "embodichain_tasks.manipulation." + task.replace("/", ".")
        )
    cfg = config_to_cfg(payload, source_path=path)
    assert type(cfg.robot) is RobotCfg
    assert set(cfg.robot.urdf_cfg.components) == {
        "chassis",
        "torso",
        "head",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    if task == "repeated_pick_place":
        restored = RobotCfg.from_dict(json.loads(json.dumps(cfg.robot.to_dict())))
        assert cfg.robot.urdf_cfg.fname == "DexforceW1V030_pika"
        assert restored.urdf_cfg.fname == cfg.robot.urdf_cfg.fname
        assert restored.urdf_cfg.fpath_prefix == cfg.robot.urdf_cfg.fpath_prefix
        assert restored.control_parts == cfg.robot.control_parts
        assert restored.init_qpos == cfg.robot.init_qpos
        assert restored.urdf_cfg.component_prefix == cfg.robot.urdf_cfg.component_prefix
        for side in ("left_arm", "right_arm"):
            torch.testing.assert_close(
                torch.tensor(restored.solver_cfg[side].tcp),
                torch.tensor(cfg.robot.solver_cfg[side].tcp),
            )
        for before, after in zip(cfg.robot.sensors, restored.sensors, strict=True):
            assert after.extrinsics.parent == before.extrinsics.parent
            torch.testing.assert_close(
                after.extrinsics.transformation, before.extrinsics.transformation
            )
    assert payload["id"].endswith("-DexforceW1V030Pika-v1")
    assert cfg.control_parts == ["left_arm", "left_eef", "right_arm", "right_eef"]
    assert isinstance(cfg.sim_cfg.physics_config, DefaultPhysicsCfg)
    assert str(cfg.sim_cfg.device) == "cpu"
    assert cfg.sensor == []
    assert cfg.sim_cfg.render_cfg.denoising.window == "dlss"
    assert cfg.sim_cfg.render_cfg.denoising.offscreen == "dlss"
    assert [s.uid for s in cfg.robot.sensors] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert cfg.robot.root_props.fixed_base
    assert cfg.robot.init_qpos[-4:] == [0.05] * 4
    if physical.config["env"].get("dataset"):
        depth = cfg.dataset.lerobot.params["depth_video"]
        assert depth["enable"] and depth["lossless"]
        assert not depth["keep_numeric_fallback"] and not depth["use_log"]
        assert depth["input_unit"] == depth["output_unit"] == "m"
        assert depth["depth_min"] <= min(s.near for s in cfg.robot.sensors)
        assert depth["depth_max"] >= max(s.far for s in cfg.robot.sensors)
        assert (depth["depth_max"] - depth["depth_min"]) / (2 * 4095) < 0.0004
    profile = physical.embodiment_skill_profile
    assert profile["contract_id"] == "dual_arm_parallel_gripper"
    assert profile["runtime_services"]["control_part_evidence"]["open_qpos"] == [
        0.05,
        0.05,
    ]
    assert "object_ids" not in profile["runtime_services"]["control_part_evidence"]
    assert all(
        p["commands"] == {"open": [0.05, 0.05], "grasp": [0.0, 0.0]}
        for p in profile["command_presets"]
    )
    if segments is None:
        assert "task_program" not in payload
        assert REGISTERED_ENVS[payload["id"]].cls is not EmbodiedEnv
        return
    assert REGISTERED_ENVS[payload["id"]].cls is EmbodiedEnv
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=profile,
        base_dir=path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    catalog = deployment.integration.registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    assert catalog.preflight(program).segment_count == segments
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    expected = (
        "verified" if task in ("repeated_pick_place", "hand_over") else "projected"
    )
    assert preset.effect_assurance.value == expected
    if task == "tableware/pour_water":
        # The action owns adaptive preparation and tilt; deployment geometry
        # identifies the receiving opening instead of a fixed bottle pose.
        factories = (
            deployment.integration.registration.registered_semantic_lowerer_factories
        )
        factory = next(f for f in factories if f.call_id == "simulation.pour")
        assert factory.target.receiver_id == "cup"
        target = factory.target.make_target()
        assert target.receiver_pose.entity_id == "cup"
        assert float(target.source_mouth_pose[2, 3]) == pytest.approx(
            0.0697246268391609
        )
        assert float(target.receiver_pose.relative_pose[2, 3]) == pytest.approx(
            0.04387804865837097
        )
        assert 0 < target.mouth_radius < target.receiver_radius
        assert "bottle_pour_pose" not in program.targets
