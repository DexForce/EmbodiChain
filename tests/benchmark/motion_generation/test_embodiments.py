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

"""Tests for static official-embodiment resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmark.motion_generation.config import EmbodimentSpecCfg, RobotSpecCfg
from scripts.benchmark.motion_generation.embodiments import (
    EmbodimentResolution,
    YamlEmbodimentProvider,
    check_embodiment_capabilities,
    resolve_embodiment,
)


def _write_component(path: Path) -> None:
    path.write_text(
        """
embodiment_id: test_panda
simulation:
  class_type: FrankaPanda
  robot_type: panda
sensor: []
skill_profile:
  resources:
    - resource_id: primary
      endpoints:
        - endpoint_id: motion
          control_part: arm
          capabilities: [motion.cartesian_pose]
        - endpoint_id: grasp
          control_part: hand
          capabilities: [interaction.grasp]
  runtime_services:
    grasp_pose_generators:
      hand:
        kind: antipodal_parallel_jaw
""".lstrip(),
        encoding="utf-8",
    )


def test_yaml_embodiment_resolution_collects_bindings_and_capabilities(
    tmp_path: Path,
) -> None:
    component = tmp_path / "embodiment.yaml"
    _write_component(component)

    resolution = YamlEmbodimentProvider().resolve(
        EmbodimentSpecCfg(component="embodiment.yaml"),
        base_dir=tmp_path,
    )

    assert isinstance(resolution, EmbodimentResolution)
    assert resolution.embodiment_id == "test_panda"
    assert resolution.endpoint_bindings == {
        "primary": {"motion": "arm", "grasp": "hand"}
    }
    assert resolution.capabilities == frozenset(
        {"motion.cartesian_pose", "interaction.grasp"}
    )
    assert resolution.runtime_services["grasp_pose_generators"]["hand"]["kind"] == (
        "antipodal_parallel_jaw"
    )
    assert resolution.to_metadata()["component"] == str(component.resolve())
    assert (
        check_embodiment_capabilities(resolution, {"motion.cartesian_pose"}).status
        == "supported"
    )
    assert (
        check_embodiment_capabilities(resolution, {"interaction.force"}).status
        == "unsupported"
    )


def test_legacy_robot_resolution_is_available_without_a_component() -> None:
    resolution = resolve_embodiment(
        None,
        RobotSpecCfg(
            id="legacy_panda",
            provider="franka_panda",
            config={"uid": "test"},
        ),
    )

    assert resolution.embodiment_id == "legacy_panda"
    assert resolution.component_path is None
    assert resolution.source == "robot_provider:franka_panda"


def test_missing_component_is_reported_with_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        YamlEmbodimentProvider().resolve(
            EmbodimentSpecCfg(component="missing.yaml"),
            base_dir=tmp_path,
        )
