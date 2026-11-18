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

from embodichain_benchmarks.motion_generation.config import (
    EmbodimentSpecCfg,
    RobotSpecCfg,
    stable_hash,
)
from embodichain_benchmarks.motion_generation.embodiments import (
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


@pytest.mark.parametrize(
    "provider,config",
    [
        ("franka_panda", {"uid": "test"}),
        ("ur5", {}),
        ("ur5_pgi", {}),
        ("dual_arm", {"robot_type": "ur5"}),
        ("dual_arm", {"robot_type": "franka", "gripper_type": None}),
        ("dual_arm", {"robot_type": "franka", "gripper_type": "robotiq_2f_140"}),
    ],
)
def test_legacy_robot_resolution_preserves_unchanged_provider_hashes(
    provider: str, config: dict[str, object]
) -> None:
    resolution = resolve_embodiment(
        None,
        RobotSpecCfg(
            id="legacy_robot",
            provider=provider,
            config=config,
        ),
    )

    assert resolution.embodiment_id == "legacy_robot"
    assert resolution.component_path is None
    assert resolution.source == f"robot_provider:{provider}"
    assert resolution.config_hash == stable_hash(config)


def test_implicit_franka_hand_identity_matches_explicit_robotiq_without_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain_benchmarks.motion_generation import _atomic_fixtures
    from embodichain_benchmarks.motion_generation.robots.dual_arm import DualArmProvider

    def forbid_assets(*args: object, **kwargs: object) -> None:
        pytest.fail("identity resolution constructed native robot assets")

    monkeypatch.setattr(DualArmProvider, "build_cfg", forbid_assets)
    monkeypatch.setattr(_atomic_fixtures, "get_data_path", forbid_assets)
    config = {"robot_type": "franka"}
    implicit = resolve_embodiment(
        None, RobotSpecCfg(id="franka", provider="dual_arm", config=config)
    )
    explicit = resolve_embodiment(
        None,
        RobotSpecCfg(
            id="franka",
            provider="dual_arm",
            config={"robot_type": "franka", "gripper_type": "robotiq_2f_140"},
        ),
    )
    assert implicit.to_metadata() == explicit.to_metadata()
    assert implicit.config_hash != stable_hash(config)
    assert config == {"robot_type": "franka"}


def test_missing_component_is_reported_with_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        YamlEmbodimentProvider().resolve(
            EmbodimentSpecCfg(component="missing.yaml"),
            base_dir=tmp_path,
        )
