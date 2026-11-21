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

"""Compare Cube control with pinned upstream MuJoCo-generated values."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.extensions.tasks.manipulation.ogbench_cube import (
    OGBenchCubeCfg,
    OGBenchCubeEnv,
    _CubeController,
    _source_state_observation,
)

ROOT = Path(__file__).with_name("data") / "ogbench_cube"
REFERENCE = json.loads((ROOT / "controller.json").read_text())


def test_dexuni_authors_connect_anchors_before_the_joint_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dexsim.spawn import NewtonEqualityDesc
    from dexsim.spawn.adapters.newton_adapter import NewtonSpawnAdapter

    from embodichain.extensions.tasks.manipulation._ogbench_cube_newton import (
        _CubeNewtonSpawnAdapter,
    )

    parent = Mock()
    monkeypatch.setattr(NewtonSpawnAdapter, "_build_scene", parent)
    connect = NewtonEqualityDesc(
        name="anchor",
        equality_type=0,
        object1="arm",
        object2="finger",
        data=(0.01, 0.02, 0.03),
        solref=(0.01, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    )
    gear = NewtonEqualityDesc(
        name="gear",
        equality_type=2,
        object1="leader",
        object2="follower",
        data=(0.0, 1.0, 0.0, 0.0, 0.0),
        solref=(0.01, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    )
    adapter = _CubeNewtonSpawnAdapter((connect, gear))
    binding = SimpleNamespace(
        links=[
            SimpleNamespace(name="arm", body_id=4),
            SimpleNamespace(name="finger", body_id=7),
        ],
        joints=[
            SimpleNamespace(name="leader", joint_id=8),
            SimpleNamespace(name="follower", joint_id=9),
        ],
    )
    result = Mock()
    result.get_articulation.return_value._physics_binding = binding
    backend = Mock()
    empty = SimpleNamespace(articulations=[])
    adapter._build_scene(result, None, 0, empty, "", backend, -1)
    backend.builder.add_custom_values.assert_not_called()

    arena = SimpleNamespace(articulations=[SimpleNamespace(name="ogbench_robot")])
    adapter._build_scene(result, None, 1, arena, "arena_0/", backend, 0)
    parent.assert_called_with(result, None, 1, arena, "arena_0/", backend, 0)
    result.get_articulation.assert_called_once_with("arena_0/ogbench_robot")
    point, joint = (
        call.kwargs for call in backend.builder.add_custom_values.call_args_list
    )
    assert point["mujoco:equality_constraint_type"] == 0
    assert point["mujoco:equality_constraint_body1"] == 4
    assert point["mujoco:equality_constraint_body2"] == 7
    assert point["mujoco:eq_solref"] == connect.solref
    np.testing.assert_allclose(
        tuple(point["mujoco:equality_constraint_anchor"]), (0.01, 0.02, 0.03)
    )
    assert joint["mujoco:equality_constraint_type"] == 2
    assert joint["mujoco:equality_constraint_joint1"] == 8
    assert joint["mujoco:equality_constraint_joint2"] == 9
    assert joint["mujoco:equality_constraint_polycoef"] == list(gear.data)


@pytest.mark.parametrize("case", range(3))
def test_28d_state_matches_pinned_source_observation(case: int) -> None:
    reference = json.loads((ROOT / "state_observation.json").read_text())["cases"][case]

    def tensor(name: str) -> torch.Tensor:
        return torch.tensor(reference[name], dtype=torch.float64).unsqueeze(0)

    actual = _source_state_observation(
        tensor("qpos"),
        tensor("qvel"),
        tensor("pinch"),
        tensor("cube_xyzw"),
        tensor("pad_wrench"),
    )
    assert actual.shape == (1, 28)
    np.testing.assert_allclose(actual.numpy()[0], reference["expected"], atol=1e-11)


def test_dexuni_rejects_multiple_envs_per_world() -> None:
    cfg = OGBenchCubeCfg(reference_root="unused", num_envs=2)
    with pytest.raises(ValueError, match="requires one Cube environment per World"):
        OGBenchCubeEnv(cfg)


def test_missing_source_sleep_setting_requires_reference_reexport(
    tmp_path: Path,
) -> None:
    (tmp_path / "physical-contract.json").write_text("{}")
    cfg = OGBenchCubeCfg(reference_root=str(tmp_path), enable_images=False)
    with pytest.raises(ValueError, match="Re-export official-reference"):
        OGBenchCubeEnv(cfg)


@pytest.fixture(scope="module")
def controller() -> _CubeController:
    return _CubeController(ROOT, REFERENCE["robot"])


@pytest.mark.parametrize("index", range(6))
def test_ik_matches_official_controller(
    controller: _CubeController, index: int
) -> None:
    case = REFERENCE["cases"][index]
    actual = controller.solve(np.array(case["target"]), np.array(case["seed"]))
    # Golden targets come from the pinned original controller, not this chain.
    # The final case also exercises workspace/action saturation and asymmetric
    # gripper state while the requested arm motion remains independent of it.
    np.testing.assert_allclose(actual, case["expected_ik"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("index", range(6))
def test_passive_and_split_tendon_forces_match_mujoco(index: int) -> None:
    case = REFERENCE["cases"][index]
    env = object.__new__(OGBenchCubeEnv)
    env.contract = REFERENCE["force_contract"]
    env._names = REFERENCE["names"]
    env._control = np.array(case["control"])[None]
    actual = env._passive_and_tendon_forces(
        np.array(case["qpos"])[None], np.array(case["qvel"])[None]
    )
    # Source qfrc_passive plus tendon qfrc_actuator, excluding the native arm PD.
    np.testing.assert_allclose(
        actual[0], case["expected_passive_tendon"], atol=1e-10, rtol=0
    )
