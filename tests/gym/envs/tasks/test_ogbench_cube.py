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

import numpy as np
import pytest
import torch

from embodichain_tasks.manipulation.ogbench_cube import (
    OGBenchCubeCfg,
    OGBenchCubeEnv,
    _CubeController,
    _base_mimic_compliance,
    _source_body_external_wrench,
    _source_state_observation,
)

ROOT = Path(__file__).with_name("data") / "ogbench_cube"
REFERENCE = json.loads((ROOT / "controller.json").read_text())


@pytest.mark.parametrize("index", range(3))
def test_point_compliance_preserves_source_force_units(index: int) -> None:
    from embodichain_tasks.manipulation.ogbench_cube import _base_point_compliance

    reference = json.loads((ROOT / "equality_point.json").read_text())["cases"][index]
    stiffness, damping = _base_point_compliance(
        reference["solref"],
        reference["solimp"],
        reference["approximate_inverse_weight"],
    )
    assert stiffness == pytest.approx(reference["force_stiffness"], rel=1e-12)
    assert damping == pytest.approx(reference["force_damping"], rel=1e-12)


def test_mimic_compliance_preserves_source_base_force_response() -> None:
    reference = json.loads((ROOT / "equality_scalar.json").read_text())
    frequency, damping = _base_mimic_compliance(
        reference["solref"], reference["solimp"]
    )
    # Independently compiled MuJoCo KBIP/R values define the force law. The
    # two free hinges have the same diagonal and actual inverse inertia.
    inverse_inertia = reference["inverse_effective_inertia"]
    assert frequency**2 / inverse_inertia == pytest.approx(
        reference["force_stiffness"], rel=1e-12
    )
    assert 2 * frequency * damping / inverse_inertia == pytest.approx(
        reference["force_damping"], rel=1e-12
    )


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


def test_external_pad_wrench_matches_source_rne_with_child_and_rotated_com() -> None:
    reference = json.loads((ROOT / "body_wrench_rne.json").read_text())

    def tensor(name: str, batch: bool = False) -> torch.Tensor:
        value = torch.tensor(reference[name], dtype=torch.float64)
        return value.unsqueeze(0) if batch else value

    actual = _source_body_external_wrench(
        tensor("pose", True),
        tensor("velocity", True),
        tensor("acceleration", True),
        tensor("incoming", True),
        tensor("mass"),
        tensor("com"),
        tensor("inertia"),
        tensor("joint_frames", True),
        reference["body_index"],
        reference["children"],
        tensor("gravity"),
    )
    np.testing.assert_allclose(
        actual.numpy()[0], reference["expected_external_wrench"], atol=1e-10
    )


def test_camera_capacity_is_rejected_before_native_world_creation() -> None:
    cfg = OGBenchCubeCfg(reference_root="unused", num_envs=50)
    with pytest.raises(ValueError, match="at most 32"):
        OGBenchCubeEnv(cfg)


def test_uncached_materials_are_rejected_before_native_world_creation() -> None:
    cfg = OGBenchCubeCfg(reference_root="unused")
    cfg.sim_cfg.physics_cfg.cache_material = False
    with pytest.raises(ValueError, match="cache_material=True"):
        OGBenchCubeEnv(cfg)


def test_temporal_solver_is_rejected_before_native_world_creation() -> None:
    cfg = OGBenchCubeCfg(reference_root="unused")
    cfg.sim_cfg.physics_cfg.enable_tgs = True
    with pytest.raises(ValueError, match="enable_tgs=False"):
        OGBenchCubeEnv(cfg)


def test_stale_point_metadata_requires_reference_reexport(tmp_path: Path) -> None:
    (tmp_path / "physical-contract.json").write_text("{}")
    cfg = OGBenchCubeCfg(reference_root=str(tmp_path))
    with pytest.raises(ValueError, match="re-export official-reference"):
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
