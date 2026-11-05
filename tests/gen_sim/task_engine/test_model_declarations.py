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

"""Task Engine model binding emits pure declarations through public owners."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import pytest
import yaml


@pytest.mark.parametrize(
    "identity",
    ["dual_franka_robotiq_arg2f_140", "dual_ur10_robotiq_arg2f_140", "tianji_marvin"],
)
def test_model_hand_binding_uses_public_descriptors_and_preserves_declared_evidence(
    identity, monkeypatch
):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions.grip_evidence import DriverOpeningBinding
    from embodichain.gen_sim.task_engine.embodiment_declarations import (
        resolve_source_hand_services,
        declare_source_grip_evidence,
        declare_source_opening_bindings,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("frontend declaration created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    from embodichain.utils.config_paths import resolve_config_path

    component = Path(
        resolve_config_path(
            f"extensions/embodichain_tasks/configs/components/embodiments/{identity}.yaml"
        )
    ).resolve()
    original = yaml.safe_load(component.read_text())
    before = deepcopy(original)
    if identity != "tianji_marvin":
        del original["skill_profile"]["runtime_services"]["hand_geometries"]
    resolved, hands, inputs = resolve_source_hand_services(original)
    assert len(hands) == 2 and original["simulation"] == before["simulation"]
    if identity != "tianji_marvin":
        assert (
            inputs[str(component.resolve())]
            == hashlib.sha256(component.read_bytes()).hexdigest()
        )
    from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
        get_parallel_jaw_model_asset_files,
    )

    generators = resolved["skill_profile"]["runtime_services"]["grasp_pose_generators"]
    from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

    expected_model_files = set()
    for generator in generators.values():
        model = resolve_parallel_jaw_gripper_model(generator["model"])
        expected_model_files.update(
            get_parallel_jaw_model_asset_files(
                model.model_id, side=generator.get("side")
            )
        )
    assert expected_model_files and all(
        inputs[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
        for path in expected_model_files
    )
    if identity == "tianji_marvin":
        assert resolved["simulation"]["class_type"] == "TianjiMarvin"
    opening = declare_source_opening_bindings(resolved, hands)
    for declaration in opening:
        binding = DriverOpeningBinding(
            declaration["control_part"],
            tuple(declaration["joint_names"]),
            tuple(declaration["driver_indices"]),
            tuple(declaration["open_qpos"]),
            declaration["tolerance"],
        )
        assert binding.driver_indices == (0,)
    scene = dict(
        rigid_objects=[
            dict(entity_id="table", simulation_uid="table_native", affordances=[]),
            dict(
                entity_id="object",
                simulation_uid="object_native",
                affordances=[dict(kind="antipodal_grasp")],
            ),
        ]
    )
    default = declare_source_grip_evidence(deepcopy(resolved), hands, scene)
    explicit = declare_source_grip_evidence(
        deepcopy(resolved), hands, scene, experimental_open_release=True
    )
    assert default["algorithm_revision"] == 1 and "opening_bindings" not in default
    assert (
        explicit["algorithm_revision"] == 2 and explicit["opening_bindings"] == opening
    )
    assert explicit["object_bindings"] == [["object", "object_native"]]
    from embodichain.lab.task_program.integrations.simulation.grip_evidence import (
        NativeGripEvidenceCfg,
        NativeGripEvidenceFactory,
    )

    shared_factory = NativeGripEvidenceFactory(
        NativeGripEvidenceCfg(
            sensor_uid=explicit["sensor_uid"],
            object_bindings=tuple(tuple(row) for row in explicit["object_bindings"]),
            pad_bindings=tuple(
                (part, tuple(names)) for part, names in explicit["pad_bindings"]
            ),
            algorithm_revision=explicit["algorithm_revision"],
            opening_bindings=tuple(
                DriverOpeningBinding(
                    row["control_part"],
                    tuple(row["joint_names"]),
                    tuple(row["driver_indices"]),
                    tuple(row["open_qpos"]),
                    row["tolerance"],
                )
                for row in opening
            ),
            max_contacts_per_env=explicit["max_contacts_per_env"],
            trace_capacity=explicit["trace_capacity"],
        )
    )
    assert (
        shared_factory.algorithm_revision == 2
        and len(shared_factory.opening_bindings) == 2
    )
    assert len(explicit["pad_bindings"]) == 2
    assert (
        resolved["skill_profile"]["runtime_services"]["control_part_evidence"]["kind"]
        == "joint_position_constraint"
    )


def test_robot_profile_alias_selects_standard_native_description():
    from embodichain.gen_sim.task_engine.robot_profiles import (
        canonical_robot_profile,
        profile_lateral_positions,
    )

    assert (
        canonical_robot_profile("dual_tianji")
        == canonical_robot_profile("tianji_marvin")
        == "tianji_marvin"
    )
    assert profile_lateral_positions("dual_franka") is None
    assert profile_lateral_positions("dual_ur10") is None
    with pytest.raises(ValueError, match="Unsupported"):
        profile_lateral_positions("unsupported")


def test_experimental_open_release_requires_boolean_before_any_model_resolution():
    from embodichain.gen_sim.task_engine.embodiment_declarations import (
        declare_source_grip_evidence,
    )

    with pytest.raises(TypeError, match="explicit boolean"):
        declare_source_grip_evidence({}, (), {}, experimental_open_release="false")
