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

import ast
from pathlib import Path

import pytest

import embodichain.gen_sim.action_agent_pipeline as action_agent_pipeline_package
from embodichain.gen_sim.action_agent_pipeline.runtime import atom_actions
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
    normalize_atomic_action_spec,
)

_PACKAGE_ROOT = Path(next(iter(action_agent_pipeline_package.__path__))).resolve()


@pytest.mark.parametrize(
    "spec",
    [
        {
            "atomic_action_class": "PickUp",
            "robot_name": "left_arm",
            "target_object": {"obj_name": "can_0", "affordance": "antipodal"},
        },
        {
            "atomic_action_class": "MoveEndEffector",
            "robot_name": "left_arm",
            "target_pose": {"reference": "absolute", "position": [0.1, 0.2, 0.3]},
        },
        {
            "atomic_action_class": "MoveJoints",
            "robot_name": "left_arm",
            "target_qpos": {"source": "initial"},
        },
        {
            "atomic_action_class": "MoveHeldObject",
            "robot_name": "right_arm",
            "target_object_pose": {
                "reference": "object",
                "obj_name": "plate_0",
                "offset": [0.0, 0.0, 0.02],
                "orientation_goal": "preserve",
                "orientation_axis": "none",
                "z_policy": "object_on_surface",
                "support": "plate_0",
                "surface_clearance": 0.003,
            },
        },
        {
            "atomic_action_class": "Place",
            "robot_name": "right_arm",
            "target_pose": {
                "reference": "relative",
                "offset": [0.0, 0.0, 0.0],
                "frame": "world",
            },
        },
        {
            "atomic_action_class": "CoordinatedPickment",
            "robot_name": "dual_arm",
            "target_object": {
                "obj_name": "tray_0",
                "affordance": "antipodal",
                "payloads": ["can_0"],
            },
            "target_object_pose": {
                "reference": "absolute",
                "position": [0.0, 0.0, 0.4],
                "orientation_goal": "preserve",
                "orientation_axis": "none",
            },
            "cfg": {"max_grasp_separation_angle_to_world_y_degrees": 20.0},
        },
    ],
)
def test_all_atomic_action_classes_normalize(spec: dict) -> None:
    normalized = normalize_atomic_action_spec(spec)

    assert normalized["atomic_action_class"] == spec["atomic_action_class"]
    assert normalized["control"] == "arm"
    assert AtomicActionSpec.from_mapping(spec).to_dict() == normalized


@pytest.mark.parametrize(
    ("control", "target_qpos"),
    [
        ("arm", {"source": "initial"}),
        ("hand", {"source": "gripper_state", "state": "open"}),
        ("arm", {"source": "joint_delta", "joint_index": 2, "delta_degrees": 15}),
    ],
)
def test_qpos_sources_remain_supported(
    control: str,
    target_qpos: dict,
) -> None:
    normalized = normalize_atomic_action_spec(
        {
            "atomic_action_class": "MoveJoints",
            "robot_name": "left_arm",
            "control": control,
            "target_qpos": target_qpos,
        }
    )

    assert normalized["target_qpos"] == target_qpos


@pytest.mark.parametrize("legacy_field", ["fn", "action", "target"])
def test_legacy_action_schemas_remain_rejected(legacy_field: str) -> None:
    with pytest.raises(ValueError, match="Legacy"):
        normalize_atomic_action_spec({legacy_field: {}})


def test_unknown_action_fields_remain_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported atomic action spec fields"):
        normalize_atomic_action_spec(
            {
                "atomic_action_class": "MoveJoints",
                "robot_name": "left_arm",
                "target_qpos": {"source": "initial"},
                "surprise": True,
            }
        )


def test_atom_actions_facade_preserves_type_and_normalizer_identity() -> None:
    assert atom_actions.AtomicActionSpec is AtomicActionSpec
    assert atom_actions.normalize_atomic_action_spec is normalize_atomic_action_spec


def test_new_builder_and_runtime_modules_do_not_import_their_facades() -> None:
    generation_modules = (
        "builder_protocols.py",
        "placement_action_specs.py",
        "task_plan_builders.py",
        "task_graph_builders.py",
        "diagnostic_common.py",
        "arrangement_diagnostics.py",
        "stacking_diagnostics.py",
        "relative_task_diagnostics.py",
        "relative_background_diagnostics.py",
        "relative_action_diagnostics.py",
    )
    runtime_modules = (
        "atomic_action_spec.py",
        "action_runtime_types.py",
        "action_parts.py",
        "pose_utils.py",
        "object_pose.py",
        "coordinated_grasp_geometry.py",
        "coordinated_grasp_ik.py",
        "coordinated_grasp.py",
        "coordinated_payload.py",
        "grasp_support.py",
        "action_targets.py",
        "trajectory_runtime.py",
        "action_execution.py",
        "parallel_execution.py",
    )

    for filename in generation_modules:
        source = (_PACKAGE_ROOT / "generation" / filename).read_text(encoding="utf-8")
        assert "generation.prompt_builders" not in source
    for filename in runtime_modules:
        source = (_PACKAGE_ROOT / "runtime" / filename).read_text(encoding="utf-8")
        assert "runtime.atom_actions" not in source


def test_new_internal_module_import_graph_is_acyclic() -> None:
    roots = (_PACKAGE_ROOT / "generation", _PACKAGE_ROOT / "runtime")
    module_paths = {
        f"embodichain.gen_sim.action_agent_pipeline.{root.name}.{path.stem}": path
        for root in roots
        for path in root.glob("*.py")
        if path.name not in {"__init__.py", "prompt_builders.py", "atom_actions.py"}
    }
    edges: dict[str, set[str]] = {name: set() for name in module_paths}
    for module_name, path in module_paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in module_paths:
                edges[module_name].add(node.module)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in visiting:
            raise AssertionError(f"Internal import cycle reaches {module_name}")
        if module_name in visited:
            return
        visiting.add(module_name)
        for dependency in edges[module_name]:
            visit(dependency)
        visiting.remove(module_name)
        visited.add(module_name)

    for module_name in edges:
        visit(module_name)
