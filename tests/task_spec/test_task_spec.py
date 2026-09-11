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

from copy import deepcopy
import json
import pytest

from embodichain.task_spec import (
    canonical_template,
    semantic_hash,
    validate_task_template,
)
from embodichain.task_spec.expressions import validate_expression


def template():
    return {
        "schema_version": "taskspec/template/v0.1",
        "semantic_version": "0.1",
        "roles": {"cube": {"kind": "object"}, "tray": {"kind": "container"}},
        "init": [],
        "goal": [
            {
                "predicate": "relative_position",
                "object": "cube",
                "reference": "tray",
                "relation": "left_of",
                "margin": {"value": 10, "unit": "cm"},
            }
        ],
        "invariants": [],
        "requirements": [],
    }


def seal(value):
    result = deepcopy(value)
    result["semantic_hash"] = semantic_hash(result)
    return result


def test_roundtrip_and_repeated_hashing_are_stable():
    value = seal(template())
    assert semantic_hash(value) == value["semantic_hash"]
    assert validate_task_template(json.loads(json.dumps(value))) == value
    canonical = canonical_template(value)
    assert canonical_template(canonical) == canonical


def test_renaming_roles_does_not_depend_on_alphabetical_order():
    first = template()
    renamed = template()
    renamed["roles"] = {"zz": {"kind": "object"}, "aa": {"kind": "container"}}
    renamed["goal"][0].update(object="zz", reference="aa")
    assert semantic_hash(first) == semantic_hash(renamed)


def test_identical_role_kinds_and_commutative_terms_are_canonical():
    first = template()
    first["roles"]["tray"]["kind"] = "object"
    first["goal"].append(
        {
            "predicate": "upright",
            "object": "cube",
            "max_tilt": {"value": 0.1, "unit": "rad"},
        }
    )
    renamed = template()
    renamed["roles"] = {"right": {"kind": "object"}, "left": {"kind": "object"}}
    renamed["goal"][0].update(object="right", reference="left")
    renamed["goal"].insert(
        0,
        {
            "predicate": "upright",
            "object": "right",
            "max_tilt": {"value": 0.1, "unit": "rad"},
        },
    )
    assert semantic_hash(first) == semantic_hash(renamed)


@pytest.mark.parametrize(
    "relation,inverse", [("left_of", "right_of"), ("in_front_of", "behind")]
)
def test_inverse_relations_and_si_units(relation, inverse):
    first = template()
    first["goal"][0]["relation"] = relation
    second = deepcopy(first)
    second["goal"][0].update(
        object="tray",
        reference="cube",
        relation=inverse,
        margin={"value": 0.1, "unit": "m"},
    )
    assert semantic_hash(first) == semantic_hash(second)


def test_normative_order_and_threshold_changes_change_identity():
    first = template()
    first["temporal"] = [
        {
            "op": "sequence",
            "args": [
                {
                    "predicate": "upright",
                    "object": "cube",
                    "max_tilt": {"value": 0.1, "unit": "rad"},
                },
                deepcopy(first["goal"][0]),
            ],
        }
    ]
    changed = deepcopy(first)
    changed["temporal"][0]["args"].reverse()
    assert semantic_hash(first) != semantic_hash(changed)
    changed = deepcopy(first)
    changed["goal"][0]["margin"]["value"] = 20
    assert semantic_hash(first) != semantic_hash(changed)


@pytest.mark.parametrize(
    "field", ["plan", "trajectory", "witness", "seed", "robot_model"]
)
def test_execution_details_are_rejected_at_template_boundary(field):
    value = template()
    value[field] = "not task semantics"
    with pytest.raises(ValueError, match="fields"):
        semantic_hash(value)


@pytest.mark.parametrize(
    "expression",
    [
        {"predicate": "pour_volume", "object": "cube"},
        {"predicate": "upright", "object": "cube"},
        {
            "predicate": "upright",
            "object": "cube",
            "max_tilt": {"value": True, "unit": "rad"},
        },
        {
            "predicate": "upright",
            "object": "cube",
            "max_tilt": {"value": -1, "unit": "rad"},
        },
        {
            "predicate": "upright",
            "object": "cube",
            "max_tilt": {"value": 1, "unit": "m"},
        },
        {
            "predicate": "released",
            "object": "cube",
            "holder": "arm",
            "module": "os.system",
        },
        {"op": "not", "args": []},
        {
            "op": "and",
            "args": [{"predicate": "released", "object": "cube", "holder": "arm"}],
            "extra": 1,
        },
        {"predicate": ["upright"]},
        {"op": "or", "args": [lambda: True]},
    ],
)
def test_restricted_ast_rejects_invalid_or_executable_values(expression):
    with pytest.raises(ValueError):
        validate_expression(expression)


def test_missing_roles_unknown_capabilities_and_invalid_versions_are_rejected():
    value = template()
    value["goal"][0]["object"] = "missing"
    with pytest.raises(ValueError, match="role"):
        semantic_hash(value)
    value = template()
    value["roles"]["cube"]["capabilities"] = ["liquid_volume_observation"]
    with pytest.raises(ValueError, match="capabilit"):
        semantic_hash(value)
    value = template()
    value["semantic_version"] = "999"
    with pytest.raises(ValueError, match="version"):
        semantic_hash(value)


def test_nonfinite_cyclic_and_excessively_deep_inputs_are_rejected():
    for invalid in [float("nan"), float("inf"), -float("inf"), lambda: None]:
        value = template()
        value["goal"][0]["margin"]["value"] = invalid
        with pytest.raises(ValueError):
            semantic_hash(value)
    cyclic = {"op": "not"}
    cyclic["args"] = [cyclic]
    with pytest.raises(ValueError):
        validate_expression(cyclic)
    expression = {"predicate": "released", "object": "cube", "holder": "arm"}
    for _ in range(40):
        expression = {"op": "not", "args": [expression]}
    with pytest.raises(ValueError):
        validate_expression(expression)


def test_validator_returns_detached_data_and_rejects_stale_hash():
    value = seal(template())
    decoded = validate_task_template(value)
    decoded["roles"]["cube"]["kind"] = "support"
    assert value["roles"]["cube"]["kind"] == "object"
    value["goal"][0]["margin"]["value"] = 20
    with pytest.raises(ValueError, match="hash"):
        validate_task_template(value)


def test_canonical_quantities_are_exact_and_independent_of_decimal_context():
    from decimal import localcontext

    first = template()
    first["goal"][0]["margin"] = {"value": 0.123456789, "unit": "m"}
    expected = semantic_hash(first)
    with localcontext() as context:
        context.prec = 6
        assert semantic_hash(first) == expected
    second = deepcopy(first)
    first["goal"][0]["margin"]["value"] = 2**53
    second["goal"][0]["margin"]["value"] = 2**53 + 1
    assert semantic_hash(first) != semantic_hash(second)
    first["goal"][0]["margin"]["value"] = 10**400
    with pytest.raises(ValueError):
        semantic_hash(first)


def test_decimal_units_preserve_exact_threshold_and_zero():
    first = template()
    second = template()
    first["goal"][0]["margin"] = {"value": "0.123456789123456789", "unit": "m"}
    second["goal"][0]["margin"] = {"value": "12.3456789123456789", "unit": "cm"}
    assert semantic_hash(first) == semantic_hash(second)
    second["goal"][0]["margin"]["value"] = "12.3456789123456788"
    assert semantic_hash(first) != semantic_hash(second)
    first["goal"][0]["margin"]["value"] = "-0.00"
    second["goal"][0]["margin"] = {"value": 0, "unit": "m"}
    assert semantic_hash(first) == semantic_hash(second)


def test_si_normalization_remains_closed_at_small_decimal_boundaries():
    value = template()
    value["goal"][0]["margin"] = {"value": "1e-100", "unit": "mm"}
    canonical = canonical_template(value)
    assert canonical_template(canonical) == canonical


def test_bounded_template_rejects_excessive_predicate_count_before_hashing():
    value = template()
    value["goal"] = [
        {
            "predicate": "upright",
            "object": "cube",
            "max_tilt": {"value": i, "unit": "rad"},
        }
        for i in range(129)
    ]
    with pytest.raises(ValueError, match="predicate"):
        semantic_hash(value)


@pytest.mark.parametrize(
    "value", ["1e" + "9" * 100, "1e-9999", "NaN", "Infinity", True]
)
def test_invalid_decimal_quantities_raise_protocol_errors(value):
    expression = {
        "predicate": "upright",
        "object": "cube",
        "max_tilt": {"value": value, "unit": "rad"},
    }
    with pytest.raises(ValueError):
        validate_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        {
            "predicate": "object_at_target",
            "object": "item",
            "target": "site",
            "tolerance": {"value": 2, "unit": "mm"},
        },
        {
            "predicate": "stack_supported",
            "object": "item",
            "support": "base",
            "max_tilt": {"value": 0.1, "unit": "rad"},
            "max_gap": {"value": 0.2, "unit": "cm"},
            "max_offset": {"value": 5, "unit": "mm"},
        },
        {
            "predicate": "held_stable",
            "object": "item",
            "holder": "arm",
            "position_tolerance": {"value": 2, "unit": "mm"},
            "angular_tolerance": {"value": 0.1, "unit": "rad"},
            "duration": {"value": 3000, "unit": "ms"},
        },
        {"predicate": "released", "object": "item", "holder": "arm"},
        {
            "predicate": "joint_position",
            "joint": "hinge",
            "position": {"value": -0.5, "unit": "rad"},
            "tolerance": {"value": 0.1, "unit": "rad"},
        },
    ],
)
def test_whitelisted_predicates_roundtrip_without_executing_observers(expression):
    normalized = validate_expression(expression)
    assert validate_expression(normalized) == normalized
    logical = {"op": "not", "args": [{"op": "or", "args": [expression, expression]}]}
    assert validate_expression(logical)["op"] == "not"


def test_snapshot_mutation_does_not_change_supported_vocabulary():
    from embodichain.task_spec import registry_snapshot

    snapshot = registry_snapshot()
    snapshot["predicates"]["upright"]["roles"]["object"].append("liquid")
    snapshot["capabilities"].append("arbitrary")
    assert (
        "liquid" not in registry_snapshot()["predicates"]["upright"]["roles"]["object"]
    )
    assert "arbitrary" not in registry_snapshot()["capabilities"]


def test_pure_package_imports_without_optional_dependencies():
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import embodichain.task_spec; "
            "assert not any(m.startswith(('torch', 'numpy', 'embodichain.lab', 'embodichain.gen_sim')) for m in sys.modules)",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
