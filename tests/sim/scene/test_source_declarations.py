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

"""Static source declarations reuse Spawn and preserve source precision."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from embodichain.lab.sim.scene.component_declarations import (
    source_scene_component_declaration,
)
from embodichain.lab.sim.scene.frozen_declarations import (
    effective_articulation_declaration,
    validate_frozen_pose_fields,
)
from embodichain.lab.sim.scene.source_contact import source_grasp_contact_clearances
from embodichain.lab.sim.scene.prismatic_declarations import (
    source_prismatic_physics_declarations,
)
from embodichain.lab.sim.scene.articulation_inputs import (
    discover_prismatic_parts,
    prismatic_reset_state,
    prismatic_handle_center_world,
)

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/source_declarations_reference.json").read_text()
)
_ASSET = Path(__file__).parent / _REFERENCE["fixture"]


@pytest.fixture(autouse=True)
def no_runtime_construction(monkeypatch):
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects import Robot

    def forbidden(*args, **kwargs):
        pytest.fail("static source declaration constructed a native runtime")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)


def _config():
    return {
        "uid": "cabinet",
        "fpath": str(_ASSET.resolve()),
        "fix_base": True,
        "body_scale": [2.0, 2.0, 2.0],
        "init_pos": [1.0, 2.0, 3.0],
        "init_rot": [0.0, 0.0, 90.0],
    }


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda c: c["mode"])
def test_effective_usd_declaration_matches_frozen_source_precision(case):
    original = _ASSET.read_bytes()
    assert hashlib.sha256(original).hexdigest() == _REFERENCE["fixture_sha256"]
    config = {
        "uid": "cabinet",
        "fpath": str(_ASSET.resolve()),
        "asset_physics_mode": case["mode"],
        "body_scale": [1.234567890123] * 3,
        "init_pos": [1.0, 2.0, 0.8],
        "init_rot": [12.0, -7.0, 30.0],
        "root_props": {"fixed_base": True},
    }
    before = deepcopy(config)
    effective = effective_articulation_declaration(config)
    assert {key: effective[key] for key in case["expected"]} == case["expected"]
    assert config == before and _ASSET.read_bytes() == original
    effective["root_props"]["fixed_base"] = False
    assert config["root_props"]["fixed_base"] is True


def test_non_usd_declaration_is_owned_without_claiming_asset_validation():
    config = {
        "uid": "opaque",
        "fpath": "opaque.urdf",
        "root_props": {"fixed_base": True},
    }
    output = effective_articulation_declaration(config)
    assert output == config
    output["root_props"]["fixed_base"] = False
    assert config["root_props"]["fixed_base"] is True


@pytest.mark.parametrize("kind", ["translation", "nonrigid", "shape"])
def test_frozen_pose_redundancy_is_checked_not_repaired(kind):
    value = {
        "uid": "box",
        "init_pos": [0.0, 0.0, 0.0],
        "init_local_pose": np.eye(4).tolist(),
    }
    if kind == "translation":
        value["init_pos"][0] = 1.0
    elif kind == "nonrigid":
        value["init_local_pose"][0][0] = 1.001
    else:
        value["init_local_pose"] = np.eye(3).tolist()
    original = deepcopy(value)
    with pytest.raises(ValueError):
        validate_frozen_pose_fields(value, value)
    assert value == original


def test_valid_matrix_projects_missing_fields_into_owned_raw_declaration():
    config = {
        "uid": "box",
        "init_local_pose": [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    original = deepcopy(config)
    merged = deepcopy(config)
    validate_frozen_pose_fields(merged, config)
    assert merged["init_pos"] == [1.0, 2.0, 3.0]
    assert merged["init_rot"] == pytest.approx([0.0, 0.0, 90.0], abs=1e-12)
    assert config == original


def test_component_projection_preserves_order_pose_and_collision_ownership():
    table = {
        "uid": "table",
        "init_pos": [1, 2, 3],
        "init_rot": [0, 0, 90],
        "max_convex_hull_num": 1,
        "shape": {
            "shape_type": "Mesh",
            "fpath": "table.glb",
            "acd_method": "visacd",
            "max_convex_hull_num": 1,
        },
    }
    cup = {
        "uid": "cup",
        "init_local_pose": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]],
        "max_convex_hull_num": 2,
        "acd_method": "visacd",
        "shape": {"shape_type": "Mesh", "fpath": "cup.glb"},
    }
    articulation = dict(
        _config(),
        description="semantic label",
        attributes={"semantic": True},
        joint_drive_props={"drive_type": "none"},
    )
    inputs = deepcopy((table, cup, articulation))
    output = source_scene_component_declaration(
        "source", background=[table], rigid_objects=[cup], articulations=[articulation]
    )
    assert (table, cup, articulation) == inputs
    simulation = output["simulation"]
    assert (
        simulation["background"][0]["uid"] == "table"
        and simulation["rigid_object"][0]["uid"] == "cup"
    )
    pose = np.asarray(simulation["background"][0]["init_local_pose"])
    assert pose[:3, 3].tolist() == [1, 2, 3]
    assert np.allclose(
        pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15, rtol=0
    )
    assert simulation["background"][0]["shape"]["collision"] == {
        "approximation": "convex_hull"
    }
    assert simulation["rigid_object"][0]["init_local_pose"] == cup["init_local_pose"]
    assert simulation["rigid_object"][0]["shape"]["collision"] == {
        "approximation": "convex_decomposition",
        "max_hulls": 2,
        "acd_method": "visacd",
    }
    art = simulation["articulation"][0]
    assert (
        art["root_props"]["fixed_base"] is True
        and art["asset_physics_mode"] == "overlay"
        and art["build_pk_chain"] is False
    )
    assert (
        "attributes" not in art and "description" not in art and "fix_base" not in art
    )
    with pytest.raises(ValueError, match="scene_id"):
        source_scene_component_declaration(
            " source ", background=[], rigid_objects=[], articulations=[]
        )


def test_contact_clearances_retain_default_and_explicit_link_margins():
    from dexsim.spawn import DexsimCollisionDesc

    objects = [
        {"uid": "a", "attrs": {"collision_props": {"contact_offset": 0.003}}},
        {"uid": "b"},
    ]
    robot = {
        "attrs": {"collision_props": {"contact_offset": 0.002}},
        "link_attrs": {
            "hand": {"attrs": {"collision_props": {"contact_offset": 0.005}}}
        },
    }
    before = deepcopy((objects, robot))
    assert source_grasp_contact_clearances(objects, robot) == {
        "a": 0.008,
        "b": 0.005 + DexsimCollisionDesc().contact_offset,
    }
    assert (objects, robot) == before
    with pytest.raises(ValueError, match="finite non-negative"):
        source_grasp_contact_clearances(
            [
                {
                    "uid": "a",
                    "attrs": {"collision_props": {"contact_offset": float("nan")}},
                }
            ],
            {},
        )


def test_source_prismatic_calibration_is_explicit_owned_and_preserves_other_owners():
    config = _config()
    original = deepcopy(config)
    calibrated = source_prismatic_physics_declarations(
        [config], bound_uids=["cabinet"]
    )[0]
    assert config == original
    assert calibrated["joint_drive_props"] == {
        "drive_type": "none",
        "friction": {"drawer_joint": 0.01},
    }
    assert calibrated["link_attrs"]["gen_sim_prismatic_link_mass"] == {
        "link_names_expr": ["drawer"],
        "attrs": {"mass_props": {"mass": 1.0}},
    }
    explicit = dict(
        config,
        joint_drive_props={
            "drive_type": "force",
            "target_mode": "position",
            "friction": 0.2,
        },
    )
    result = source_prismatic_physics_declarations(
        [explicit], bound_uids=["cabinet"], source_mass_owned_uids=["cabinet"]
    )[0]
    assert (
        result["joint_drive_props"] == explicit["joint_drive_props"]
        and "link_attrs" not in result
    )
    with pytest.raises(ValueError, match="undeclared"):
        source_prismatic_physics_declarations([config], bound_uids=["missing"])


def test_part_catalog_inputs_are_authored_geometry_not_live_state():
    config = _config()
    part = discover_prismatic_parts(config)[0]
    assert part.part_id == _REFERENCE["prismatic"]["part_id"]
    assert prismatic_reset_state(config, part.part_id) == {"openness": "closed"}
    assert prismatic_reset_state(dict(config, init_qpos=[0.0]), part.part_id) == {}
    assert prismatic_reset_state(config, "missing") == {}
    assert (
        prismatic_handle_center_world(config, part.part_id)
        == _REFERENCE["prismatic"]["center_world"]
    )
    with pytest.raises(ValueError, match="Unknown"):
        prismatic_handle_center_world(config, "missing")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_component_matrix_lists_and_typed_arrays_have_equal_effective_declarations(
    dtype,
):
    config = dict(_config(), asset_physics_mode="overlay")
    component = source_scene_component_declaration(
        "source", background=[], rigid_objects=[], articulations=[config]
    )
    serialized = json.dumps(component)
    restored = json.loads(serialized)["simulation"]["articulation"][0]
    before = deepcopy(restored)
    typed = deepcopy(restored)
    typed["init_local_pose"] = np.asarray(typed["init_local_pose"], dtype=dtype)
    typed_before = typed["init_local_pose"].copy()
    expected = effective_articulation_declaration(typed)
    actual = effective_articulation_declaration(restored)
    assert actual == expected
    assert restored == before and json.dumps(component) == serialized
    assert typed["init_local_pose"].dtype == dtype
    assert np.array_equal(typed["init_local_pose"], typed_before)


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "homogeneous"])
def test_matrix_list_decoding_does_not_repair_invalid_pose(kind):
    pose = np.eye(4)
    if kind == "shape":
        pose = np.eye(3)
    elif kind == "nonfinite":
        pose[0, 3] = np.nan
    else:
        pose[3, 0] = 0.1
    config = dict(
        _config(), asset_physics_mode="overlay", init_local_pose=pose.tolist()
    )
    before = deepcopy(config)
    error = (ValueError, IndexError) if kind == "shape" else ValueError
    with pytest.raises(error):
        effective_articulation_declaration(config)
    assert np.array_equal(
        np.asarray(config["init_local_pose"]),
        np.asarray(before["init_local_pose"]),
        equal_nan=True,
    )
