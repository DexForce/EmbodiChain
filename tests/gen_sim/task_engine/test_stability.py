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

"""Native-contact support heights must not be confused with a reference rim."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.stability import (
    StabilityConstraint,
    TaskStabilityPort,
)
from embodichain.gen_sim.task_engine._task_program.support_evidence import (
    support_from_contacts,
)
from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
from embodichain.lab.task_program.language.schema import WaitStablePostCfg
from embodichain.lab.task_program.semantics import SceneObjectRef

__all__: list[str] = []

# Recorded task1190 poses and qualified contact patch; reduced box geometry
# keeps the regression portable without depending on generated/ignored assets.
_OBJECT_POSE = [
    [0.0964841843, -0.9952951670, 0.0088553717, 0.0148401288],
    [0.9945542812, 0.0967569351, 0.0387298688, 0.4825783670],
    [-0.0394044667, 0.0050703287, 0.9992104769, 0.7208061814],
    [0.0, 0.0, 0.0, 1.0],
]
_REFERENCE_POSE = [
    [0.0000079870, -0.9999902248, 0.0044091949, 0.0122178318],
    [0.9999929667, -0.0000084639, -0.0037342138, 0.5074306130],
    [0.0037342152, 0.0044091935, 0.9999833107, 0.7077052593],
    [0.0, 0.0, 0.0, 1.0],
]
_CONTACTS = np.array(
    [
        [0.0525466762, 0.5134806037, 0.7169609070],
        [0.0429285839, 0.4464045763, 0.7190517783],
        [-0.0162706412, 0.5203672051, 0.7173089981],
    ]
)
_CENTER_OF_MASS = np.array([0.0152580990, 0.4838037491, 0.7547985911])
_UPWARD_IMPULSE = 0.0009710995
_RIM_HEIGHT = 0.0362962335
_FLOOR_HEIGHT = 0.0065888311


def _case(damage: str | None = None, *, rim: bool = True):
    obj = torch.tensor(_OBJECT_POSE, dtype=torch.float32)[None]
    ref = torch.tensor(_REFERENCE_POSE, dtype=torch.float32)[None]
    initial = obj.clone()
    local_com = torch.linalg.solve(
        initial[0, :3, :3],
        torch.tensor(_CENTER_OF_MASS, dtype=torch.float32) - initial[0, :3, 3],
    )
    child_vertices = torch.tensor(
        list(product((-0.035, 0.035), (-0.034, 0.034), (0.0, 0.068)))
    )[None]
    support_vertices = torch.tensor(
        list(
            product(
                (-0.15, 0.15),
                (-0.1, 0.1),
                (0.0, _FLOOR_HEIGHT, _RIM_HEIGHT if rim else _FLOOR_HEIGHT),
            )
        )
    )[None]
    observations = []

    def observe(object_pose, reference_pose):
        if damage == "overflow":
            raise ValueError("Native support contact query saturated.")
        positions = _CONTACTS.copy()
        positions[:, :2] += (object_pose[0, :2, 3] - initial[0, :2, 3]).numpy()
        normals = np.tile([0.0, 0.0, 1.0], (len(positions), 1))
        impulses = np.full(len(positions), _UPWARD_IMPULSE / len(positions))
        com = (object_pose[0, :3, :3] @ local_com + object_pose[0, :3, 3]).numpy()
        if damage == "no_contact":
            impulses[:] = 0.0
        elif damage == "side_rim":
            normals[:] = [1.0, 0.0, 0.0]
        elif damage == "rim_edge":
            positions[:, 0] += 0.1
        elif damage == "nonfinite":
            positions[0, 2] = np.nan
        accepted, row = support_from_contacts(
            np.tile([11, 22], (len(positions), 1)),
            positions,
            normals,
            impulses,
            com,
            object_actor=22,
            reference_actor=11,
            tolerance=0.003,
        )
        if damage == "missing_com":
            row.pop("center_of_mass_world")
        elif damage == "no_upward_impulse":
            row["upward_impulse"] = 0.0
        elif damage == "nonfinite_row":
            row["contacts_world"][0][2] = float("nan")
        elif damage == "invalid_mask":
            accepted = False
        result = {
            "scope": "native_contact_support",
            "entity": "object",
            "reference": "support",
            "object_pose": object_pose.tolist(),
            "reference_pose": reference_pose.tolist(),
            "supported": [accepted],
            "env_ids": [0],
            "rows": [row],
        }
        observations.append(result)
        return torch.tensor([accepted]), result

    entities = {
        "object": SimpleNamespace(
            get_local_pose=lambda **kw: obj.clone(),
            get_vertices=lambda **kw: child_vertices.clone(),
        ),
        "support": SimpleNamespace(
            get_local_pose=lambda **kw: ref.clone(),
            get_vertices=lambda **kw: support_vertices.clone(),
        ),
    }
    preset = "gen_sim.support.stable"
    policy = CompiledPostPolicy(
        cfg=WaitStablePostCfg(entity="object", preset=preset),
        entity=SceneObjectRef("object"),
        source_path=("program", "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    port = TaskStabilityPort(
        Mock(),
        SimpleNamespace(get_rigid_object=entities.get, get_articulation=entities.get),
        SimpleNamespace(get_qpos=lambda **kw: torch.zeros(1, 2)),
        SimpleNamespace(
            rigid_objects=tuple(
                SimpleNamespace(entity_id=uid, simulation_uid=uid) for uid in entities
            ),
            articulations=(),
        ),
        {
            preset: StabilityConstraint(
                entity="object",
                reference="support",
                kind="supported_placement",
                displacement=(0.0, 0.0, 0.0465185078),
                position_tolerance=0.05,
                duration=0.08,
                timeout=0.16,
            )
        },
        step_dt=0.04,
        support_observers={("object", "support"): SimpleNamespace(observe=observe)},
    )
    return port, policy, segment, obj, ref, observations


def _run(port, policy, segment):
    list(port.actions(policy, segment=segment, active_mask=torch.tensor([True])))
    return port.post_policy_result(policy, segment=segment).tolist()


@pytest.mark.parametrize("acknowledge", [False, True])
def test_strong_stability_requires_real_clock_steps(acknowledge):
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, policy, segment, _, _, _ = _case()
    cfg = port._constraints[policy.cfg.preset]
    port._constraints[policy.cfg.preset] = replace(cfg, support_contract_revision=1)
    observer = port._support_observers[("object", "support")]
    legacy = observer.observe
    observer.bind_clock = lambda clock: None
    observer.observe = lambda obj, ref, **kw: legacy(obj, ref)
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    actions = port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    next(actions)
    if not acknowledge:
        with pytest.raises(ValueError, match="acknowledged environment steps"):
            next(actions)
        assert not port.post_policy_result(policy, segment=segment).any()
        return
    clock.advance_after_env_step()
    for _ in actions:
        clock.advance_after_env_step()
    assert port.post_policy_result(policy, segment=segment).tolist() == [True]
    result = port.post_policy_metadata(policy, segment=segment)
    assert result["start_step"] == 0 and result["end_step"] == 2
    assert result["clock_started_at"] == 0 and result["clock_ended_at"] == 0.08
    assert result["stable_seconds"] == [0.08]


def test_contact_patch_surface_not_reference_rim_controls_height():
    port, policy, segment, _, _, observations = _case()
    assert _run(port, policy, segment) == [True]
    measured = port.post_policy_metadata(policy, segment=segment)["measurements"]
    assert measured["native_support"]["supported"] == [True]
    assert observations[-1]["rows"][0]["com_patch_distance_m"] == pytest.approx(0.0)
    assert abs(measured["support_gap"][0]) <= 0.01


@pytest.mark.parametrize(
    "damage",
    [
        "no_contact",
        "side_rim",
        "rim_edge",
        "missing_com",
        "no_upward_impulse",
        "nonfinite_row",
        "invalid_mask",
    ],
)
def test_unqualified_contacts_never_refine_support_height(damage):
    port, policy, segment, _, _, _ = _case(damage)
    assert _run(port, policy, segment) == [False]


@pytest.mark.parametrize("damage", ["no_contact", "side_rim", "rim_edge"])
def test_geometrically_valid_flat_surface_still_requires_native_support(damage):
    port, policy, segment, _, _, observations = _case(damage, rim=False)
    assert _run(port, policy, segment) == [False]
    assert observations[-1]["supported"] == [False]


@pytest.mark.parametrize("damage", ["penetration", "hover", "position", "footprint"])
def test_native_support_does_not_override_geometry(damage):
    port, policy, segment, obj, _, observations = _case()
    cfg = port._constraints[policy.cfg.preset]
    if damage == "penetration":
        obj[:, 2, 3] -= 0.03
        port._constraints[policy.cfg.preset] = replace(
            cfg, displacement=(0.0, 0.0, cfg.displacement[2] - 0.03)
        )
    elif damage == "hover":
        obj[:, 2, 3] += 0.03
        port._constraints[policy.cfg.preset] = replace(
            cfg, displacement=(0.0, 0.0, cfg.displacement[2] + 0.03)
        )
    elif damage == "position":
        obj[:, 0, 3] += 0.06
    else:
        obj[:, 0, 3] += 0.3
        port._constraints[policy.cfg.preset] = replace(
            cfg, displacement=(0.3, 0.0, cfg.displacement[2])
        )
    assert _run(port, policy, segment) == [False]
    assert observations[-1]["supported"] == [True]
    measured = port.post_policy_metadata(policy, segment=segment)["measurements"]
    if damage in {"penetration", "hover"}:
        assert measured["position_error"][0] < cfg.position_tolerance
        assert measured["support_footprint"] == [True]
        if damage == "penetration":
            assert measured["support_gap"][0] < -cfg.support_tolerance
        else:
            assert measured["support_gap"][0] > cfg.support_tolerance
    elif damage == "position":
        assert abs(measured["support_gap"][0]) < cfg.support_tolerance
        assert measured["support_footprint"] == [True]
        assert measured["position_error"][0] > cfg.position_tolerance
    else:
        assert abs(measured["support_gap"][0]) < cfg.support_tolerance
        assert measured["position_error"][0] < cfg.position_tolerance
        assert measured["support_footprint"] == [False]


def test_contact_supported_motion_still_requires_uninterrupted_stability():
    port, policy, segment, obj, ref, observations = _case()
    initial_obj, initial_ref = obj.clone(), ref.clone()
    for step, _ in enumerate(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    ):
        obj.copy_(initial_obj)
        ref.copy_(initial_ref)
        obj[:, 0, 3] += 0.03 if step % 2 == 0 else 0.0
        ref[:, 0, 3] += 0.03 if step % 2 == 0 else 0.0
    assert port.post_policy_result(policy, segment=segment).tolist() == [False]
    assert all(value["supported"] == [True] for value in observations)


@pytest.mark.parametrize("damage", ["nonfinite", "overflow"])
def test_invalid_native_query_fails_closed(damage):
    port, policy, segment, _, _, _ = _case(damage)
    with pytest.raises(ValueError, match="finite|saturated"):
        _run(port, policy, segment)


def _region_case(*, hole=False, shifted=False):
    import trimesh

    obj = torch.tensor(
        [
            [-0.0542342663, -0.9955053329, -0.0776389837, -0.0886215121],
            [0.2067077458, 0.0648759604, -0.9762494564, -0.1683324128],
            [0.9768984318, -0.0689947605, 0.2022601962, 0.7823621631],
            [0, 0, 0, 1],
        ]
    )[None]
    child = trimesh.creation.box(extents=[0.078, 0.094, 0.08])
    child.vertices += [-0.092, -0.212, 0.7904227]
    child.vertices = (child.vertices - obj[0, :3, 3].numpy()) @ obj[0, :3, :3].numpy()
    table = (
        trimesh.creation.annulus(r_min=0.15, r_max=0.6, height=0.05)
        if hole
        else trimesh.creation.box(extents=[1, 1, 0.05])
    )
    ref = torch.eye(4)[None]
    ref[:, 2, 3] = 0.723622
    if hole:
        ref[:, :2, 3] = obj[:, :2, 3]
    if shifted:
        obj[:, 0, 3] = 0.49
    queries = {}
    natives = {}
    for name, mesh, pose in [("object", child, obj), ("table", table, ref)]:
        queries[name] = Mock(
            side_effect=lambda mesh=mesh: torch.tensor(
                np.asarray(mesh.vertices), dtype=torch.float32
            )[None]
        )
        natives[name] = SimpleNamespace(
            get_local_pose=lambda pose=pose, **kw: pose.clone(),
            get_vertices=lambda name=name, **kw: queries[name](),
            get_triangles=lambda mesh=mesh: torch.tensor(
                mesh.faces.copy(), dtype=torch.int64
            )[None],
        )

    def observe(pose, reference):
        return torch.tensor([True]), {
            "scope": "native_contact_support",
            "entity": "object",
            "reference": "table",
            "object_pose": pose.tolist(),
            "reference_pose": reference.tolist(),
            "supported": [True],
            "env_ids": [0],
            "rows": [
                {
                    "contacts_world": [
                        [float(pose[0, 0, 3]), float(pose[0, 1, 3]), 0.748622]
                    ],
                    "center_of_mass_world": pose[0, :3, 3].tolist(),
                    "com_patch_distance_m": 0.000729,
                    "com_tolerance_m": 0.002,
                    "upward_impulse": 0.000971,
                }
            ],
        }

    preset = "region"
    policy = CompiledPostPolicy(
        cfg=WaitStablePostCfg(entity="object", preset=preset),
        entity=SceneObjectRef("object"),
        source_path=("program", "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    port = TaskStabilityPort(
        Mock(),
        SimpleNamespace(get_rigid_object=natives.get, get_articulation=natives.get),
        SimpleNamespace(get_qpos=lambda **kw: torch.zeros(1, 2)),
        SimpleNamespace(
            rigid_objects=tuple(
                SimpleNamespace(entity_id=uid, simulation_uid=uid) for uid in natives
            ),
            articulations=(),
        ),
        {
            preset: StabilityConstraint(
                entity="object",
                reference="table",
                kind="supported_region",
                duration=0.08,
                timeout=0.16,
            )
        },
        step_dt=0.04,
        support_observers={("object", "table"): SimpleNamespace(observe=observe)},
    )
    return port, policy, segment, obj, ref, queries


def test_supported_region_accepts_53mm_origin_error_with_actual_contact_and_full_footprint():
    port, policy, segment, obj, _, queries = _region_case()
    proposal = torch.tensor([-0.1195624098, -0.2046827150, 0.7593013886])
    assert torch.linalg.vector_norm(obj[0, :3, 3] - proposal).item() == pytest.approx(
        0.053014, abs=1e-6
    )
    assert _run(port, policy, segment) == [True]
    result = port.post_policy_metadata(policy, segment=segment)
    assert result["kind"] == "supported_region"
    measured = result["measurements"]
    assert "position_error" not in measured
    assert measured["landing_region"]["full_footprint_inside"] == [True]
    assert abs(measured["support_gap"][0]) < 0.01
    assert measured["native_support"]["supported"] == [True]
    assert queries["object"].call_count <= 2 and queries["table"].call_count <= 2


@pytest.mark.parametrize(
    "kwargs",
    [{"reference": None}, {"displacement": (0, 0, 0)}, {"target_position": (0, 0, 0)}],
)
def test_supported_region_requires_reference_and_forbids_point_contract(kwargs):
    values = {
        "entity": "object",
        "reference": "table",
        "kind": "supported_region",
        **kwargs,
    }
    with pytest.raises(ValueError):
        StabilityConstraint(**values)


@pytest.mark.parametrize("kwargs", [{"hole": True}, {"shifted": True}])
def test_native_contact_does_not_override_full_region_footprint(kwargs):
    port, policy, segment, _, _, _ = _region_case(**kwargs)
    assert _run(port, policy, segment) == [False]
    measured = port.post_policy_metadata(policy, segment=segment)["measurements"]
    assert measured["native_support"]["supported"] == [True]
    assert measured["landing_region"]["full_footprint_inside"] == [False]


@pytest.mark.parametrize(
    "damage", ["hover", "penetration", "object_drift", "reference_drift", "no_contact"]
)
def test_region_keeps_contact_gap_and_uninterrupted_stability_gates(damage):
    port, policy, segment, obj, ref, _ = _region_case()
    if damage in {"hover", "penetration"}:
        obj[:, 2, 3] += 0.03 if damage == "hover" else -0.03
    elif damage == "no_contact":
        port._support_observers[("object", "table")] = SimpleNamespace(
            observe=lambda pose, reference: (
                torch.tensor([False]),
                {"scope": "native_contact_support", "supported": [False], "rows": []},
            )
        )
    initial_obj, initial_ref = obj.clone(), ref.clone()
    for step, _ in enumerate(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    ):
        if damage in {"object_drift", "reference_drift"}:
            value, initial = (
                (obj, initial_obj) if damage == "object_drift" else (ref, initial_ref)
            )
            value.copy_(initial)
            value[:, 0, 3] += 0.03 if step % 2 == 0 else 0.0
    assert port.post_policy_result(policy, segment=segment).tolist() == [False]


def test_cached_planar_geometry_matches_fresh_and_remeasures_dynamic_poses():
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        measure_planar_placement,
    )

    port, _, _, obj, ref, _ = _region_case()
    binding, reference = port._rigid_bindings["object"], port._rigid_bindings["table"]
    geometry = port._region_geometry[("object", "table")]
    for shift in (0.0, 0.9):
        pose = obj.clone()
        pose[:, 0, 3] += shift
        fresh = measure_planar_placement(
            port._simulation, binding, reference, pose, ref
        )
        cached = measure_planar_placement(
            port._simulation, binding, reference, pose, ref, _geometry=geometry
        )
        assert cached == fresh
        assert cached[0] == [shift == 0.0]


@pytest.mark.parametrize("damage", ["simulation", "binding", "batch", "ref_tilt"])
def test_geometry_cache_never_replaces_identity_or_pose_validation(damage):
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        measure_planar_placement,
    )

    port, _, _, obj, ref, _ = _region_case()
    simulation = port._simulation
    binding, reference = port._rigid_bindings["object"], port._rigid_bindings["table"]
    if damage == "simulation":
        simulation = SimpleNamespace(get_rigid_object=simulation.get_rigid_object)
    elif damage == "binding":
        binding = SimpleNamespace(simulation_uid="another_object")
    elif damage == "batch":
        obj, ref = obj.repeat(2, 1, 1), ref.repeat(2, 1, 1)
    else:
        ref[:, :3, :3] = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    with pytest.raises(ValueError, match="cache|near-level"):
        measure_planar_placement(
            simulation,
            binding,
            reference,
            obj,
            ref,
            _geometry=port._region_geometry[("object", "table")],
        )


def test_region_cannot_replace_missing_native_patch_with_mesh_height():
    port, policy, segment, _, _, _ = _region_case()
    original = port._support_observers[("object", "table")].observe

    def damaged(pose, reference):
        supported, observation = original(pose, reference)
        observation["rows"][0].pop("center_of_mass_world")
        return supported, observation

    port._support_observers[("object", "table")] = SimpleNamespace(observe=damaged)
    assert _run(port, policy, segment) == [False]


@pytest.mark.parametrize("kwargs", [{}, {"hole": True}, {"shifted": True}])
def test_exact_duplicate_payload_points_keep_complete_metrics(kwargs):
    from embodichain.gen_sim.task_engine._task_program import placement_evidence

    port, _, _, obj, ref, _ = _region_case(**kwargs)
    geometry = port._region_geometry[("object", "table")]
    args = (
        port._simulation,
        port._rigid_bindings["object"],
        port._rigid_bindings["table"],
        obj,
        ref,
    )
    repeated = replace(geometry, vertices=geometry.vertices.repeat(1, 3, 1))
    for angle in (0.0, 0.3, -0.8):
        pose = obj.clone()
        c, s = np.cos(angle), np.sin(angle)
        pose[:, :3, :3] = (
            pose.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ obj[:, :3, :3]
        )
        values = (*args[:3], pose, ref)
        assert placement_evidence.measure_planar_placement(
            *values, _geometry=repeated
        ) == placement_evidence.measure_planar_placement(*values, _geometry=geometry)


def test_degenerate_footprint_is_not_accepted_with_cached_geometry():
    from embodichain.gen_sim.task_engine._task_program import placement_evidence

    port, _, _, obj, ref, _ = _region_case()
    geometry = replace(
        port._region_geometry[("object", "table")], vertices=torch.zeros(1, 8, 3)
    )
    args = (
        port._simulation,
        port._rigid_bindings["object"],
        port._rigid_bindings["table"],
        obj,
        ref,
    )
    with pytest.raises(ValueError, match="footprint"):
        placement_evidence.measure_planar_placement(*args, _geometry=geometry)


def test_private_geometry_preparation_preserves_batch_point_sets():
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        _prepare_placement_geometry,
    )

    port, _, _, _, _, _ = _region_case()
    originals = {
        uid: native.get_vertices(scale=True).clone()
        for uid, native in port._objects.items()
    }
    for uid, native in port._objects.items():
        points = originals[uid].repeat(2, 3, 1)
        native.get_vertices = lambda points=points, **kw: points
        faces = native.get_triangles().repeat(2, 1, 1)
        native.get_triangles = lambda faces=faces: faces
    geometry = _prepare_placement_geometry(
        port._simulation,
        port._rigid_bindings["object"],
        port._rigid_bindings["table"],
        2,
    )
    assert geometry.vertices.shape == (2, originals["object"].shape[1], 3)
    for row in geometry.vertices:
        assert {tuple(value.tolist()) for value in row} == {
            tuple(value.tolist()) for value in originals["object"][0]
        }
