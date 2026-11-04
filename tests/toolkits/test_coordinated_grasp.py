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

"""Paired sampling preserves source populations, query traces and RNG ownership."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from unittest.mock import Mock
import weakref

import pytest
import torch

from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    FixedGripperCollisionGeometry,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import coordinated
from embodichain.toolkits.graspkit.pg_grasp.contact_fit import (
    SampledHandGeometry,
    HandContactChecker,
)


def _cfg() -> coordinated.CoordinatedGraspCfg:
    return coordinated.CoordinatedGraspCfg(
        contact_tolerance=0.0005,
        minimum_normal_cosine=0.5,
        normal_probe_distance=1e-5,
        max_candidates=32,
        candidate_chunk_size=32,
        cargo_clearance_reserve=0.008,
    )


def _geometry(role: str) -> SampledHandGeometry:
    gaps = torch.tensor([0.12, 0.02])
    pads = []
    for gap in gaps:
        pads.append(
            torch.tensor(
                [
                    [
                        [side * float(gap) / 2, y, z]
                        for y in (-0.01, 0.01)
                        for z in (0.01, 0.05)
                    ]
                    for side in (-1, 1)
                ]
            )
        )
    return SampledHandGeometry(
        torch.tensor([[[0.2 if role == "left" else -0.2, 0, 0.1]]])
        .expand(2, -1, -1)
        .clone(),
        torch.stack(pads),
        gaps,
    )


def _delegate(role: str) -> AntipodalGraspPoseGenerator:
    result = AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id=role),
        collision_geometry=FixedGripperCollisionGeometry(
            torch.zeros(1, 3), {"grasp_from_tcp": torch.eye(4).tolist()}
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(filter_ground_collision=False),
    )
    return result


def _query(rows: int = 1) -> dict:
    poses = torch.eye(4).repeat(rows, 1, 1)
    if rows == 2:
        poses[1, 0, 3] = 1
    return {
        "mesh_vertices": torch.tensor([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0]]),
        "mesh_triangles": torch.tensor([[0, 1, 2]]),
        "obj_poses": poses,
        "left_to_right_arm_direction": torch.tensor([1.0, 0, 0]),
        "approach_direction": torch.tensor([0.0, 0, -1.0]),
    }


def _ports(monkeypatch, *, reject_all=False) -> list[dict]:
    trace = []
    backends = {}

    class Surface:
        def __init__(self, vertices, faces, *, normal_probe_distance):
            trace.append(
                {
                    "op": "surface",
                    "vertex_count": len(vertices),
                    "normal_probe_distance": normal_probe_distance,
                }
            )

    class Checker(HandContactChecker):
        def __init__(
            self, geometry, target, *, contact_tolerance, minimum_normal_cosine
        ):
            self.target = target
            self.role = "left" if float(geometry.body[0, 0, 0]) > 0 else "right"
            trace.append(
                {
                    "op": "checker",
                    "role": self.role,
                    "contact_tolerance": contact_tolerance,
                    "minimum_normal_cosine": minimum_normal_cosine,
                }
            )

        def fit_contacts(self, obj, poses, widths):
            ids = torch.round((poses[:, 0, 3] - obj[0, 3]).abs() / 0.01).long()
            trace.append(
                {
                    "op": "fit",
                    "role": self.role,
                    "ids": ids.tolist(),
                    "z": [round(float(x), 7) for x in poses[:, 2, 3]],
                }
            )
            return poses.clone(), widths + 0.001

        def query(self, obj, poses, widths, **kwargs):
            ids = torch.round((poses[:, 0, 3] - obj[0, 3]).abs() / 0.01).long()
            fraction = (0.2 - poses[:, 2, 3]) / 0.046
            layer = torch.where(
                (fraction - 1).abs() < 0.001,
                0,
                torch.where((fraction - 0.5).abs() < 0.001, 1, 2),
            )
            rejected = (ids % 3 != layer) | (ids % 5 == 0) | reject_all
            trace.append(
                {
                    "op": "query",
                    "role": self.role,
                    "ids": ids.tolist(),
                    "layer": layer.tolist(),
                    "rejected": rejected.tolist(),
                    "ground": kwargs.get("is_filter_ground_collision"),
                }
            )
            return rejected, torch.zeros_like(widths)

    class Backend:
        device = torch.device("cpu")

        def __init__(self, model):
            self.model = model

        def get_dual_arm_valid_grasp_poses(self, *, object_pose, **kwargs):
            trace.append(
                {
                    "op": "propose",
                    "model": self.model,
                    "object_x": float(object_pose[0, 3]),
                    "random": torch.rand(3).tolist(),
                }
            )
            if float(object_pose[0, 3]) > 0.5:
                return None
            result = {}
            for role, sign in (("left", 1), ("right", -1)):
                ids = torch.arange(70)
                poses = torch.eye(4).repeat(70, 1, 1)
                poses[:, 0, 3] = object_pose[0, 3] + sign * ids * 0.01
                poses[:, 2, 3] = 0.2
                widths, costs = 0.04 + ids * 0.00001, ids.float()
                costs[2] = torch.inf
                costs[33] = torch.inf
                result[role] = {
                    "is_success": True,
                    "grasp_poses": poses,
                    "open_lengths": widths,
                    "total_cost": costs,
                }
            return result

    def backend(generator, *args, proposals_only=False):
        assert (
            proposals_only
        ), "paired refinement must explicitly request unqualified proposals"
        return backends.setdefault(
            id(generator), Backend(generator.gripper_model.model_id)
        )

    monkeypatch.setattr(AntipodalGraspPoseGenerator, "_backend", backend)
    monkeypatch.setattr(coordinated, "TriangleSurfaceQuery", Surface)
    monkeypatch.setattr(coordinated, "HandContactChecker", Checker)
    return trace


def _sampler() -> coordinated.CoordinatedGraspSampler:
    return coordinated.CoordinatedGraspSampler(
        _delegate("left"),
        _delegate("right"),
        _geometry("left"),
        _geometry("right"),
        cfg=_cfg(),
    )


def _assert_rows(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if want is None:
            assert got is None
            continue
        for role in ("left", "right"):
            assert got[role]["is_success"] is want[role]["is_success"]
            for key in ("grasp_poses", "open_lengths", "total_cost"):
                torch.testing.assert_close(
                    got[role][key],
                    torch.tensor(want[role][key]).reshape(
                        (-1, 4, 4) if key == "grasp_poses" else (-1,)
                    ),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("case_name", ["bounded", "cargo", "partial_rows", "rejected"])
def test_paired_population_query_trace_and_rng_match_pinned_source(
    monkeypatch, case_name
):
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("coordinated_sampler_trace.json")
        .read_text()
    )
    case = next(row for row in fixture["cases"] if row["case"] == case_name)
    trace = _ports(monkeypatch, reject_all=case_name == "rejected")
    sampler = _sampler()
    bounds = torch.tensor(case["cargo_bounds"]) if case["cargo_bounds"] else None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fixture["seed"])
        actual = sampler.get_dual_arm_valid_grasp_poses(
            **_query(case["rows"]), cargo_bounds=bounds
        )
        next_random = torch.rand(4)
    # Object-query construction is shared now. Keep every proposal, fit and
    # qualification event exactly equal to the frozen algorithm trace.
    assert [event for event in trace if event["op"] != "surface"] == [
        event for event in case["trace"] if event["op"] != "surface"
    ]
    surfaces = [event for event in trace if event["op"] == "surface"]
    assert len(surfaces) == 1
    assert surfaces[0] == next(
        event for event in case["trace"] if event["op"] == "surface"
    )
    _assert_rows(actual, case["output"])
    torch.testing.assert_close(
        next_random, torch.tensor(case["next_random"]), rtol=0, atol=0
    )


def test_explicit_bounds_do_not_leak_between_queries_or_replace_population(monkeypatch):
    _ports(monkeypatch)
    sampler = _sampler()
    query = _query()
    bounds = torch.tensor([[[-0.12, -0.03, 0], [0.12, 0.03, 0.3]]])
    biased = sampler.get_dual_arm_valid_grasp_poses(**query, cargo_bounds=bounds)[0]
    saved_costs = biased["left"]["total_cost"].clone()
    ordinary = sampler.get_dual_arm_valid_grasp_poses(**query)[0]
    for role in ("left", "right"):
        torch.testing.assert_close(
            biased[role]["grasp_poses"], ordinary[role]["grasp_poses"]
        )
        torch.testing.assert_close(
            biased[role]["open_lengths"], ordinary[role]["open_lengths"]
        )
        assert len(biased[role]["total_cost"]) == 32
        assert torch.isfinite(biased[role]["total_cost"]).all()
        assert not torch.equal(biased[role]["total_cost"], ordinary[role]["total_cost"])
    torch.testing.assert_close(biased["left"]["total_cost"], saved_costs)


def test_single_arm_facade_delegates_without_invoking_paired_provider():
    delegate = _delegate("single")
    provider = Mock(spec=coordinated.PairedGraspProvider)
    facade = coordinated.CoordinatedGraspPoseGenerator(delegate, provider)
    query = {
        key: value
        for key, value in _query().items()
        if key != "left_to_right_arm_direction"
    }
    poses, widths, costs = torch.eye(4)[None], torch.tensor([0.04]), torch.tensor([1.0])
    expected = {
        "get_valid_grasp_poses": [(poses, costs)],
        "get_best_grasp_poses": (torch.tensor([True]), poses, widths),
        "get_grasp_candidates": [(poses, widths, costs)],
    }
    for name, result in expected.items():
        method = Mock(return_value=result)
        setattr(delegate, name, method)
        assert getattr(facade, name)(**query) is result
        assert set(method.call_args.kwargs) == set(query)
        assert all(
            method.call_args.kwargs[key] is value for key, value in query.items()
        )
    provider.get_dual_arm_valid_grasp_poses.assert_not_called()


def test_unresolved_or_invalid_budget_is_rejected_before_sampling():
    left, right = _delegate("left"), _delegate("right")
    with pytest.raises(TypeError):
        coordinated.CoordinatedGraspSampler(
            left,
            right,
            _geometry("left"),
            _geometry("right"),
            cfg=coordinated.CoordinatedGraspCfg(),
        )
    cfg = _cfg()
    cfg.max_candidates = 0
    with pytest.raises(ValueError, match="max_candidates"):
        coordinated.CoordinatedGraspSampler(
            left, right, _geometry("left"), _geometry("right"), cfg=cfg
        )


def test_inferred_ground_policy_requires_an_explicit_supported_configuration():
    left = AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(),
        collision_geometry=FixedGripperCollisionGeometry(
            torch.zeros(1, 3), {"grasp_from_tcp": torch.eye(4).tolist()}
        ),
    )
    with pytest.raises(ValueError, match="filter_ground_collision=False"):
        coordinated.CoordinatedGraspSampler(
            left, _delegate("right"), _geometry("left"), _geometry("right"), cfg=_cfg()
        )


def test_shared_target_query_is_rebuilt_when_mesh_changes(monkeypatch):
    trace = _ports(monkeypatch)
    sampler = _sampler()
    query = _query()
    sampler.get_dual_arm_valid_grasp_poses(**query)
    sampler.get_dual_arm_valid_grasp_poses(**query)
    assert sum(event["op"] == "surface" for event in trace) == 1
    assert sum(event["op"] == "checker" for event in trace) == 2
    for version in range(1, 5):
        old_surface = weakref.ref(next(iter(sampler._left._surface_queries.values())))
        old_checkers = [
            weakref.ref(next(iter(hand._validators.values())))
            for hand in (sampler._left, sampler._right)
        ]
        query["mesh_vertices"].add_(0.001)
        sampler.get_dual_arm_valid_grasp_poses(**query)
        gc.collect()
        assert old_surface() is None
        assert all(reference() is None for reference in old_checkers)
        assert sum(event["op"] == "surface" for event in trace) == version + 1
        assert sum(event["op"] == "checker" for event in trace) == 2 * (version + 1)

    retained = [weakref.ref(value) for value in sampler._left._surface_queries.values()]
    sampler.clear_cache()
    gc.collect()
    assert all(reference() is None for reference in retained)
    before = len(trace)
    sampler.get_dual_arm_valid_grasp_poses(**query)
    assert sum(event["op"] == "surface" for event in trace[before:]) == 1
    assert sum(event["op"] == "checker" for event in trace[before:]) == 2


def test_real_contact_refinement_accepts_opposed_box_grasps_without_fixed_checker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trimesh
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGeneratorCfg
    from embodichain.toolkits.graspkit.pg_grasp import (
        _antipodal_backend as implementation,
    )

    nms = Mock(wraps=implementation.pose_nms)
    monkeypatch.setattr(implementation, "pose_nms", nms)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Paired contact refinement constructed a fixed-hand checker"
        )

    def annotate(backend):
        backend._hit_point_pairs = torch.tensor(
            [
                [[-0.075, -0.03, 0.0], [-0.075, 0.03, 0.0]],
                [[0.075, -0.03, 0.0], [0.075, 0.03, 0.0]],
            ]
        )
        return backend._hit_point_pairs

    monkeypatch.setattr(implementation, "GripperCollisionChecker", forbidden)
    monkeypatch.setattr(
        implementation._AntipodalMeshBackend,
        "_get_cache_dir",
        lambda *args: str(tmp_path / "unprepared.npy"),
    )
    monkeypatch.setattr(implementation._AntipodalMeshBackend, "annotate", annotate)

    def delegate(role):
        return AntipodalGraspPoseGenerator(
            ParallelJawGripperModelCfg(model_id=role),
            collision_geometry=FixedGripperCollisionGeometry(
                torch.zeros(1, 3), {"grasp_from_tcp": torch.eye(4).tolist()}
            ),
            collision_cfg=ParallelJawGraspCollisionCfg(filter_ground_collision=False),
            algorithm_cfg=AntipodalGraspPoseGeneratorCfg(approach_direction_samples=1),
        )

    mesh = trimesh.creation.box(extents=[0.2, 0.06, 0.1])
    sampler = coordinated.CoordinatedGraspSampler(
        delegate("left"),
        delegate("right"),
        _geometry("left"),
        _geometry("right"),
        cfg=_cfg(),
    )
    result = sampler.get_dual_arm_valid_grasp_poses(
        mesh_vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
        mesh_triangles=torch.tensor(mesh.faces),
        obj_poses=torch.eye(4)[None],
        left_to_right_arm_direction=torch.tensor([1.0, 0, 0]),
        approach_direction=torch.tensor([0.0, 0, -1]),
    )[0]
    assert nms.call_count == 2  # Only the side consumed by each hand is materialized.
    for role, x in (("left", -0.075), ("right", 0.075)):
        assert result[role]["is_success"]
        torch.testing.assert_close(
            result[role]["open_lengths"], torch.tensor([0.0602]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            result[role]["grasp_poses"][:, 0, 3], torch.tensor([x]), atol=1e-6, rtol=0
        )
    native_query = weakref.ref(next(iter(sampler._left._surface_queries.values())))
    sampler.clear_cache()
    gc.collect()
    assert native_query() is None
