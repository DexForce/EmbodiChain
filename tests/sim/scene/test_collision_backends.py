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

"""Optional collision backends fail explicitly when their binding is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.lab.sim.scene import _collision_backends as backends


@pytest.mark.parametrize(
    "failure", [ModuleNotFoundError("newton"), OSError("missing native library")]
)
def test_missing_newton_has_actionable_dependency_error_and_no_fallback(
    monkeypatch, failure
):
    requested = []

    def missing(name):
        requested.append(name)
        raise failure

    monkeypatch.setattr(backends, "import_module", missing)
    with pytest.raises(ImportError, match="Newton") as caught:
        backends._require_newton()
    assert requested == ["newton"]
    assert caught.value.__cause__ is failure


def test_incomplete_newton_binding_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ImportError, match="collision pipeline"):
        backends._require_newton()


def test_installed_newton_preflight():
    pytest.importorskip(
        "newton", reason="Newton runtime is not installed; coverage is unqualified"
    )
    backends._require_newton()


def test_newton_distance_band_keeps_threshold_uncertainty_and_closes_owned_state():
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
    from embodichain.lab.sim.scene.feasibility.collision import (
        UnsupportedFeasibilityError,
    )

    left, right = _ConvexCollection(), _ConvexCollection()
    mesh = trimesh.creation.box((0.2, 0.2, 0.2))
    left.add_object("payload", mesh)
    pose = np.eye(4)
    pose[0, 3] = 0.202
    right.add_object("obstacle", mesh, transform=pose)
    interval = left.distance_interval(right, 0.002001)
    assert interval.lower < 0.002001 < interval.upper
    with pytest.raises(UnsupportedFeasibilityError, match="precision interval"):
        interval.below(0.002001)
    pose[0, 3] = 1.0
    right.set_transform("obstacle", pose)
    far = left.distance_interval(right, 0.002001)
    assert far.lower < 0.03 and far.upper == float("inf")
    assert far.below(0.002001) is False
    left.close()
    right.close()
    with pytest.raises(RuntimeError, match="closed"):
        left.distance_interval(right, 0.002001)


@pytest.mark.parametrize(
    "offset, expected", [(0.0, True), (-(2**-20), True), (2**-20, False), (1.0, False)]
)
@pytest.mark.parametrize("kind", ["box", "mesh"])
def test_native_strict_surfaces_decide_binary_touch_penetration_and_separation(
    kind, offset, expected
):
    import json
    import numpy as np
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
    )

    if kind == "mesh":
        import trimesh

        mesh = trimesh.creation.box((1.0, 1.0, 1.0))
        shape = dict(
            vertices=tuple(map(tuple, mesh.vertices)),
            triangles=tuple(map(tuple, mesh.faces)),
        )
    else:
        shape = dict(dimensions=(1.0, 1.0, 1.0))
    checker = NewtonCollisionChecker(
        tuple(
            CollisionBody(name, name, kind, robot=True, **shape) for name in ("a", "b")
        ),
        allow_triangle_meshes=kind == "mesh",
    )
    pose = np.eye(4)
    pose[0, 3] = 1.0 + offset
    frames = {"a": np.eye(4), "b": pose}
    before = {key: value.copy() for key, value in frames.items()}
    try:
        result = checker.contact_query(frames)
        assert result.below(0.0) is expected
        assert checker.contact_pairs(frames) == ((("a", "b"),) if expected else ())
        json.dumps(result.metadata(), allow_nan=False)
        assert all(np.array_equal(value, before[key]) for key, value in frames.items())
        if offset == 0:
            # Strict native surface information does not turn the old
            # conversion interval into an exact distance oracle.
            checker.check(frames)
            assert checker.last_clearance_interval[1] > 0.0
    finally:
        checker.close()


def test_native_strict_contact_preserves_conversion_unknown_and_overflow():
    import numpy as np
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    bodies = (
        CollisionBody("a", "a", "box", (1.0, 1.0, 1.0), robot=True),
        CollisionBody("b", "b", "box", (1.0, 1.0, 1.0)),
    )
    pose = np.eye(4)
    pose[0, 3] = 1.0 + 2**-26
    checker = NewtonCollisionChecker(bodies)
    try:
        result = checker.contact_query({"a": np.eye(4), "b": pose})
        assert result.representation_error_m > 0
        with pytest.raises(UnsupportedFeasibilityError, match="cannot decide"):
            result.below(0.0)
        with pytest.raises(UnsupportedFeasibilityError, match="cannot decide"):
            checker.contact_pairs({"a": np.eye(4), "b": pose})
    finally:
        checker.close()
    checker = NewtonCollisionChecker(bodies, max_contacts=1)
    try:
        with pytest.raises(UnsupportedFeasibilityError, match="overflowed"):
            checker.contact_pairs({"a": np.eye(4), "b": np.eye(4)})
    finally:
        checker.close()


def test_native_batch_reports_all_pairs_once_and_owned_results_survive_reuse(
    monkeypatch,
):
    import numpy as np
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
    )

    checker = NewtonCollisionChecker(
        tuple(
            CollisionBody(name, name, "sphere", (0.5,), robot=True)
            for name in ("a", "b", "c")
        )
    )
    calls = []
    native = checker._pipeline.collide

    def counted(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    monkeypatch.setattr(checker._pipeline, "collide", counted)
    frames = {name: np.eye(4) for name in ("a", "b", "c")}
    frames["b"][0, 3], frames["c"][0, 3] = 0.5, 1.0
    try:
        assert checker.contact_pairs(frames) == (("a", "b"), ("a", "c"), ("b", "c"))
        assert len(calls) == 1
        result = checker.contact_query(frames)
        assert checker.contact_pairs(
            frames, allowed_contacts=(("a", "b"), ("b", "c"))
        ) == (("a", "c"),)
        frames["b"][0, 3], frames["c"][0, 3] = 3.0, 6.0
        assert checker.contact_pairs(frames) == ()
        assert result.below(0.0) is True
    finally:
        checker.close()
    assert result.below(0.0) is True


def test_explicit_triangle_mesh_surface_keeps_concavity_and_pair_ids():
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    first, second = trimesh.creation.box((0.5, 0.5, 0.5)), trimesh.creation.box(
        (0.5, 0.5, 0.5)
    )
    first.apply_translation((-1.0, 0.0, 0.0))
    second.apply_translation((1.0, 0.0, 0.0))
    mesh = trimesh.util.concatenate((first, second))
    probe = trimesh.creation.box((0.25, 0.25, 0.25))
    bodies = (
        CollisionBody(
            "surface",
            "surface",
            "mesh",
            vertices=tuple(map(tuple, mesh.vertices)),
            triangles=tuple(map(tuple, mesh.faces)),
        ),
        CollisionBody(
            "probe",
            "probe",
            "mesh",
            vertices=tuple(map(tuple, probe.vertices)),
            triangles=tuple(map(tuple, probe.faces)),
            robot=True,
        ),
    )
    with pytest.raises(UnsupportedFeasibilityError, match="explicit collision adapter"):
        NewtonCollisionChecker(bodies)
    collapsed = CollisionBody(
        "bad",
        "bad",
        "mesh",
        vertices=((1.0, 0.0, 0.0), (1.0 + 2**-30, 0.0, 0.0), (1.0, 1.0, 0.0)),
        triangles=((0, 1, 2),),
    )
    with pytest.raises(UnsupportedFeasibilityError, match="collapsed faces"):
        NewtonCollisionChecker((collapsed,), allow_triangle_meshes=True)
    checker = NewtonCollisionChecker(bodies, allow_triangle_meshes=True)
    try:
        # A hull would bridge this explicitly empty gap.
        assert checker.contact_pairs({"surface": np.eye(4), "probe": np.eye(4)}) == ()
        pose = np.eye(4)
        pose[0, 3] = 1.25
        assert checker.contact_pairs({"surface": np.eye(4), "probe": pose}) == (
            ("surface", "probe"),
        )
    finally:
        checker.close()


@pytest.mark.parametrize("x", [1.0, 1.0 + 2**-26, 2.0, 1.5e6])
def test_collection_clearance_bound_preserves_native_precision_and_near_contacts(
    monkeypatch, x
):
    import json
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
    from embodichain.lab.sim.scene.feasibility import (
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    left, right = _ConvexCollection(geometry_kind="mesh"), _ConvexCollection(
        geometry_kind="mesh"
    )
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    pose = np.eye(4)
    pose[0, 3] = x
    left.add_object("a", mesh)
    right.add_object("b", mesh, pose)
    builds = []
    original = NewtonCollisionChecker._build_query

    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(NewtonCollisionChecker, "_build_query", counted)
    try:
        lower = left.clearance_lower_bound(right)
        assert not builds
        assert lower is None or type(lower) is float
        json.dumps({"lower": lower}, allow_nan=False)
        if x > 1e6:
            assert lower is None
            with pytest.raises(UnsupportedFeasibilityError, match="precision range"):
                left.contact_query(right, 0.0)
        elif x > 1.5:
            assert 0 < lower <= x - 1.0
            assert left.contact_query(right, 0.0).below(0.0) is False
        else:
            assert lower == 0.0
            result = left.contact_query(right, 0.0)
            if x == 1.0:
                assert result.below(0.0) is True
            else:
                with pytest.raises(UnsupportedFeasibilityError, match="cannot decide"):
                    result.below(0.0)
    finally:
        left.close()
        right.close()


def test_collection_far_bound_culls_without_native_queries_or_model_rebuild(
    monkeypatch,
):
    import json
    import time
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene._newton_queries import _ConvexCollection

    left, right = _ConvexCollection(), _ConvexCollection()
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    left.add_object("a", mesh)
    pose = np.eye(4)
    pose[0, 3] = 2.0
    right.add_object("b", mesh, pose)
    try:
        start = time.perf_counter_ns()
        assert left.contact_query(right, 0.0).below(0.0) is False
        first_ns = time.perf_counter_ns() - start
        query = next(iter(left._queries.values()))
        original, calls = query._pipeline.collide, []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(query._pipeline, "collide", counted)
        positions = [2.0 + index / 8 for index in range(32)]
        direct_ns, bound_ns = [], []
        for x in positions:
            pose[0, 3] = x
            right.set_transform("b", pose)
            start = time.perf_counter_ns()
            assert left.contact_query(right, 0.0).below(0.0) is False
            direct_ns.append(time.perf_counter_ns() - start)
        assert len(calls) == len(positions)
        calls.clear()
        for x in positions:
            pose[0, 3] = x
            right.set_transform("b", pose)
            start = time.perf_counter_ns()
            lower = left.clearance_lower_bound(right)
            assert 0 < lower <= x - 1.0
            bound_ns.append(time.perf_counter_ns() - start)
        assert not calls
        assert len(left._queries) == 1
        print(
            json.dumps(
                {
                    "workload": "32 frozen unit-box far pairs/CPU",
                    "first_native_ns": first_ns,
                    "native_calls": 32,
                    "bound_native_calls": 0,
                    "direct_ns": direct_ns,
                    "bound_ns": bound_ns,
                }
            )
        )
    finally:
        left.close()
        right.close()


def test_mixed_mesh_primitive_no_contact_needs_a_separation_certificate():
    import json
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    # Independent triangle shells leave a real gap, but their combined AABB
    # alone cannot certify that gap. No contact is not a mixed-route certificate.
    shells = []
    for x in (-1.0, 1.0):
        shell = trimesh.creation.box((0.5, 0.5, 0.5))
        shell.apply_translation((x, 0.0, 0.0))
        shells.append(shell)
    mesh = trimesh.util.concatenate(shells)
    checker = NewtonCollisionChecker(
        (
            CollisionBody(
                "surface",
                "surface",
                "mesh",
                vertices=tuple(map(tuple, mesh.vertices)),
                triangles=tuple(map(tuple, mesh.faces)),
            ),
            CollisionBody("sphere", "sphere", "sphere", (0.125,), robot=True),
        ),
        allow_triangle_meshes=True,
    )
    frames = {"surface": np.eye(4), "sphere": np.eye(4)}
    try:
        result = checker.contact_query(frames)
        assert result.native_contact_count == 0
        for operation in (
            lambda: result.below(0.0),
            lambda: checker.contact_pairs(frames),
            lambda: checker.check(frames),
        ):
            with pytest.raises(
                UnsupportedFeasibilityError, match="mixed mesh/primitive"
            ):
                operation()
        assert result.lower == 0.0
        assert result.clearance_unavailable_reason is not None
        json.dumps(result.metadata(), allow_nan=False)
        frames["sphere"][0, 3] = 3.0
        assert checker.contact_query(frames).below(0.0) is False
        assert checker.contact_pairs(frames) == ()
        assert checker.check(frames)[0].status == "passed"
        frames["sphere"][0, 3] = 1.25
        assert checker.contact_query(frames).below(0.0) is True
        assert checker.contact_pairs(frames) == (("surface", "sphere"),)
    finally:
        checker.close()


def test_mixed_mesh_box_overlap_cannot_be_reported_as_clear_without_native_evidence():
    import numpy as np
    import trimesh
    from embodichain.lab.sim.scene.feasibility import (
        CollisionBody,
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    # Frozen numerical dimensions and poses from the public Motion owner
    # diagnostic: a pad surface intersects the scene box by centimetres.
    mesh = trimesh.creation.box((0.1, 0.06, 0.06))
    bodies = (
        CollisionBody("origin", "origin", "sphere", (0.125,)),
        CollisionBody(
            "pad",
            "pad",
            "mesh",
            vertices=tuple(map(tuple, mesh.vertices)),
            triangles=tuple(map(tuple, mesh.faces)),
            robot=True,
        ),
        CollisionBody("scene", "scene", "box", (0.07999999821186066,) * 3),
    )
    ignored = (("origin", "pad"), ("origin", "scene"))
    checker = NewtonCollisionChecker(
        bodies, ignored_pairs=ignored, allow_triangle_meshes=True
    )
    pad, scene = np.eye(4), np.eye(4)
    pad[:3, 3] = (0.8399999737739563, 0.019999999552965164, 0.30000001192092896)
    scene[:3, 3] = (0.8500000238418579, 0.0, 0.30000001192092896)
    frames = {"origin": np.eye(4), "pad": pad, "scene": scene}
    try:
        result = checker.contact_query(frames)
        # A corrected SDK may supply contacts; neither an unavailable route nor
        # a reported overlap can become certified clear on any public entry.
        try:
            assert result.below(0.0) is True
        except UnsupportedFeasibilityError:
            assert result.clearance_unavailable_reason is not None
        try:
            assert checker.check(frames)[0].status == "failed"
        except UnsupportedFeasibilityError:
            pass
        try:
            assert checker.contact_pairs(frames) == (("pad", "scene"),)
        except UnsupportedFeasibilityError:
            pass
        assert checker.contact_pairs(frames, allowed_contacts=(("pad", "scene"),)) == ()
        assert (
            checker.check(frames, allowed_contacts=(("pad", "scene"),))[0].status
            == "passed"
        )
    finally:
        checker.close()
