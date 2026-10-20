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

"""Explicit upright Pick quality preserves undeclared callers and real pad axes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program.pick_closure_quality import (
    UprightClosurePolicy,
    bind_policies,
    declarations,
    inspect_tilt,
    pad_closing_axes,
)


@pytest.mark.parametrize("bad", [True, -1, 90, float("inf"), float("nan"), "30"])
def test_invalid_quality_does_not_silently_default(bad):
    with pytest.raises(ValueError):
        declarations({"pick": bad})


@pytest.mark.parametrize("up", [(0, 0, 0), (0, float("nan"), 1), (1, 0)])
def test_up_axis_must_be_measured_finite_direction(up):
    with pytest.raises(ValueError):
        UprightClosurePolicy(30, up)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, True),
        (29.999, True),
        (30, True),
        (30.001, False),
        (59.5, False),
        (90, False),
    ],
)
def test_closure_tilt_boundary_and_symmetric_roll(angle, expected):
    policy = UprightClosurePolicy(30)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    for sign in (1, -1):
        result = inspect_tilt(pose, np.array([[sign, 0.0, 0.0]]), policy)
        assert result["valid"] is expected
        assert result["maximum_closure_tilt_degrees"] == pytest.approx(angle)


def test_named_native_pad_axes_do_not_assume_tcp_x():
    mesh = SimpleNamespace(
        vertices=np.array([[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]])
    )
    pads = []
    for sign in (-1, 1):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, 2, 3] = sign * np.array([0.03, 0.02, 0.01])
        pads.append(SimpleNamespace(pad=True, meshes=(mesh,), tcp_poses=poses))
    axes = pad_closing_axes(tuple(pads))
    assert np.allclose(axes, [0, 0, 1])
    assert not inspect_tilt(np.eye(4), axes, UprightClosurePolicy(30))["valid"]


def test_invalid_or_coincident_pad_geometry_is_not_an_x_axis_fallback():
    mesh = SimpleNamespace(vertices=np.array([[-0.01] * 3, [0.01] * 3]))
    part = SimpleNamespace(pad=True, meshes=(mesh,), tcp_poses=np.eye(4)[None])
    with pytest.raises(ValueError, match="degenerate"):
        pad_closing_axes((part, part))


def test_policy_binds_only_declared_exact_pour_invocation():
    from embodichain.lab.task_program.semantics import Pick, SceneObjectRef

    call = SimpleNamespace(call=Pick(SceneObjectRef("container")), segment_call_index=0)
    segments = [
        SimpleNamespace(name=name, segment_id=name, calls=(call,))
        for name in ("ordinary", "pour", "other_pour")
    ]
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    purposes = {
        "p/ordinary:0": "ordinary",
        "p/pour:0": "pour",
        "p/other_pour:0": "pour",
    }
    assert bind_policies({}, program, purposes) == ()
    result = bind_policies({"pour": 30}, program, purposes)
    assert result == (("p/pour:0", UprightClosurePolicy(30)),)
    with pytest.raises(ValueError, match="purpose"):
        bind_policies({"ordinary": 30}, program, purposes)
    with pytest.raises(ValueError, match="missing"):
        bind_policies({"unknown": 30}, program, purposes)
    segments[1].calls = (SimpleNamespace(call=object(), segment_call_index=0),)
    with pytest.raises(ValueError, match="exact Pick"):
        bind_policies({"pour": 30}, program, purposes)


def test_generated_quality_is_opt_in_and_rejects_unrelated_calls():
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _task_stability_payload,
    )

    graph = {
        "targets": {},
        "nodes": [
            {
                "id": name,
                "task_instance_id": name,
                "task_type": kind,
                "call": {"kind": "pick", "object": "container"},
            }
            for name, kind in (("ordinary", "E1"), ("pour", "E3"))
        ],
        "task_groups": [
            {"id": name, "node_ids": [name]} for name in ("ordinary", "pour")
        ],
    }
    args = (
        graph,
        SimpleNamespace(planner_objects=()),
        {"skill_profile": {"resources": []}},
    )
    baseline = _task_stability_payload(*args, pick_contact_revision=4)
    assert "upright_container_picks" not in baseline
    assert (
        _task_stability_payload(
            *args, pick_contact_revision=4, upright_container_picks={}
        )
        == baseline
    )
    declared = _task_stability_payload(
        *args, pick_contact_revision=4, upright_container_picks={"pour": 30}
    )
    assert declared == {**baseline, "upright_container_picks": {"pour": 30}}
    for name in ("ordinary", "missing"):
        with pytest.raises(ValueError, match="explicit E3"):
            _task_stability_payload(
                *args, pick_contact_revision=4, upright_container_picks={name: 30}
            )
    with pytest.raises(ValueError, match="native Pick contact"):
        _task_stability_payload(*args, upright_container_picks={"pour": 30})
