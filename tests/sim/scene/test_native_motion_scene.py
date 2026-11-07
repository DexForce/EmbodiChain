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

"""Native motion rest queries retain frame order, offsets and observed ownership."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
import torch
import trimesh
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.scene import native_motion_scene as module
from embodichain.lab.sim.scene.native_motion_scene import (
    NativeRestCollisionPart,
    NativeRestSceneBody,
    NativeRestSceneQuery,
    capture_native_rest_scene,
    capture_native_arm_parts,
    native_path_link_poses,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/native_motion_scene_b0.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_native_rest_scene_preserves_analytic_box_decisions_and_first_pair(case):
    parts = tuple(
        NativeRestCollisionPart(
            row["name"], (trimesh.creation.box(row["extents"]),), row["rest"]
        )
        for row in case["moving"]
    )
    bodies = tuple(
        NativeRestSceneBody(
            NativeRestCollisionPart(
                row["name"], (trimesh.creation.box(row["extents"]),), row["rest"]
            ),
            np.asarray(row["pose"]),
        )
        for row in case["obstacles"]
    )
    with NativeRestSceneQuery(
        bodies,
        scope=NativeQueryScope("rest", 0.0, 0),
        validate_observation=lambda: None,
    ) as query:
        if case["name"] == "later_frame":
            # Before the later penetration, this input visits exactly the
            # pair's5mm rest boundary. Native representation uncertainty must
            # remain unavailable rather than forcing the historical FCL verdict.
            from embodichain.lab.sim.scene.feasibility import (
                UnsupportedFeasibilityError,
            )

            with pytest.raises(
                UnsupportedFeasibilityError, match="canonical AABB separation"
            ):
                query.inspect(
                    parts,
                    np.asarray(case["poses"]),
                    case["source_id"],
                    source_contact_from=case["source_contact_from"],
                )
            return
        result = query.inspect(
            parts,
            np.asarray(case["poses"]),
            case["source_id"],
            source_contact_from=case["source_contact_from"],
        )
    # Original fixtures specify axis-aligned box geometry. Reuse only their
    # input paths, not the old backend's scalar distances or query metadata.
    assert result["valid"] is (case["name"] in {"clear", "contact_deferred"})
    assert result["collision_backend"] == "newton.CollisionPipeline"
    if case["name"] == "clear":
        assert result["queries"] == 0
        assert result["broadphase_culled_pairs"] > 0
    if not result["valid"]:
        assert result["frame"] == (3 if case["name"] == "later_frame" else 0)
        assert result["link"] == "pad"
        assert result["obstacle"] == (
            "target" if case["name"] == "contact_early" else "wall"
        )
        assert result["required_clearance_m"] == 0.005
    assert query.closed


def fixture(tmp_path, monkeypatch):
    mesh = trimesh.creation.box((0.04, 0.04, 0.04))
    body = lambda: NS(meshes=(mesh,), get_rest_offset=lambda: 0.001)
    robotpose = torch.eye(4)[None]
    root = tmp_path / "arm.urdf"
    root.write_text(
        '<robot name="arm"><link name="root"><collision/></link><link name="arm"><collision/></link><link name="hand"><collision/></link><joint name="joint" type="revolute"><parent link="root"/><child link="arm"/></joint><joint name="fixed" type="fixed"><parent link="arm"/><child link="hand"/></joint></robot>'
    )
    robot = NS(
        uid="robot",
        cfg=NS(fpath=str(root)),
        joint_names=("joint",),
        get_solver=lambda part: NS(root_link_name="root"),
        get_qpos=lambda: torch.zeros(1, 1),
        get_local_pose=lambda **kw: robotpose,
    )
    robot.compute_fk = (
        lambda *, qpos, link_names, qpos_joint_names: torch.eye(4)
        .expand(len(qpos), len(link_names), 4, 4)
        .clone()
    )
    native_robot = NS(
        get_name=lambda: "robot",
        arena_name="arena",
        get_link_names=lambda: ("root", "arm", "hand"),
        render_articulation=NS(get_physical_body=lambda link: body()),
        get_link_pose=lambda name: np.eye(4),
    )
    rigid_handle = NS(
        get_name=lambda: "rigid",
        arena_name="arena",
        native=lambda: NS(get_physical_body=body),
    )
    articulation = NS(
        get_name=lambda: "cabinet",
        arena_name="arena",
        get_link_names=lambda: ("base", "drawer"),
        render_articulation=NS(get_physical_body=lambda link: body()),
        get_link_pose=lambda name: np.eye(4),
    )
    handles = {"robot": native_robot, "rigid": rigid_handle, "cabinet": articulation}
    sim = NS(
        num_envs=1,
        is_default_backend=True,
        get_robot=lambda uid: robot,
        spawn_result=NS(
            get_paths=lambda: tuple(handles), get=lambda path: handles[path]
        ),
        get_rigid_object_uid_list=lambda: ("rigid",),
        get_articulation_uid_list=lambda: ("cabinet",),
        get_rigid_object=lambda uid: NS(get_local_pose=lambda **kw: robotpose),
    )
    monkeypatch.setattr(module, "_native_meshes", lambda selected: selected.meshes)
    return robot, sim, handles


def test_native_scene_capture_keeps_physical_order_and_arm_topology(
    tmp_path, monkeypatch
):
    robot, sim, _ = fixture(tmp_path, monkeypatch)
    checks = []

    def guard():
        checks.append(True)

    query = capture_native_rest_scene(
        sim,
        robot,
        scope=NativeQueryScope("capture", 0.0, 0),
        validate_observation=guard,
    )
    assert [value.part.name for value in query.bodies] == [
        "rigid",
        "cabinet:base",
        "cabinet:drawer",
    ]
    hand = (
        NativeRestCollisionPart(
            "hand", (trimesh.creation.box((0.01, 0.01, 0.01)),), 0.001
        ),
    )
    parts = capture_native_arm_parts(
        sim, robot, "custom_arm", hand, validate_observation=guard
    )
    assert [value.name for value in parts] == ["arm", "root"]
    poses = native_path_link_poses(robot, torch.zeros(3, 1), parts)
    assert poses.shape == (3, 2, 4, 4) and poses.dtype == np.float64
    query.close()
    assert checks


def test_native_query_rejects_original_observation_drift_and_isolation():
    state = [True]

    def guard():
        if not state[0]:
            raise ValueError("original observation changed")

    body = NativeRestSceneBody(
        NativeRestCollisionPart("body", (trimesh.creation.box((1, 1, 1)),), 0.001),
        np.eye(4),
    )
    first = NativeRestSceneQuery(
        (body,), scope=NativeQueryScope("one", 0.0, 0), validate_observation=guard
    )
    second = NativeRestSceneQuery(
        (body,),
        scope=NativeQueryScope("two", 0.0, 0),
        validate_observation=lambda: None,
    )
    state[0] = False
    with pytest.raises(ValueError, match="original observation"):
        first.inspect((body.part,), np.eye(4)[None, None], "body")
    first.close()
    assert second.inspect((body.part,), np.eye(4)[None, None], "body")["valid"]
    second.close()
    with pytest.raises(RuntimeError, match="closed"):
        second.assert_current()


def test_native_capture_does_not_rebind_replaced_physical_handles(
    tmp_path, monkeypatch
):
    robot, sim, handles = fixture(tmp_path, monkeypatch)
    query = capture_native_rest_scene(
        sim,
        robot,
        scope=NativeQueryScope("capture", 0.0, 0),
        validate_observation=lambda: None,
    )
    handles["rigid"] = NS()
    with pytest.raises(ValueError, match="inventory changed"):
        query.assert_current()
    query.close()


@pytest.mark.parametrize("arm_capture", [False, True])
@pytest.mark.parametrize("unsupported", ["batch", "backend"])
def test_native_capture_rejects_unsupported_scope_before_native_reads(
    tmp_path, monkeypatch, arm_capture, unsupported
):
    robot, sim, _ = fixture(tmp_path, monkeypatch)
    if unsupported == "batch":
        sim.num_envs = 2
    else:
        sim.is_default_backend = False

    def forbidden(uid):
        raise AssertionError("Unsupported scope must reject before native owner lookup")

    sim.get_robot = forbidden
    with pytest.raises(ValueError, match="exact Default/B1 Robot owner"):
        if arm_capture:
            capture_native_arm_parts(
                sim, robot, "custom_arm", (), validate_observation=lambda: None
            )
        else:
            capture_native_rest_scene(
                sim,
                robot,
                scope=NativeQueryScope("unsupported", 0.0, 0),
                validate_observation=lambda: None,
            )


def test_native_rest_query_closes_all_owned_groups_when_precision_is_unavailable(
    monkeypatch,
):
    from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError

    class Groups:
        owned = []

        def __init__(self, *, geometry_kind):
            assert geometry_kind == "mesh"
            self.closed = False
            self.owned.append(self)

        def add_object(self, *args, **kwargs):
            pass

        def set_transform(self, *args):
            pass

        def clearance_lower_bound(self, other):
            return None

        def contact_query(self, other, limit):
            assert limit == 0.0
            raise UnsupportedFeasibilityError("threshold precision unavailable")

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "_ConvexCollection", Groups)
    mesh = trimesh.creation.box((0.5, 0.5, 0.5))
    part = NativeRestCollisionPart("moving", (mesh,), 0.0)
    obstacle = NativeRestSceneBody(
        NativeRestCollisionPart("obstacle", (mesh,), 0.0), np.eye(4)
    )
    with NativeRestSceneQuery(
        (obstacle,),
        scope=NativeQueryScope("precision", 0.0, 0),
        validate_observation=lambda: None,
    ) as query:
        with pytest.raises(UnsupportedFeasibilityError, match="precision"):
            query.inspect((part,), np.eye(4)[None, None], "not-excluded")
        assert not Groups.owned[0].closed  # This query still owns its obstacle.
        assert Groups.owned[1].closed  # The failed call released its moving group.
    assert all(group.closed for group in Groups.owned)


def test_native_rest_mesh_gap_without_volume_evidence_remains_unavailable():
    # Preserve both original triangle components and their 0.75m geometric gap.
    # No native contact inside the 0.03m band does not exclude deep containment
    # when their group AABBs overlap, so this cannot certify solid clearance.
    from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError

    left = trimesh.creation.box((0.25, 0.25, 0.25))
    right = left.copy()
    left.apply_translation((-1.0, 0.0, 0.0))
    right.apply_translation((1.0, 0.0, 0.0))
    split = trimesh.util.concatenate((left, right))
    assert not split.is_convex
    part = NativeRestCollisionPart(
        "moving", (trimesh.creation.box((0.25, 0.25, 0.25)),), 0.01
    )
    obstacle = NativeRestSceneBody(
        NativeRestCollisionPart("split", (split,), 0.01), np.eye(4)
    )
    with NativeRestSceneQuery(
        (obstacle,),
        scope=NativeQueryScope("mesh-gap", 0.0, 0),
        validate_observation=lambda: None,
    ) as query:
        with pytest.raises(
            UnsupportedFeasibilityError, match="canonical AABB separation"
        ):
            query.inspect((part,), np.eye(4)[None, None], "not-excluded")
        assert not query.bodies[0].part.meshes[0].is_convex
    assert query.closed
