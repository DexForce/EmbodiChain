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
import json
from pathlib import Path
import pytest
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.scene.native_hand_scene import (
    NativeSceneBody,
    NativeScenePlane,
    NativeHandSceneQuery,
)
from ._native_hand_scene_fixture import clearance_case


@pytest.mark.parametrize(
    "mode",
    [
        "plane_clear",
        "plane_envelope",
        "mesh_clear",
        "mesh_envelope",
        "mesh_intersection",
        "later_frame",
        "ordered_obstacles",
        "mixed",
    ],
)
def test_native_hand_scene_keeps_analytic_contact_order_with_native_backend(mode):
    parts, obstacles, poses = clearance_case(NativeSceneBody, NativeScenePlane, mode)
    with NativeHandSceneQuery(
        scope=NativeQueryScope("case-" + mode, 1.25, 3),
        robot_uid="robot",
        joint_names=("joint",),
        root_pose=parts[0].pose,
        parts=parts,
        obstacles=obstacles,
        inventory=[],
        native_revision=7,
        read_native_revision=lambda: 7,
    ) as query:
        actual = query.inspect(poses)
    # Boxes/planes and the supplied path give these independent analytic cases;
    # the archived FCL distances remain historical and are not native goldens.
    assert actual["valid"] is (mode in {"plane_clear", "mesh_clear"})
    assert actual["collision_backend"] == "newton.CollisionPipeline"
    assert "fcl_queries" not in actual
    if not actual["valid"]:
        assert actual["frame"] == (1 if mode == "later_frame" else 0)
        assert actual["receiver_link"] == "finger"
        if mode == "ordered_obstacles":
            assert actual["obstacle_name"] == "first"
    if mode.startswith("plane") or mode == "mixed":
        assert actual["plane_queries"] == 1
    assert query.closed and query.link_names == ("finger",)


def test_native_hand_scene_rejects_revision_drift_and_closes_only_its_own_query():
    parts, obstacles, poses = clearance_case(
        NativeSceneBody, NativeScenePlane, "plane_clear"
    )
    state = [7]

    def make(identity):
        return NativeHandSceneQuery(
            scope=NativeQueryScope(identity, 1.25, 3),
            robot_uid="robot",
            joint_names=("joint",),
            root_pose=parts[0].pose,
            parts=parts,
            obstacles=obstacles,
            inventory=[{"owner": "floor"}],
            native_revision=7,
            read_native_revision=lambda: state[0],
        )

    first, second = make("first"), make("second")
    detached = first.inventory
    detached[0]["owner"] = "changed"
    assert first.inventory == [{"owner": "floor"}]
    state[0] = 8
    with pytest.raises(ValueError, match="revision changed"):
        first.inspect(poses)
    first.close()
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.inspect(poses)
    state[0] = 7
    assert second.inspect(poses)["valid"]
    second.close()


@pytest.fixture
def native_ports(tmp_path, monkeypatch):
    import dexsim.engine
    from embodichain.lab.sim.objects.backends import collision as shape_decoder
    from ._native_capture_fixture import Box, Convex
    from ._native_hand_scene_fixture import Plane, capture_case

    monkeypatch.setattr(dexsim.engine, "BoxGeometry", Box)
    monkeypatch.setattr(shape_decoder, "BoxGeometry", Box)
    monkeypatch.setattr(dexsim.engine, "ConvexMeshGeometry", Convex)
    monkeypatch.setattr(shape_decoder, "ConvexMeshGeometry", Convex)
    monkeypatch.setattr(dexsim.engine, "PlaneGeometry", Plane)
    return capture_case(tmp_path)


def capture_arguments(robot, reader=lambda: 7):
    return dict(
        scope=NativeQueryScope("capture", 2.5, 4),
        expected_qpos=robot.qpos.clone(),
        root_pose=robot.pose.clone(),
        receiver_control_part="arm",
        receiver_joint_ids=(2, 1),
        required_owner_uids=("source", "support", "cabinet", "plane"),
        excluded_owner_uid="source",
        native_revision=7,
        read_native_revision=reader,
    )


def test_native_scene_capture_retains_collision_only_links_planes_and_effective_scale(
    native_ports,
):
    from embodichain.lab.sim.scene.native_hand_scene import capture_native_hand_scene
    from ._native_hand_scene_fixture import capture_record

    robot, simulation = native_ports
    with capture_native_hand_scene(
        robot, simulation, **capture_arguments(robot)
    ) as query:
        actual = capture_record(query, robot)
        expected = json.loads(
            (
                Path(__file__).parent / "data/native_hand_scene_capture_reference.json"
            ).read_text()
        )
        assert actual == expected
        assert query.link_names == (
            "hand_base",
            "negative_finger_pad",
            "positive_finger_pad",
        )
    assert query.closed


def test_native_scene_capture_checks_revision_after_reads_before_native_queries(
    native_ports, monkeypatch
):
    from embodichain.lab.sim.scene import native_hand_scene as module
    from embodichain.lab.sim.scene.native_hand_scene import capture_native_hand_scene

    robot, simulation = native_ports
    revisions = iter([7, 8])

    def forbidden():
        raise AssertionError("No native resources before snapshot qualification")

    monkeypatch.setattr(module, "_ConvexCollection", forbidden)
    with pytest.raises(ValueError, match="revision changed"):
        capture_native_hand_scene(
            robot, simulation, **capture_arguments(robot, lambda: next(revisions))
        )
    assert robot.reads == 2 and len(robot.calls) == 1


def test_native_hand_capture_requires_complete_moving_joint_identity(native_ports):
    from embodichain.lab.sim.scene.native_hand_scene import capture_native_hand_scene

    robot, simulation = native_ports
    args = capture_arguments(robot)
    args["receiver_joint_ids"] = (2,)
    with pytest.raises(ValueError, match="all_moving_joints_including_mimics"):
        capture_native_hand_scene(robot, simulation, **args)
    assert not robot.calls


def test_native_hand_capture_rejects_same_uid_from_another_robot_owner(native_ports):
    from copy import copy
    from embodichain.lab.sim.scene.native_hand_scene import capture_native_hand_scene

    robot, simulation = native_ports
    foreign = copy(robot)
    with pytest.raises(ValueError, match="exact registered robot instance"):
        capture_native_hand_scene(foreign, simulation, **capture_arguments(robot))
    assert robot.reads == 0 and not robot.calls


def test_native_hand_constructor_releases_prefix_when_native_geometry_is_unsupported(
    monkeypatch,
):
    from embodichain.lab.sim.scene import native_hand_scene as module
    from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError

    class Groups:
        owned = []

        def __init__(self, *, geometry_kind):
            assert geometry_kind == "mesh"
            self.closed = False
            self.owned.append(self)

        def add_object(self, *args, **kwargs):
            if len(self.owned) == 2:
                raise UnsupportedFeasibilityError("native topology unavailable")

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "_ConvexCollection", Groups)
    parts, obstacles, _ = clearance_case(
        NativeSceneBody, NativeScenePlane, "mesh_envelope"
    )
    with pytest.raises(UnsupportedFeasibilityError, match="topology"):
        NativeHandSceneQuery(
            scope=NativeQueryScope("failed-capture", 1.0, 0),
            robot_uid="robot",
            joint_names=("joint",),
            root_pose=parts[0].pose,
            parts=parts,
            obstacles=obstacles,
            inventory=[],
            native_revision=7,
            read_native_revision=lambda: 7,
        )
    assert len(Groups.owned) == 2 and all(group.closed for group in Groups.owned)


def test_native_hand_near_threshold_cannot_bypass_conversion_uncertainty(monkeypatch):
    import numpy as np
    from embodichain.lab.sim.scene import native_hand_scene as module
    from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError

    calls = []

    class Groups:
        def __init__(self, *, geometry_kind):
            assert geometry_kind == "mesh"

        @staticmethod
        def metadata():
            return {"collision_backend": "newton.CollisionPipeline"}

        def add_object(self, *args, **kwargs):
            pass

        def set_transform(self, *args):
            pass

        def contact_query(self, other, limit):
            calls.append(limit)
            raise UnsupportedFeasibilityError("native conversion spans threshold")

        def close(self):
            pass

    monkeypatch.setattr(module, "_ConvexCollection", Groups)
    parts, obstacles, poses = clearance_case(
        NativeSceneBody, NativeScenePlane, "mesh_envelope"
    )
    threshold = parts[0].contact_offset + obstacles[0].contact_offset
    # Raw float64 AABBs are separated by threshold+1nm. The0.01m vertex
    # coordinate is not exactly float32-representable, so this is no proof
    # that the native query's conversion interval can classify the threshold.
    assert float(np.float32(0.01)) != 0.01
    poses[0, 0, 0, 3] = 0.2 - 0.02 - threshold - 1e-9
    with NativeHandSceneQuery(
        scope=NativeQueryScope("near-contact", 0.0, 0),
        robot_uid="robot",
        joint_names=("joint",),
        root_pose=parts[0].pose,
        parts=parts,
        obstacles=obstacles,
        inventory=[],
        native_revision=7,
        read_native_revision=lambda: 7,
    ) as query:
        with pytest.raises(UnsupportedFeasibilityError, match="conversion"):
            query.inspect(poses)
    assert calls == [threshold]
