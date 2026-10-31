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

"""Sampled collision checks for annotated, fixed-geometry manipulation.

The host owns the actual scene and supplies one world pose per object and
sample. In particular, held and released object poses must come from the
appropriate grasp relation and release policy, or from measured simulation
state. The checker does not infer a missing object path or advance physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "BoxCollisionBody",
    "CollisionPhase",
    "PlaneCollisionBody",
    "SampledCollisionReport",
    "SampledRobotCollisionChecker",
]


def _pose(value: object, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-8, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a proper rigid transform")
    return np.array(pose, copy=True)


def _pair(value: object) -> tuple[str, str]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or value[0] == value[1]
    ):
        raise ValueError("collision pairs must contain two distinct entity IDs")
    return tuple(sorted(value))


@dataclass(frozen=True)
class BoxCollisionBody:
    """One physical box attached to a scene body, in metres.

    ``uid`` is the scene body ID; its collision entity ID is ``body:<uid>``.
    The host must obtain dimensions and local pose from physical geometry,
    rather than a visual mesh. Static-static pairs are not evaluated.
    """

    uid: str
    half_extents: tuple[float, float, float]
    local_pose: object
    is_static: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.uid, str)
            or not self.uid.strip()
            or self.uid != self.uid.strip()
        ):
            raise ValueError("body uid must be a nonempty stripped string")
        values = np.asarray(self.half_extents, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("box half extents must be three positive finite values")
        if type(self.is_static) is not bool:
            raise TypeError("is_static must be a boolean")
        pose = _pose(self.local_pose, "body local_pose")
        pose.setflags(write=False)
        object.__setattr__(self, "half_extents", tuple(float(x) for x in values))
        object.__setattr__(self, "local_pose", pose)

    @classmethod
    def from_rigid_object(cls, obj: object, *, env_id: int = 0) -> BoxCollisionBody:
        """Read a single physical box from a live scene object.

        Other shapes or multi-shape bodies are rejected; replacing them with
        a guessed bounding box would not establish collision compatibility.
        """
        getter = getattr(obj, "get_collision_shapes", None)
        if not callable(getter):
            raise TypeError("object must expose physical collision shapes")
        shapes = tuple(getter(env_id=env_id))
        if len(shapes) != 1 or getattr(shapes[0].shape_type, "name", None) != "BOX":
            raise ValueError("one physical box shape is required")
        shape = shapes[0]
        if shape.half_extents is None:
            raise ValueError("physical box dimensions are unavailable")

        def array(value: object) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)

        return cls(
            uid=obj.uid,
            half_extents=tuple(array(shape.half_extents).tolist()),
            local_pose=array(shape.local_pose),
            is_static=obj.is_static,
        )

    @property
    def geometry_sha256(self) -> str:
        """Identify box geometry independent of its pose, role and scene ID."""
        digest = hashlib.sha256(b"embodichain-real2sim-box-v1\0")
        digest.update(np.asarray(self.half_extents, dtype="<f8").tobytes())
        digest.update(np.asarray(self.local_pose, dtype="<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class PlaneCollisionBody:
    """A fixed plane with occupied half-space along its local negative Z."""

    uid: str
    local_pose: object
    is_static: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.uid, str)
            or not self.uid.strip()
            or self.uid != self.uid.strip()
        ):
            raise ValueError("body uid must be a nonempty stripped string")
        if self.is_static is not True:
            raise ValueError("the initial plane support must be static")
        pose = _pose(self.local_pose, "plane local_pose")
        pose.setflags(write=False)
        object.__setattr__(self, "local_pose", pose)


@dataclass(frozen=True)
class CollisionPhase:
    """A half-open interval with explicit allowed contact pairs and depths.

    Pair names use ``robot:<URDF link>`` and ``body:<scene uid>``. Contact
    exemptions are symmetric and only apply within this phase. Maximum
    penetration is measured by the collision backend in metres.
    """

    name: str
    start: int
    stop: int
    allowed_contacts: Mapping[tuple[str, str], float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("collision phase needs a nonempty name")
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("collision phase bounds must be a nonempty interval")
        if not isinstance(self.allowed_contacts, Mapping):
            raise TypeError("allowed_contacts must map entity pairs to depth limits")
        result = {}
        for raw_pair, raw_depth in self.allowed_contacts.items():
            pair = _pair(raw_pair)
            if pair in result:
                raise ValueError("collision contact pair is duplicated")
            if (
                isinstance(raw_depth, bool)
                or not isinstance(raw_depth, (int, float))
                or not np.isfinite(raw_depth)
                or raw_depth < 0
            ):
                raise ValueError("contact depth limits must be finite and nonnegative")
            result[pair] = float(raw_depth)
        from types import MappingProxyType

        object.__setattr__(self, "allowed_contacts", MappingProxyType(result))


@dataclass(frozen=True)
class SampledCollisionReport:
    """Result for the supplied samples; no continuous-path claim is made."""

    accepted: bool
    sample_count: int
    geometry_sha256: str
    first_failure_sample: int | None = None
    first_failure_phase: str | None = None
    first_failure_pair: tuple[str, str] | None = None
    first_failure_penetration_m: float | None = None
    reason: str | None = None
    allowed_contact_samples: int = 0


class SampledRobotCollisionChecker:
    """Check full robot and scene geometry at B=1 trajectory samples.

    Robot collision meshes are read once from the full assembled URDF. The
    sampled joint names must exactly match its scalar joints. Same-rigid-body
    and directly connected joint pairs are excluded from self-collision;
    those topology exclusions are available in ``excluded_self_pairs``.
    All other robot self pairs and robot/scene pairs are evaluated. Object
    paths are explicit per-sample world poses and are never silently frozen.

    Args:
        urdf_path: Full fixed-base robot URDF including its end effector.
        joint_names: Complete simulator joint order, including mimic joints.
        bodies: Physical scene boxes and fixed planes to include in each sample.
        world_from_base: Fixed world pose of the URDF root.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        joint_names: Sequence[str],
        bodies: Sequence[BoxCollisionBody | PlaneCollisionBody],
        world_from_base: object,
    ) -> None:
        import coal
        import pinocchio as pin

        path = Path(urdf_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("robot URDF must be a file")
        self._pin, self._coal = pin, coal
        self._model, self._geometry, _ = pin.buildModelsFromUrdf(
            str(path), package_dirs=[str(path.parent)]
        )
        self._data = self._model.createData()
        self._geometry_data = self._geometry.createData()
        names = tuple(joint_names)
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("joint names must be unique nonempty strings")
        model_names = tuple(self._model.names[1:])
        if len(names) != self._model.nq or set(names) != set(model_names):
            raise ValueError("joint names must cover every scalar URDF joint exactly")
        if any(joint.nq != 1 or joint.nv != 1 for joint in self._model.joints[1:]):
            raise ValueError("collision checker supports scalar fixed-base joints only")
        self._joint_indices = np.array(
            [names.index(name) for name in model_names], dtype=np.int64
        )
        self._joint_names = names
        self._joint_count = len(names)
        self._base = _pose(world_from_base, "world_from_base")
        body_values = tuple(bodies)
        if not body_values or any(
            not isinstance(body, (BoxCollisionBody, PlaneCollisionBody))
            for body in body_values
        ):
            raise ValueError("bodies must contain physical box or plane descriptions")
        if len({body.uid for body in body_values}) != len(body_values):
            raise ValueError("collision body IDs must be unique")
        self._bodies = body_values
        self._body_geometry = tuple(
            (
                coal.Box(*(2 * np.asarray(body.half_extents)))
                if isinstance(body, BoxCollisionBody)
                else coal.Halfspace(np.array([0.0, 0.0, 1.0]), 0.0)
            )
            for body in body_values
        )
        if not self._geometry.geometryObjects:
            raise ValueError("robot URDF contains no collision geometry")
        self._links = tuple(
            "robot:" + self._model.frames[shape.parentFrame].name
            for shape in self._geometry.geometryObjects
        )
        self._entity_ids = frozenset(self._links) | frozenset(
            "body:" + body.uid for body in body_values
        )
        self_pairs, excluded = [], set()
        for first, a in enumerate(self._geometry.geometryObjects):
            for second in range(first):
                b = self._geometry.geometryObjects[second]
                pair = (
                    _pair((self._links[first], self._links[second]))
                    if self._links[first] != self._links[second]
                    else None
                )
                ja, jb = a.parentJoint, b.parentJoint
                if (
                    ja == jb
                    or self._model.parents[ja] == jb
                    or self._model.parents[jb] == ja
                ):
                    if pair is not None:
                        excluded.add(pair)
                    continue
                if pair is not None:
                    self_pairs.append((first, second, pair))
        self._self_pairs = tuple(self_pairs)
        self.excluded_self_pairs = tuple(sorted(excluded))
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        primitive_paths = {
            "BOX",
            "SPHERE",
            "CAPSULE",
            "CYLINDER",
            "CONE",
            "PLANE",
            "ELLIPSOID",
        }
        mesh_paths = sorted(
            {
                Path(item.meshPath).resolve(strict=True)
                for item in self._geometry.geometryObjects
                if item.meshPath and item.meshPath not in primitive_paths
            }
        )
        for mesh in mesh_paths:
            digest.update(mesh.name.encode("utf-8"))
            digest.update(hashlib.sha256(mesh.read_bytes()).digest())
        self.geometry_sha256 = digest.hexdigest()

    @property
    def bodies(self) -> tuple[BoxCollisionBody | PlaneCollisionBody, ...]:
        """Return the immutable physical geometry declarations."""
        return self._bodies

    def validate_source_geometry(
        self, source: object, *, object_uid: str, target_uid: str | None = None
    ) -> None:
        """Require source geometry identities to match this live model.

        The source must have been captured with independent asset identities;
        joint-name or pose equality is not sufficient. Legacy unqualified
        sources are rejected and can still be used in explicit diagnostics.
        """
        from embodichain.data_pipeline.real2sim import Real2SimTrajectory

        if not isinstance(source, Real2SimTrajectory):
            raise TypeError("source must be a Real2SimTrajectory")
        bodies = {body.uid: body for body in self._bodies}
        if object_uid not in bodies or (
            target_uid is not None and target_uid not in bodies
        ):
            raise ValueError("source geometry references an unknown scene body")
        if not isinstance(bodies[object_uid], BoxCollisionBody) or (
            target_uid is not None
            and not isinstance(bodies[target_uid], BoxCollisionBody)
        ):
            raise ValueError("source object and target geometry must be physical boxes")
        expected = {
            "robot_geometry_sha256": self.geometry_sha256,
            "object_geometry_sha256": bodies[object_uid].geometry_sha256,
        }
        if source.target_pose is not None:
            if target_uid is None:
                raise ValueError("source target geometry requires a target body")
            expected["target_geometry_sha256"] = bodies[target_uid].geometry_sha256
        for name, actual in expected.items():
            recorded = getattr(source, name)
            if recorded is None:
                raise ValueError(f"source {name} is unverified")
            if recorded != actual:
                raise ValueError(
                    f"source {name} does not match current physical geometry"
                )

    def first_link_body_contact(
        self,
        start: object,
        end: object,
        *,
        link: str,
        body_uid: str,
        body_pose: object,
        iterations: int = 24,
    ) -> float:
        """Locate the first sampled contact along a joint sweep.

        This is a geometric estimate for a grasp command, not evidence that a
        physical controller will attain or maintain contact. The caller must
        Search uses 64 coarse intervals followed by bisection. Contacts
        entering and leaving between coarse samples may not be detected.

        Returns:
            The last non-contact fraction in ``[0, 1]``.

        Raises:
            ValueError: The pair, input, or endpoint contact is unavailable.
        """
        values = []
        for value in (start, end):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (self._joint_count,) or not np.isfinite(array).all():
                raise ValueError("contact sweep must contain finite full-joint states")
            values.append(array)
        if type(iterations) is not int or not 1 <= iterations <= 32:
            raise ValueError("contact sweep iterations must be between 1 and 32")
        link_id = "robot:" + link
        link_indices = [
            index for index, name in enumerate(self._links) if name == link_id
        ]
        body_index = next(
            (index for index, body in enumerate(self._bodies) if body.uid == body_uid),
            None,
        )
        if not link_indices or body_index is None:
            raise ValueError("contact sweep references an unknown link or body")
        matrix = (
            _pose(body_pose, "contact body pose") @ self._bodies[body_index].local_pose
        )
        body_transform = self._coal.Transform3s(matrix[:3, :3], matrix[:3, 3])
        request = self._coal.CollisionRequest()
        request.num_max_contacts = 1
        result = self._coal.CollisionResult()

        def in_contact(fraction: float) -> bool:
            q = values[0] + fraction * (values[1] - values[0])
            self._pin.forwardKinematics(self._model, self._data, q[self._joint_indices])
            self._pin.updateGeometryPlacements(
                self._model, self._data, self._geometry, self._geometry_data
            )
            for index in link_indices:
                transform = self._base @ self._geometry_data.oMg[index].homogeneous
                pose = self._coal.Transform3s(transform[:3, :3], transform[:3, 3])
                result.clear()
                if self._coal.collide(
                    self._geometry.geometryObjects[index].geometry,
                    pose,
                    self._body_geometry[body_index],
                    body_transform,
                    request,
                    result,
                ):
                    return True
            return False

        if in_contact(0.0):
            return 0.0
        coarse_steps = 64
        lower, upper = 0.0, None
        for sample in range(1, coarse_steps + 1):
            candidate = sample / coarse_steps
            if in_contact(candidate):
                upper = candidate
                lower = (sample - 1) / coarse_steps
                break
        if upper is None:
            raise ValueError("gripper sweep does not reach the object")
        for _ in range(iterations):
            middle = (lower + upper) * 0.5
            if in_contact(middle):
                upper = middle
            else:
                lower = middle
        return lower

    def validate_prismatic_link(self, link: str, joint_name: str) -> None:
        """Require a declared finger to be attached to its prismatic joint."""
        if joint_name not in self._joint_names:
            raise ValueError("finger joint is absent from the complete robot layout")
        index = self._model.getJointId(joint_name)
        link_indices = [
            i for i, name in enumerate(self._links) if name == "robot:" + link
        ]
        if (
            not link_indices
            or not self._model.joints[index].shortname().startswith("JointModelP")
            or any(
                self._geometry.geometryObjects[i].parentJoint != index
                for i in link_indices
            )
        ):
            raise ValueError(
                "finger collision link must be attached to its prismatic joint"
            )

    def validate(
        self,
        joint_positions: object,
        *,
        body_poses: Mapping[str, object],
        phases: Sequence[CollisionPhase],
    ) -> SampledCollisionReport:
        """Evaluate every supplied full-joint and object pose sample.

        Body poses must be ``(T, 4, 4)`` and cover every configured body;
        callers must explicitly repeat a frozen pose. Phase intervals must
        cover every sample once. A failed result identifies the first pair
        in deterministic sample/pair order.
        """
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.detach().cpu().numpy()
        joints = np.asarray(joint_positions, dtype=np.float64)
        if (
            joints.ndim != 2
            or joints.shape[0] == 0
            or joints.shape[1] != self._joint_count
            or not np.isfinite(joints).all()
        ):
            raise ValueError("joint_positions must be finite (T, full_joint_count)")
        count = len(joints)
        if not isinstance(body_poses, Mapping) or set(body_poses) != {
            body.uid for body in self._bodies
        }:
            raise ValueError(
                "body_poses must cover exactly the configured physical bodies"
            )
        poses = {}
        for body in self._bodies:
            values = body_poses[body.uid]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (count, 4, 4):
                raise ValueError(
                    f"body {body.uid!r} requires one pose per joint sample"
                )
            poses[body.uid] = np.stack(
                [_pose(value, f"body {body.uid} pose") for value in values]
            )
        phases = tuple(phases)
        if not phases or any(not isinstance(phase, CollisionPhase) for phase in phases):
            raise ValueError("phases must contain typed collision intervals")
        previous = 0
        for phase in phases:
            if phase.start != previous or phase.stop > count:
                raise ValueError(
                    "collision phases must cover samples without gaps or overlaps"
                )
            if any(set(pair) - self._entity_ids for pair in phase.allowed_contacts):
                raise ValueError("collision exemption names an unknown entity")
            previous = phase.stop
        if previous != count:
            raise ValueError("collision phases must cover every sample")
        pin, coal = self._pin, self._coal
        request = coal.CollisionRequest()
        request.enable_contact = True
        # A fixed robot base touching a plane represented by a large box can
        # produce hundreds of mesh contacts. Treat a full buffer as unknown,
        # rather than certifying a depth from a truncated contact manifold.
        request.num_max_contacts = 5000
        result = coal.CollisionResult()
        phase_index = 0
        allowed_samples = 0

        def check(a, ta, b, tb, pair, phase):
            result.clear()
            if not coal.collide(a, ta, b, tb, request, result):
                return None
            contacts = result.getContacts()
            if not contacts:
                return (None, "collision backend returned no penetration evidence")
            if result.numContacts() >= request.num_max_contacts:
                return (None, "collision backend contact limit was reached")
            depths = [-float(contact.penetration_depth) for contact in contacts]
            if not np.isfinite(depths).all():
                return (None, "collision backend returned nonfinite penetration")
            depth = max(0.0, max(depths))
            limit = phase.allowed_contacts.get(pair)
            if limit is None:
                return (depth, "contact pair is not allowed in this phase")
            if depth > limit + 1e-9:
                return (depth, "allowed contact exceeds its penetration limit")
            return True

        for sample, q in enumerate(joints):
            while sample >= phases[phase_index].stop:
                phase_index += 1
            phase = phases[phase_index]
            pin.forwardKinematics(self._model, self._data, q[self._joint_indices])
            pin.updateGeometryPlacements(
                self._model, self._data, self._geometry, self._geometry_data
            )
            link_transforms = []
            for place in self._geometry_data.oMg:
                matrix = self._base @ place.homogeneous
                link_transforms.append(coal.Transform3s(matrix[:3, :3], matrix[:3, 3]))
            object_transforms = []
            for body in self._bodies:
                matrix = poses[body.uid][sample] @ body.local_pose
                object_transforms.append(
                    coal.Transform3s(matrix[:3, :3], matrix[:3, 3])
                )
            pairs = [
                (
                    self._geometry.geometryObjects[a].geometry,
                    link_transforms[a],
                    self._geometry.geometryObjects[b].geometry,
                    link_transforms[b],
                    pair,
                )
                for a, b, pair in self._self_pairs
            ]
            pairs.extend(
                (
                    link.geometry,
                    link_transforms[i],
                    self._body_geometry[j],
                    object_transforms[j],
                    _pair((self._links[i], "body:" + body.uid)),
                )
                for i, link in enumerate(self._geometry.geometryObjects)
                for j, body in enumerate(self._bodies)
            )
            pairs.extend(
                (
                    self._body_geometry[i],
                    object_transforms[i],
                    self._body_geometry[j],
                    object_transforms[j],
                    _pair(("body:" + a.uid, "body:" + b.uid)),
                )
                for i, a in enumerate(self._bodies)
                for j, b in enumerate(self._bodies[:i])
                if not (a.is_static and b.is_static)
            )
            sample_has_allowed = False
            for first, first_pose, second, second_pose, pair in pairs:
                verdict = check(first, first_pose, second, second_pose, pair, phase)
                if verdict is True:
                    sample_has_allowed = True
                elif verdict is not None:
                    depth, reason = verdict
                    return SampledCollisionReport(
                        False,
                        sample + 1,
                        self.geometry_sha256,
                        sample,
                        phase.name,
                        pair,
                        depth,
                        reason,
                        allowed_samples + int(sample_has_allowed),
                    )
            allowed_samples += int(sample_has_allowed)
        return SampledCollisionReport(
            True, count, self.geometry_sha256, allowed_contact_samples=allowed_samples
        )
