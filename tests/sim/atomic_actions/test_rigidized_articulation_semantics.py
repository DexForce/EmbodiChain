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
"""Grasp geometry of a locked articulation must reach the affordance in the
articulation-root frame, because grounding publishes the root pose."""

from __future__ import annotations

import math

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    SceneEntity,
    create_rigidized_articulation_antipodal_semantics,
)

# A link mesh deliberately off-centre, so a wrong frame cannot cancel out.
LINK_VERTICES = torch.tensor(
    [
        [0.02, 0.0, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, 0.0, 0.04],
        [0.02, 0.03, 0.04],
    ],
    dtype=torch.float32,
)
LINK_TRIANGLES = torch.tensor(
    [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=torch.int64
)


def _transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Assemble a homogeneous transform from a rotation and a translation."""
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = rotation.to(torch.float64)
    matrix[:3, 3] = translation.to(torch.float64)
    return matrix


def _rotation_z(angle: float) -> torch.Tensor:
    """Return a rotation of ``angle`` radians about the z axis."""
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


class _StubArticulation:
    """Articulation double exposing only what the adapter is allowed to use.

    ``pk_chain`` is ``None`` because USD-backed articulations never build one;
    the adapter must then read the link pose the simulator reports.
    """

    pk_chain = None

    def __init__(
        self,
        *,
        root_to_link: torch.Tensor,
        qpos: dict[str, float],
        root_pose: torch.Tensor | None = None,
        vertices: torch.Tensor = LINK_VERTICES,
        triangles: torch.Tensor = LINK_TRIANGLES,
    ) -> None:
        self.uid = "rubiks_cube"
        self.link_names = ["lower_two_layers", "top_layer"]
        self.joint_names = list(qpos)
        self._qpos = qpos
        self._root_pose = (
            torch.eye(4, dtype=torch.float64) if root_pose is None else root_pose
        )
        self._root_to_link = root_to_link
        self._vertices = vertices
        self._triangles = triangles

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        return self._root_pose.reshape(1, 4, 4)

    def get_link_pose(self, link_name: str, to_matrix: bool = False) -> torch.Tensor:
        if link_name == "lower_two_layers":
            return self._root_pose.reshape(1, 4, 4)
        return (self._root_pose @ self._root_to_link).reshape(1, 4, 4)

    def get_link_vert_face(self, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._vertices, self._triangles

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        return torch.tensor([[self._qpos[name] for name in self.joint_names]])


class TestRootFrameTransform:
    """The mesh must be expressed in the articulation-root frame."""

    def test_non_identity_rotation_and_translation_moves_the_vertices(self) -> None:
        # A link rotated 90 degrees about z and offset from the root: the exact
        # case an identity-transform asset cannot expose.
        root_to_link = _transform(
            _rotation_z(math.pi / 2), torch.tensor([0.05, -0.01, 0.02])
        )
        cube = _StubArticulation(root_to_link=root_to_link, qpos={"top_turn": 0.0})

        semantics = create_rigidized_articulation_antipodal_semantics(
            cube,
            grasp_link="top_layer",
            locked_qpos={"top_turn": 0.0},
            label="rubiks_cube",
        )

        homogeneous = torch.cat(
            [
                LINK_VERTICES.to(torch.float64),
                torch.ones(len(LINK_VERTICES), 1, dtype=torch.float64),
            ],
            dim=1,
        )
        expected = (homogeneous @ root_to_link.transpose(0, 1))[:, :3]
        torch.testing.assert_close(
            semantics.affordance.mesh_vertices.to(torch.float64),
            expected,
            atol=1e-9,
            rtol=0,
        )
        # Guard the regression directly: the untransformed mesh is wrong here.
        assert not torch.allclose(
            semantics.affordance.mesh_vertices.to(torch.float64),
            LINK_VERTICES.to(torch.float64),
        )

    def test_identity_transform_leaves_the_vertices_unchanged(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64), qpos={"top_turn": 0.0}
        )

        semantics = create_rigidized_articulation_antipodal_semantics(
            cube,
            grasp_link="top_layer",
            locked_qpos={"top_turn": 0.0},
            label="rubiks_cube",
        )

        torch.testing.assert_close(
            semantics.affordance.mesh_vertices.to(torch.float64),
            LINK_VERTICES.to(torch.float64),
            atol=1e-9,
            rtol=0,
        )

    def test_root_pose_does_not_leak_into_object_local_geometry(self) -> None:
        # Grounding applies the root pose later, so a moved object must not
        # change the semantic mesh.
        root_pose = _transform(_rotation_z(0.4), torch.tensor([1.0, -2.0, 0.5]))
        root_to_link = _transform(
            _rotation_z(math.pi / 3), torch.tensor([0.01, 0.02, -0.03])
        )
        at_origin = _StubArticulation(root_to_link=root_to_link, qpos={"top_turn": 0.0})
        moved = _StubArticulation(
            root_to_link=root_to_link, qpos={"top_turn": 0.0}, root_pose=root_pose
        )

        first = create_rigidized_articulation_antipodal_semantics(
            at_origin,
            grasp_link="top_layer",
            locked_qpos={"top_turn": 0.0},
            label="cube",
        )
        second = create_rigidized_articulation_antipodal_semantics(
            moved, grasp_link="top_layer", locked_qpos={"top_turn": 0.0}, label="cube"
        )

        torch.testing.assert_close(
            first.affordance.mesh_vertices, second.affordance.mesh_vertices
        )


class TestIdentityAndContract:
    """Identity stays with the articulation root, not the grasp link."""

    def test_entity_id_is_the_articulation_root_uid(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64), qpos={"top_turn": 0.0}
        )

        semantics = create_rigidized_articulation_antipodal_semantics(
            cube,
            grasp_link="lower_two_layers",
            locked_qpos={"top_turn": 0.0},
            label="rubiks_cube",
        )

        assert semantics.entity_id == cube.uid
        assert semantics.label == "rubiks_cube"

    def test_articulation_satisfies_the_scene_entity_protocol(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64), qpos={"top_turn": 0.0}
        )

        assert isinstance(cube, SceneEntity)


class TestRejectedConfigurations:
    """A configuration that is not a locked rigid body must be refused."""

    def test_unknown_link_is_rejected(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64), qpos={"top_turn": 0.0}
        )

        with pytest.raises(ValueError, match="not part of"):
            create_rigidized_articulation_antipodal_semantics(
                cube,
                grasp_link="middle_layer",
                locked_qpos={"top_turn": 0.0},
                label="cube",
            )

    def test_undeclared_joint_is_rejected(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64),
            qpos={"top_turn": 0.0, "side_turn": 0.0},
        )

        with pytest.raises(ValueError, match="must declare every joint"):
            create_rigidized_articulation_antipodal_semantics(
                cube,
                grasp_link="top_layer",
                locked_qpos={"top_turn": 0.0},
                label="cube",
            )

    def test_joint_away_from_its_declared_lock_is_rejected(self) -> None:
        # The joint moved, so the mesh transform would describe a stale pose.
        cube = _StubArticulation(
            root_to_link=_transform(_rotation_z(0.5), torch.zeros(3)),
            qpos={"top_turn": 0.5},
        )

        with pytest.raises(ValueError, match="declared locked at"):
            create_rigidized_articulation_antipodal_semantics(
                cube,
                grasp_link="top_layer",
                locked_qpos={"top_turn": 0.0},
                label="cube",
            )

    def test_unknown_declared_joint_is_rejected(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64), qpos={"top_turn": 0.0}
        )

        with pytest.raises(ValueError, match="does not have"):
            create_rigidized_articulation_antipodal_semantics(
                cube,
                grasp_link="top_layer",
                locked_qpos={"top_turn": 0.0, "ghost_joint": 0.0},
                label="cube",
            )

    def test_empty_link_mesh_is_rejected(self) -> None:
        cube = _StubArticulation(
            root_to_link=torch.eye(4, dtype=torch.float64),
            qpos={"top_turn": 0.0},
            vertices=torch.zeros((0, 3)),
            triangles=torch.zeros((0, 3), dtype=torch.int64),
        )

        with pytest.raises(ValueError, match="no usable mesh"):
            create_rigidized_articulation_antipodal_semantics(
                cube,
                grasp_link="top_layer",
                locked_qpos={"top_turn": 0.0},
                label="cube",
            )
