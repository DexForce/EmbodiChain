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

"""CPU checks against the official Newton asset importer and Spawn lowering."""

from __future__ import annotations

from unittest.mock import MagicMock
import json
from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp

from embodichain.lab.sim.physics.solvers._reference_assets import (
    _import_reference,
    make_reference_articulation,
    reference_asset_hash,
    reference_joint_names,
    reference_profile,
)


def _lower_descriptor(descriptor):
    # Exercise production Spawn translation without creating a renderer/World.
    from dexsim.spawn.adapters.newton_articulation_adapter import (
        _translate_explicit_articulation,
    )
    from dexsim.engine.newton_physics.articulation.articulation import (
        NewtonArticulation,
    )
    from dexsim.engine.newton_physics._runtime import create_model_builder

    builder = create_model_builder()
    specs, joints = _translate_explicit_articulation(
        MagicMock(), None, descriptor, {}, "mujoco_warp", reuse_render=True
    )
    handle = NewtonArticulation(
        builder,
        None,
        descriptor.name,
        descriptor.root_link_name,
        specs,
        joints,
        build_device="cpu",
    )
    handle.finalize(
        descriptor.pose, fixed_base=descriptor.fixed_base, disable_self_collision=True
    )
    return builder


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
@pytest.mark.parametrize("physics_profile", ["native", "warp_reference"])
def test_spawn_lowering_preserves_reference_physical_properties(robot, physics_profile):
    reference = _import_reference(robot, physics_profile)
    descriptor = make_reference_articulation(robot, physics_profile)
    lowered = _lower_descriptor(descriptor)
    lower_bodies = {name: index for index, name in enumerate(lowered.body_label)}
    for source_index, label in enumerate(reference.body_label):
        target_index = lower_bodies[label.rsplit("/", 1)[-1]]
        for field in ("body_mass", "body_com", "body_inertia"):
            np.testing.assert_allclose(
                np.asarray(getattr(lowered, field)[target_index]),
                np.asarray(getattr(reference, field)[source_index]),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"{robot} {label} {field}",
            )
    # Compare physical shapes; the source also imports non-colliding visuals.
    active_shapes = [
        i
        for i, flag in enumerate(reference.shape_flags)
        if int(flag)
        & int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
    ]
    # Shape order and exact primitive scales/materials survive public descriptors.
    for field in (
        "shape_type",
        "shape_scale",
        "shape_transform",
        "shape_margin",
        "shape_gap",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_kf",
        "shape_material_mu",
        "shape_material_mu_torsional",
        "shape_material_mu_rolling",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(lowered, field)),
            np.asarray(getattr(reference, field))[active_shapes],
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{robot} {field}",
        )
    lower_joints = {name: index for index, name in enumerate(lowered.joint_label)}
    for source_joint, label in enumerate(reference.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name not in reference_joint_names(robot):
            continue
        source_dof = reference.joint_qd_start[source_joint]
        target_dof = lowered.joint_qd_start[lower_joints[name]]
        for field in (
            "joint_armature",
            "joint_damping",
            "joint_friction",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_target_ke",
            "joint_target_kd",
        ):
            np.testing.assert_allclose(
                getattr(lowered, field)[target_dof],
                getattr(reference, field)[source_dof],
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"{robot} {name} {field}",
            )
        np.testing.assert_allclose(
            np.asarray(lowered.joint_axis[target_dof]),
            np.asarray(reference.joint_axis[source_dof]),
        )
    assert len(lowered.joint_q) == reference_profile(robot)["q_dim"]
    assert len(lowered.joint_qd) == reference_profile(robot)["qd_dim"]


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_cpu_fk_matches_reference_importer(robot):
    reference = _import_reference(robot)
    lowered = _lower_descriptor(make_reference_articulation(robot))
    q = reference_profile(robot)["default_q"]
    q[-1] += 0.17
    reference.joint_q[:] = q
    lowered.joint_q[:] = q
    models = [builder.finalize(device="cpu") for builder in (reference, lowered)]
    states = [model.state() for model in models]
    for model, state in zip(models, states):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    lower_indices = {name: index for index, name in enumerate(lowered.body_label)}
    for index, name in enumerate(reference.body_label):
        actual = states[1].body_q.numpy()[lower_indices[name.rsplit("/", 1)[-1]]]
        expected = states[0].body_q.numpy()[index]
        np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-5, atol=1e-6)
        assert abs(float(np.dot(actual[3:], expected[3:]))) == pytest.approx(
            1.0, abs=1e-6
        )


def test_reference_hash_and_profiles():
    assert (
        reference_asset_hash("ant")
        == "23e9a720465a71fde3fd9fd2d5a18f8c11ec642f3dca8921c66cb4e169421fa7"
    )
    assert (
        reference_asset_hash("cartpole")
        == "f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237"
    )
    first = make_reference_articulation("ant")
    second = make_reference_articulation("ant")
    first.links[0].rigid_body.mass = 99.0
    assert second.links[0].rigid_body.mass == pytest.approx(0.48388024694610887)
    assert reference_profile("ant")["effort_limits"] == [400.0] * 8
    with pytest.raises(ValueError, match="Unsupported"):
        make_reference_articulation("unknown")


def test_warp_cartpole_import_preserves_measured_mass_and_body_armature():
    """Check measured Warp 1.8 arrays, rather than URDF inertial declarations."""
    builder = _import_reference("cartpole", "warp_reference")
    np.testing.assert_allclose(builder.body_mass, [50.0, 2.5], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(builder.body_inertia).reshape(-1, 3, 3),
        [
            np.diag([1.21833336353302, 1.21833336353302, 2.0933332443237305]),
            np.diag([0.2188541740179062, 0.2188541740179062, 0.011041666381061077]),
        ],
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(builder.joint_armature, [0.01, 0.01])
    # The modern interpretation stays available as its own physical domain.
    np.testing.assert_allclose(_import_reference("cartpole").body_mass, [1.0, 10.0])


def test_warp_ant_import_preserves_geometry_density_and_force_mode():
    """Check measured Warp 1.5.1 masses, inertia regularization and friction."""
    builder = _import_reference("ant", "warp_reference")
    np.testing.assert_allclose(
        builder.body_mass,
        [31.653451919555664, *[7.831550598144531, 13.518441200256348] * 4],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(builder.body_inertia[0]).reshape(3, 3),
        np.diag([0.5960065126419067, 0.5960065126419067, 1.0890781879425049]),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(builder.joint_damping, 0.0)
    np.testing.assert_allclose(builder.joint_target_kd, 0.0)
    np.testing.assert_allclose(builder.shape_material_mu, 0.75)
    np.testing.assert_allclose(builder.shape_margin, 0.0)
    transform = builder.shape_transform[1]
    first_slot = wp.transform_point(
        transform, wp.vec3(0.0, 0.0, builder.shape_scale[1][1])
    )
    np.testing.assert_allclose(first_slot, [0.2, 0.2, 0.0], atol=5e-8)
    assert reference_asset_hash("ant") == (
        "23e9a720465a71fde3fd9fd2d5a18f8c11ec642f3dca8921c66cb4e169421fa7"
    )


def test_unknown_reference_physics_profile_rejected():
    with pytest.raises(ValueError, match="physics profile"):
        _import_reference("ant", "unregistered")


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_warp_profile_matches_recorded_body_arrays_and_ordered_contacts(robot):
    fixture = json.loads(
        (Path(__file__).parent / "data/warp_reference_arrays.json").read_text()
    )[robot]
    builder = _import_reference(robot, "warp_reference")
    for name in ("body_mass", "body_com", "body_inertia", "joint_armature"):
        actual = np.asarray(getattr(builder, name))
        expected = np.asarray(fixture[name]).reshape(actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7, err_msg=name)
    inertia = np.asarray(builder.body_inertia).reshape(-1, 3, 3)
    inverse = np.asarray(builder.body_inv_inertia).reshape(-1, 3, 3)
    np.testing.assert_allclose(
        inertia @ inverse, np.broadcast_to(np.eye(3), inertia.shape), atol=1e-6
    )
    if robot == "ant":
        points = []
        for scale, transform in zip(builder.shape_scale, builder.shape_transform):
            offsets = (
                [wp.vec3(0.0)]
                if scale[1] == 0
                else [wp.vec3(0.0, 0.0, scale[1]), wp.vec3(0.0, 0.0, -scale[1])]
            )
            points.extend(
                list(wp.transform_point(transform, value)) for value in offsets
            )
        np.testing.assert_allclose(points, fixture["ordered_contact_points"], atol=2e-7)
