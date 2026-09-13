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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim.objects.backends import articulation_lifecycle
from embodichain.lab.sim.objects.backends.articulation_geometry import (
    get_link_vert_face,
)
from embodichain.lab.sim.objects.backends.articulation_state import (
    map_source_qpos_to_state_order,
    read_state_mimic_info,
)
from embodichain.lab.sim.objects.backends.articulation_physics import (
    apply_link_com_pose,
    apply_link_inertia,
    apply_link_mass,
)
from embodichain.lab.sim.objects.backends.rigid_physics import (
    get_legacy_damping,
    get_legacy_friction,
    get_legacy_inertia,
    get_legacy_mass,
    get_newton_physical_attr,
    mirror_newton_physical_attr,
    set_legacy_damping,
    set_legacy_friction,
    set_legacy_inertia,
    set_legacy_mass,
)
from embodichain.lab.sim.objects.backends.newton import (
    _stabilize_newton_mimic_target_write,
)
from embodichain.lab.sim.objects.backends.lifecycle import (
    apply_rigid_initial_state,
    destroy_articulation_entities,
    destroy_rigid_entities,
    finalize_articulation_spawn,
)
from embodichain.lab.sim.objects.backends.controls import (
    get_body_scale,
    set_articulation_flag,
    set_body_scale,
    set_collision_enabled,
    set_gravity_enabled,
    set_visible,
)


class _RigidEntity:
    def __init__(self, handle: int = 7) -> None:
        self.handle = handle

    def get_native_handle(self) -> int:
        return self.handle


def test_newton_physical_attr_uses_entity_handle_and_mirrors_values() -> None:
    attr = SimpleNamespace(
        mass=0.0,
        density=0.0,
        dynamic_friction=0.0,
        static_friction=0.0,
        restitution=0.0,
        contact_offset=0.0,
        rest_offset=0.0,
        linear_damping=0.0,
        angular_damping=0.0,
        sleep_threshold=0.0,
        enable_ccd=False,
        max_depenetration_velocity=0.0,
        min_position_iters=0,
        min_velocity_iters=0,
        max_linear_velocity=0.0,
        max_angular_velocity=0.0,
    )
    scene = SimpleNamespace(manager=SimpleNamespace(dexsim_meta={7: {"attr": attr}}))
    entity = _RigidEntity()
    physical_attr = SimpleNamespace(
        **{
            name: index
            for index, name in enumerate(
                (
                    "mass",
                    "density",
                    "dynamic_friction",
                    "static_friction",
                    "restitution",
                    "contact_offset",
                    "rest_offset",
                    "linear_damping",
                    "angular_damping",
                    "sleep_threshold",
                    "enable_ccd",
                    "max_depenetration_velocity",
                    "min_position_iters",
                    "min_velocity_iters",
                    "max_linear_velocity",
                    "max_angular_velocity",
                )
            )
        }
    )

    mirror_newton_physical_attr(scene, entity, 0, "fixture", physical_attr)

    assert get_newton_physical_attr(scene, entity, 0, "fixture") is attr
    assert attr.mass == 0
    assert attr.max_angular_velocity == 15


def test_legacy_rigid_physics_helpers_use_one_native_body() -> None:
    class Body:
        def __init__(self) -> None:
            self.mass = 1.0
            self.dynamic_friction = 0.2
            self.linear_damping = 0.3
            self.angular_damping = 0.4
            self.inertia = np.zeros(3, dtype=np.float32)
            self.calls = 0

        def set_mass(self, value):
            self.mass = value

        def get_mass(self):
            return self.mass

        def set_dynamic_friction(self, value):
            self.dynamic_friction = value

        def set_static_friction(self, value):
            pass

        def get_dynamic_friction(self):
            return self.dynamic_friction

        def set_linear_damping(self, value):
            self.linear_damping = value

        def set_angular_damping(self, value):
            self.angular_damping = value

        def get_linear_damping(self):
            return self.linear_damping

        def get_angular_damping(self):
            return self.angular_damping

        def set_mass_space_inertia_tensor(self, value):
            self.inertia = value

        def get_mass_space_inertia_tensor(self):
            return self.inertia

    body = Body()
    entity = SimpleNamespace(get_physical_body=lambda: body)
    set_legacy_mass(entity, 2.0)
    set_legacy_friction(entity, 0.5)
    set_legacy_damping(entity, 0.6, 0.7)
    set_legacy_inertia(entity, np.ones(3, dtype=np.float32))

    assert get_legacy_mass(entity) == 2.0
    assert get_legacy_friction(entity) == 0.5
    assert get_legacy_damping(entity) == (0.6, 0.7)
    assert get_legacy_inertia(entity).tolist() == [1.0, 1.0, 1.0]


def test_articulation_link_writes_dispatch_to_backend_specific_entity_api() -> None:
    calls: list[tuple[str, object]] = []

    class Entity:
        def set_link_mass(self, name, value):
            calls.append(("spawn_mass", (name, value)))

        def set_link_inertia(self, name, value):
            calls.append(("spawn_inertia", (name, value)))
            return 0

        def set_link_com_pose(self, name, position, quaternion):
            calls.append(("spawn_com", (name, position, quaternion)))
            return 0

    entity = Entity()
    value = np.ones(3, dtype=np.float32)
    assert (
        apply_link_mass(entity, "link", 2.0, is_spawn_bound=True, is_newton=True)
        is None
    )
    assert (
        apply_link_inertia(entity, "link", value, is_spawn_bound=True, is_newton=True)
        == 0
    )
    assert (
        apply_link_com_pose(
            entity,
            "link",
            value,
            np.array([0, 0, 0, 1], dtype=np.float32),
            is_spawn_bound=True,
            is_newton=True,
        )
        == 0
    )
    assert [name for name, _ in calls] == [
        "spawn_mass",
        "spawn_inertia",
        "spawn_com",
    ]


def test_newton_link_geometry_combines_render_meshes() -> None:
    class RenderBody:
        def get_mesh_count(self):
            return 2

        def get_vertices(self, mesh_id):
            return np.full((3, 3), mesh_id, dtype=np.float32)

        def get_triangles(self, mesh_id):
            return np.array([[0, 1, 2]], dtype=np.int32)

    entity = SimpleNamespace(get_render_body=lambda _: RenderBody())
    vertices, faces = get_link_vert_face(entity, "link", is_newton=True)

    assert vertices.shape == (6, 3)
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_newton_mimic_target_write_updates_selected_follower() -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
    view = SimpleNamespace(
        apply_qpos=lambda values, env_ids, joint_ids, *, target: calls.append(
            (values, env_ids, joint_ids, target)
        )
    )
    limits = torch.tensor([[[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]])

    _stabilize_newton_mimic_target_write(
        configured=True,
        values=torch.tensor([[0.5]]),
        env_ids=torch.tensor([0]),
        joint_ids=torch.tensor([1]),
        velocity=False,
        mimic_ids=[2],
        mimic_parents=[1],
        mimic_multipliers=[2.0],
        mimic_offsets=[0.1],
        qpos_limits=limits,
        qvel_limits=torch.full((1, 3), 10.0),
        articulation_view=view,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    values, env_ids, joint_ids, target = calls[0]
    assert torch.allclose(values, torch.tensor([[1.1]]))
    assert env_ids.tolist() == [0]
    assert joint_ids.tolist() == [2]
    assert target is True


def test_default_spawn_runtime_config_applies_once_per_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        articulation_lifecycle,
        "_apply_default_articulation_root_properties",
        lambda binding, props: calls.append((binding, props)),
    )
    binding = object()
    entity = SimpleNamespace(_physics_binding=binding)
    result = SimpleNamespace(backend="dexsim", topology_revision=4)
    root_props = SimpleNamespace(
        sleep_threshold=0.1,
        min_position_iters=None,
        min_velocity_iters=None,
    )

    revision = articulation_lifecycle.prepare_default_spawn_runtime_config(
        result,
        [entity],
        root_props,
        -1,
    )
    same_revision = articulation_lifecycle.prepare_default_spawn_runtime_config(
        result,
        [entity],
        root_props,
        revision,
    )

    assert revision == 4
    assert same_revision == 4
    assert calls == [(binding, root_props)]


def test_lifecycle_destroy_helpers_select_backend_removal_api() -> None:
    rigid_calls: list[tuple[str, object]] = []
    articulation_calls: list[tuple[str, object]] = []

    class Arena:
        def remove_actor(self, value):
            rigid_calls.append(("actor", value))

        def remove_skeleton(self, value):
            articulation_calls.append(("skeleton", value))

        def remove_articulation(self, value):
            articulation_calls.append(("articulation", value))

    env = SimpleNamespace(get_all_arenas=lambda: [Arena(), Arena()])
    world = SimpleNamespace(get_env=lambda: env)
    rigid_entities = [
        SimpleNamespace(get_name=lambda: "newton_actor"),
        SimpleNamespace(get_name=lambda: "newton_actor_2"),
    ]
    articulation_entities = [object(), object()]

    destroy_rigid_entities(world, object(), rigid_entities, is_newton=True)
    destroy_articulation_entities(
        world, object(), articulation_entities, is_newton=False
    )

    assert rigid_calls == [("actor", "newton_actor"), ("actor", "newton_actor_2")]
    assert articulation_calls == [
        ("articulation", articulation_entities[0]),
        ("articulation", articulation_entities[1]),
    ]


def test_articulation_state_helpers_map_spawn_order_and_mimic_ids() -> None:
    layout = [
        SimpleNamespace(name="j2", dof_start=0),
        SimpleNamespace(name="j0", dof_start=1),
        SimpleNamespace(name="j1", dof_start=2),
    ]
    mimic = SimpleNamespace(
        mimic_id=np.array([0], dtype=np.int32),
        mimic_parent=np.array([2], dtype=np.int32),
        mimic_multiplier=np.array([1.5], dtype=np.float32),
        mimic_offset=np.array([0.2], dtype=np.float32),
    )
    entity = SimpleNamespace(
        joint_dof_layout=layout,
        get_actived_joint_names=lambda: ["j0", "j1", "j2"],
        get_mimic_info=lambda: mimic,
    )

    mapped = map_source_qpos_to_state_order(
        entity,
        torch.tensor([[10.0, 20.0, 30.0]]),
        is_spawn_bound=True,
    )
    mimic_ids, parent_ids, multipliers, offsets = read_state_mimic_info(entity)

    assert mapped.tolist() == [[30.0, 10.0, 20.0]]
    assert mimic_ids.tolist() == [1]
    assert parent_ids.tolist() == [0]
    assert multipliers.tolist() == pytest.approx([1.5])
    assert offsets.tolist() == pytest.approx([0.2])


def test_control_helpers_forward_native_entity_operations() -> None:
    calls: list[tuple[str, object]] = []

    class Entity:
        def get_body_scale(self):
            return (1.0, 2.0, 3.0)

        def set_body_scale(self, x, y, z):
            calls.append(("scale", (x, y, z)))

        def enable_collision(self, value):
            calls.append(("collision", value))

        def set_visible(self, value):
            calls.append(("visible", value))

        def set_articulation_flag(self, flag, value):
            calls.append(("flag", (flag, value)))

        def enable_gravity(self, value):
            calls.append(("gravity", value))

    entity = Entity()
    set_body_scale(entity, [2.0, 3.0, 4.0])
    set_collision_enabled(entity, 1)
    set_visible(entity, 0)
    set_articulation_flag(entity, "FIX_BASE", 1)
    set_gravity_enabled(entity, 0)

    assert get_body_scale(entity) == (1.0, 2.0, 3.0)
    assert calls == [
        ("scale", (2.0, 3.0, 4.0)),
        ("collision", True),
        ("visible", False),
        ("flag", ("FIX_BASE", True)),
        ("gravity", False),
    ]


def test_lifecycle_initial_state_helpers_preserve_backend_reset_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "embodichain.lab.sim.objects.backends.lifecycle.is_newton_gradient_mode",
        lambda _: False,
    )
    calls: list[object] = []
    articulation = SimpleNamespace(
        _data=SimpleNamespace(is_newton_backend=True),
        reset=lambda **kwargs: calls.append(kwargs),
    )
    result = SimpleNamespace(backend="newton")

    finalize_articulation_spawn(articulation, result)

    assert calls == [{"clear_dynamics": False}]

    rigid = SimpleNamespace(
        is_spawn_bound=True,
        _spawn_result=SimpleNamespace(backend="newton"),
        clear_dynamics=lambda: calls.append("clear"),
    )
    apply_rigid_initial_state(rigid)

    assert calls[-1] == "clear"
