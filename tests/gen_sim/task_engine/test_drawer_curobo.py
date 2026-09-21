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

import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program.drawer_curobo import (
    LinkCollisionSource,
    payload_spheres,
    payload_robot_config,
    world_sources,
    attachment_models,
    snapshot_in_base,
)


def test_link_collision_keeps_panels_separate_and_tracks_live_pose() -> None:
    meshes = [trimesh.creation.box(extents=[0.02, 0.3, 0.1]) for _ in range(2)]
    meshes[0].apply_translation([-0.1, 0, 0])
    meshes[1].apply_translation([0.1, 0, 0])
    pose = torch.eye(4).unsqueeze(0)
    art = SimpleNamespace(get_link_pose=lambda *args, **kwargs: pose.clone())
    source = LinkCollisionSource("cabinet__drawer", art, "drawer", meshes)
    shapes = source.get_collision_shapes()
    assert len(shapes) == 2
    assert all(float(shape.half_extents[0]) == pytest.approx(0.01) for shape in shapes)
    assert shapes[0].local_pose[0, 3] < 0 < shapes[1].local_pose[0, 3]
    pose[0, 1, 3] = -0.12
    assert float(source.get_local_pose(to_matrix=True)[0, 1, 3]) == pytest.approx(-0.12)


def test_payload_spheres_cover_the_full_box_in_object_coordinates() -> None:
    vertices = torch.tensor(trimesh.creation.box(extents=[0.12, 0.09, 0.06]).vertices)
    spheres = payload_spheres(vertices)
    distance = torch.cdist(vertices.to(spheres), spheres[:, :3]) - spheres[:, 3]
    assert (distance.min(dim=1).values <= 1e-6).all()
    assert spheres.shape == (27, 4)


def test_snapshot_uses_live_robot_base_and_keeps_shape_offsets() -> None:
    base = torch.eye(4).unsqueeze(0)
    base[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    base[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    local = torch.eye(4).unsqueeze(0)
    local[0, :3, 3] = torch.tensor([0.4, -0.2, 0.1])
    world = base @ local
    shape = object()
    source = SimpleNamespace(
        get_local_pose=lambda **kw: world, get_collision_shapes=lambda: [shape]
    )
    snapshot = snapshot_in_base({"drawer": source}, base)["drawer"]
    torch.testing.assert_close(snapshot.get_local_pose(), local)
    assert snapshot.get_collision_shapes() == [shape]
    world[0, 0, 3] += 0.1
    torch.testing.assert_close(snapshot.get_local_pose(), local)


def test_attachment_model_preserves_gripper_and_uses_observed_locked_joints() -> None:
    config = {
        "robot_cfg": {
            "kinematics": {
                "tool_frames": ["hand"],
                "collision_link_names": ["hand"],
                "collision_spheres": {"hand": [{"center": [0, 0, 0], "radius": 0.02}]},
                "lock_joints": {"finger": 0.0},
            }
        }
    }
    updated = payload_robot_config(config, {"finger": 0.4})
    k = updated["robot_cfg"]["kinematics"]
    assert (
        k["collision_spheres"]["hand"]
        == config["robot_cfg"]["kinematics"]["collision_spheres"]["hand"]
    )
    assert k["extra_links"]["gen_sim_payload"]["parent_link_name"] == "hand"
    assert k["lock_joints"]["finger"] == pytest.approx(0.4)
    assert config["robot_cfg"]["kinematics"]["lock_joints"]["finger"] == 0


def test_world_excludes_only_carried_body_and_keeps_all_articulation_links(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import drawer_curobo as module

    mesh = trimesh.creation.box(extents=[0.1] * 3)
    monkeypatch.setattr(
        module,
        "collision_meshes",
        lambda cfg: {
            ("cabinet", "floor"): mesh,
            ("drawer_1", "wall"): mesh,
            ("drawer_2", "wall"): mesh,
        },
    )
    art = SimpleNamespace(cfg=SimpleNamespace(fpath="unused", body_scale=[1] * 3))
    sim = SimpleNamespace(
        get_rigid_object_uid_list=lambda: ["table", "cube", "cup"],
        get_rigid_object=lambda uid: uid,
        get_articulation_uid_list=lambda: ["cabinet"],
        get_articulation=lambda uid: art,
    )
    sources = world_sources(sim, "cube")
    assert set(sources) == {
        "table",
        "cup",
        "cabinet__cabinet",
        "cabinet__drawer_1",
        "cabinet__drawer_2",
    }


def test_native_attachment_detaches_even_when_cleanup_raises(monkeypatch) -> None:
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program.drawer_curobo import (
        _PayloadPlanner,
    )
    from embodichain.lab.sim.motion.planners.curobo.curobo_planner import CuroboPlanner

    manager = SimpleNamespace(detach=Mock(side_effect=RuntimeError("detach")))
    closed = Mock()
    monkeypatch.setattr(CuroboPlanner, "close", closed)
    planner = object.__new__(_PayloadPlanner)
    planner._attachments = {0: (manager,)}
    planner._backend_cache = {}
    with pytest.raises(RuntimeError, match="detach"):
        planner.close()
    closed.assert_called_once()
    planner._backend_cache = {}


def test_attachment_updates_every_distinct_rollout_sphere_buffer() -> None:
    def model():
        return SimpleNamespace(
            config=SimpleNamespace(
                kinematics_config=SimpleNamespace(link_spheres=torch.zeros(1, 30, 4))
            )
        )

    a, b, c = model(), model(), model()

    def rollout(first, second):
        return SimpleNamespace(
            transition_model=SimpleNamespace(robot_model=first),
            metrics_transition_model=SimpleNamespace(robot_model=second),
        )

    planner = SimpleNamespace(
        kinematics=a,
        ik_solver=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout(a, b)],
        ),
        trajopt_solver=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout(a, a)],
            additional_metrics_rollouts={"interpolated": rollout(b, c)},
        ),
        graph_planner=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout(c, a)]
        ),
    )
    assert {id(m) for m in attachment_models(planner)} == {id(a), id(b), id(c)}


def test_transport_failure_cleans_up_without_interpolation_fallback(
    monkeypatch,
) -> None:
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program import drawer_curobo as module

    fallback = Mock()
    monkeypatch.setattr(module.ApproachMotionGenerator, "generate", fallback)
    monkeypatch.setattr(module, "world_sources", lambda *args: {"drawer": object()})
    native = SimpleNamespace(
        planner=SimpleNamespace(close=Mock()),
        generate=Mock(side_effect=RuntimeError("native planning failed")),
    )
    monkeypatch.setattr(module, "_PayloadMotionGenerator", lambda *args, **kw: native)
    generator = object.__new__(module.DrawerMotionGenerator)
    generator._transport = None
    generator._placement_path = None
    generator._simulation = object()
    generator.robot = SimpleNamespace(uid="robot")
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(sim_instance_id="test"))
    observation = SimpleNamespace(route=SimpleNamespace(object_id="cube"))
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)), require_control_dt=lambda: 0.02
    )
    options = SimpleNamespace(strategy="ik_interp", sample_count=260, plan_opts=None)
    with pytest.raises(RuntimeError, match="native planning failed"):
        with generator.transport(observation, object(), context):
            generator.generate([], options)
    assert generator._transport is None
    native.planner.close.assert_called_once()
    fallback.assert_not_called()
    passed_options = native.generate.call_args.args[1]
    assert passed_options.strategy == "motion_gen"
    assert passed_options.sample_count is None
    assert options.strategy == "ik_interp"
    generator.generate([], options)
    fallback.assert_called_once()


def test_place_retreat_reverses_entry_and_clears_scope(monkeypatch) -> None:
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program import drawer_curobo as module

    delegate = Mock(return_value="planned")
    monkeypatch.setattr(module.ApproachMotionGenerator, "generate", delegate)
    generator = object.__new__(module.DrawerMotionGenerator)
    generator._transport = generator._placement_path = None
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    poses[0, :, :3, 3] = torch.tensor([[0.2, 0, 0.9], [0, 0, 0.9], [0, 0, 0.89]])
    with generator.placement_path(poses):
        assert generator.generate([], None) == "planned"
        path = torch.stack([p.xpos for p in delegate.call_args.args[0]], dim=1)
        torch.testing.assert_close(path, poses[:, [0, 1, 2, 1, 0]])
    assert generator._placement_path is None
    with pytest.raises(RuntimeError):
        with generator.placement_path(poses):
            raise RuntimeError("planning failed")
    assert generator._placement_path is None
    original = [object()]
    generator.generate(original, None)
    assert delegate.call_args.args[0] is original
