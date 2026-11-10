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

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from embodichain.lab.sim.motion.planners.base_planner import BasePlanner
from embodichain.lab.sim.motion.planners.curobo import curobo_planner as cp
from embodichain.lab.sim.motion.planners.curobo.payload import (
    CuroboPayload,
    CuroboQuerySnapshot,
    curobo_motion_generator,
)
from embodichain.lab.sim.motion.planners.curobo.curobo_yaml import (
    generate_curobo_world_scene,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import PlanState, MoveType
from embodichain.lab.sim.atomic_skills.trajectory_ops import to_full_robot_trajectory
from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape
from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources
from embodichain.utils.math import matrix_from_quat, quat_wxyz_to_xyzw

_SOURCE = json.loads(
    (Path(__file__).parent / "data/curobo_query_source_reference.json").read_text()
)
_CONFIG = {
    "robot_cfg": {
        "kinematics": {
            "tool_frames": ["tool"],
            "collision_link_names": ["gripper"],
            "lock_joints": {"passive": 0.0},
        }
    }
}


def _snapshot(*, payload=True, transform=None):
    return CuroboQuerySnapshot(
        full_qpos=torch.tensor([[0.1, 0.9]]),
        world_to_base=torch.eye(4)[None] if transform is None else transform,
        payload=(
            CuroboPayload(
                object_id="carried",
                control_part="arm",
                spheres=torch.tensor([[0.0, 0.0, 0.0, 0.1]]).repeat(27, 1),
                object_pose=torch.eye(4)[None],
            )
            if payload
            else None
        ),
    )


def _cpu_constructor(monkeypatch):
    robot = SimpleNamespace(
        uid="trace_robot",
        device=torch.device("cpu"),
        joint_names=["arm_joint", "passive"],
        num_instances=1,
        cfg=SimpleNamespace(fpath="unused.urdf"),
        control_parts={"arm": ["arm_joint"]},
    )

    def no_pose(*args, **kwargs):
        raise AssertionError("query reread a live robot pose")

    robot.get_link_pose = no_pose
    monkeypatch.setattr(
        BasePlanner, "__init__", lambda self, cfg: setattr(self, "robot", robot)
    )
    monkeypatch.setattr(cp, "_resolve_curobo_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    bindings = SimpleNamespace(
        Pose=lambda **kwargs: SimpleNamespace(**kwargs),
        DeviceCfg=lambda **kwargs: SimpleNamespace(**kwargs),
        JointState=SimpleNamespace(
            from_position=lambda position, **kwargs: SimpleNamespace(
                position=position, **kwargs
            )
        ),
    )
    monkeypatch.setattr(cp, "_require_curobo", lambda level: bindings)
    return robot, bindings


def _cfg(world=None):
    return cp.CuroboPlannerCfg(
        robot_uid="trace_robot",
        use_cuda_graph=False,
        preserve_plan_samples=True,
        warmup_iterations=0,
        world=cp.CuroboWorldCfg(rigid_objects=world, max_voxel_count=4_000_000),
        interpolation_dt=0.04,
    )


def _backend(native=None):
    if native is None:
        native = SimpleNamespace(
            joint_names=["arm_joint"], scene_collision_checker=object()
        )
    profile = cp._CuroboProfile(
        "unused.yml",
        {"arm_joint": "arm_joint"},
        tool_frame_name="tool",
        sim_base_link_name="base",
    )
    return cp._CuroboBackend(
        planner=native,
        control_part="arm",
        sim_joint_names=["arm_joint"],
        tool_frame="tool",
        profile=profile,
        batch_size=1,
        use_cuda_graph=False,
        planning_mode=MoveType.EEF_MOVE,
    )


def _native_models():
    def model(name, buffer):
        return SimpleNamespace(
            name=name,
            config=SimpleNamespace(
                kinematics_config=SimpleNamespace(link_spheres=buffer)
            ),
        )

    buffers = [torch.zeros(32, 4) for _ in range(3)]
    kin, opt, metrics = [
        model(name, buffer)
        for name, buffer in zip(("kinematics", "optimizer", "metrics"), buffers)
    ]
    alias = model("optimizer_alias", buffers[1])

    def rollout(a, b):
        return SimpleNamespace(
            transition_model=SimpleNamespace(robot_model=a),
            metrics_transition_model=SimpleNamespace(robot_model=b),
        )

    native = SimpleNamespace(
        joint_names=["arm_joint"],
        kinematics=kin,
        ik_solver=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout(opt, metrics)]
        ),
        trajopt_solver=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout(alias, kin)],
            additional_metrics_rollouts={"extra": rollout(metrics, kin)},
        ),
        graph_planner=None,
    )
    return native


def _managers(monkeypatch, *, fail=None):
    events = []

    class Manager:
        def __init__(self, model, **kwargs):
            self.model = model
            events.append(("create", model.name))
            if fail == "create" and model.name == "optimizer_alias":
                raise RuntimeError("create failed")

        def update(self, spheres, state, **kwargs):
            events.append(
                (
                    "update",
                    self.model.name,
                    spheres.clone(),
                    state.position.clone(),
                    kwargs["world_objects_pose_offset"],
                )
            )
            if fail == "update" and self.model.name == "optimizer_alias":
                raise RuntimeError("update failed")

        def detach(self, **kwargs):
            events.append(("detach", self.model.name))
            if fail == "detach" and self.model.name == "kinematics":
                raise RuntimeError("detach failed")

    monkeypatch.setitem(
        sys.modules,
        "curobo._src.collision.attachment_manager",
        SimpleNamespace(AttachmentManager=Manager),
    )
    return events


def test_snapshot_world_goal_and_payload_use_one_nonidentity_frame(monkeypatch):
    _cpu_constructor(monkeypatch)
    events = _managers(monkeypatch)
    world_to_base = torch.tensor(
        [[[0.0, 1, 0, -2], [-1.0, 0, 0, 1], [0, 0, 1.0, -3], [0, 0, 0, 1.0]]]
    )
    object_pose = torch.eye(4)[None]
    object_pose[0, :3, 3] = torch.tensor([4.0, 5.0, 6.0])
    shape = CollisionShapeDesc(
        "box",
        RigidBodyShape.BOX,
        torch.eye(4),
        half_extents=torch.tensor([0.1, 0.2, 0.3]),
    )
    world = snapshot_collision_sources(
        {"obstacle": (shape,)}, {"obstacle": object_pose}, world_to_base
    )
    snapshot = _snapshot(transform=world_to_base)
    snapshot = replace(
        snapshot, payload=replace(snapshot.payload, object_pose=object_pose)
    )
    planner = cp.CuroboPlanner(_cfg(world), query_snapshot=snapshot)
    backend = _backend(_native_models())
    # A nonidentity configured sim-base convention must not cause a second transform.
    backend.profile.sim_base_to_curobo_base = [
        [1.0, 0, 0, 0.2],
        [0, 0, -1.0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 0, 1.0],
    ]
    expected = world_to_base @ object_pose
    torch.testing.assert_close(
        planner._sim_world_to_curobo_base_pose(object_pose, backend), expected
    )
    planner._get_sim_base_pose(backend, 1)  # Must not call the forbidden live getter.
    scene = generate_curobo_world_scene(world)
    torch.testing.assert_close(
        torch.tensor(scene["cuboid"]["obstacle"]["pose"][:3]), expected[0, :3, 3]
    )
    planner._attach_payload(backend)
    updates = [e for e in events if e[0] == "update"]
    assert [e[1] for e in updates] == ["kinematics", "optimizer_alias", "metrics"]
    for update in updates:
        torch.testing.assert_close(update[3], torch.tensor([[0.1]]))
        pose = update[4]
        torch.testing.assert_close(pose.position, expected[:, :3, 3])
        torch.testing.assert_close(
            matrix_from_quat(quat_wxyz_to_xyzw(pose.quaternion)), expected[:, :3, :3]
        )
    original = world["obstacle"].pose.clone()
    object_pose.fill_(99)
    world_to_base.fill_(99)
    shape.half_extents.fill_(99)
    torch.testing.assert_close(world["obstacle"].pose, original)
    torch.testing.assert_close(
        world["obstacle"].shapes[0].half_extents, torch.tensor([0.1, 0.2, 0.3])
    )
    planner.close()


@pytest.mark.parametrize("failure", ["create", "update", "detach"])
def test_owned_attachments_cleanup_partial_failures_and_deduplicate_aliases(
    monkeypatch, failure
):
    _cpu_constructor(monkeypatch)
    events = _managers(monkeypatch, fail=failure)
    planner = cp.CuroboPlanner(_cfg(), query_snapshot=_snapshot())
    backend = _backend(_native_models())
    closed = []
    backend.planner.close = lambda: closed.append("closed")
    if failure in {"create", "update"}:
        monkeypatch.setattr(planner, "_robot_yaml_cache_key", lambda *args: "limits")
        monkeypatch.setattr(
            planner, "_materialize_profile", lambda part: backend.profile
        )
        monkeypatch.setattr(planner, "_build_backend", lambda **kwargs: backend)
        monkeypatch.setattr(planner, "_warmup_backend", lambda value: None)
        with pytest.raises(RuntimeError, match=failure + " failed"):
            planner._get_backend("arm", 1)
        assert not planner._payload_managers
        assert not planner._backend_cache and closed == ["closed"]
        expected = (
            ["kinematics"]
            if failure == "create"
            else ["kinematics", "optimizer_alias", "metrics"]
        )
        assert [e[1] for e in events if e[0] == "detach"] == expected
    else:
        planner._attach_payload(backend)
        planner._backend_cache[("arm", 1, False, MoveType.EEF_MOVE)] = backend
        with pytest.raises(RuntimeError, match="detach failed"):
            planner.close()
        assert [e[1] for e in events if e[0] == "detach"] == [
            "kinematics",
            "optimizer_alias",
            "metrics",
        ]
        assert closed == ["closed"]
        assert not planner._backend_cache
    before = len(events)
    planner.close()
    assert len(events) == before


def test_no_payload_validation_uses_observed_passive_locks(monkeypatch, tmp_path):
    _, bindings = _cpu_constructor(monkeypatch)
    planner = cp.CuroboPlanner(_cfg(), query_snapshot=_snapshot(payload=False))
    backend = _backend()
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(_CONFIG))
    backend.profile.robot_config_path = str(path)
    planner._backend_cache[("arm", 1, False, MoveType.EEF_MOVE)] = backend
    captured = []

    def load(**kwargs):
        captured.append(kwargs["robot_config"])
        return SimpleNamespace(self_collision_cost=object())

    bindings.RobotCollisionCheckerCfg = SimpleNamespace(load_from_config=load)
    bindings.RobotCollisionChecker = lambda cfg: SimpleNamespace(
        get_kinematics=lambda sample: SimpleNamespace(
            robot_spheres=torch.zeros(1, 1, 2, 4)
        ),
        get_bound=lambda sample: torch.zeros(1, 1, 1),
        collision_constraint=None,
    )
    result = planner.validate_joint_trajectory(
        torch.tensor([[[0.1], [0.2]]]), control_part="arm"
    )
    assert result.tolist() == [[True, True]]
    assert captured[0]["robot_cfg"]["kinematics"]["lock_joints"][
        "passive"
    ] == pytest.approx(0.9)
    assert planner.supports_joint_trajectory_validation
    planner.close()


def test_query_capabilities_reject_unsupported_lifetimes_before_native_use(monkeypatch):
    _cpu_constructor(monkeypatch)
    snapshot = _snapshot()
    for cfg in (
        replace(_cfg(), use_cuda_graph=True),
        replace(_cfg(), preserve_plan_samples=False),
    ):
        with pytest.raises(ValueError, match="Query snapshots"):
            cp.CuroboPlanner(cfg, query_snapshot=snapshot)
    planner = cp.CuroboPlanner(_cfg(), query_snapshot=snapshot)
    assert not planner.supports_joint_trajectory_validation
    assert not planner.supports_collision_world_updates
    with pytest.raises(NotImplementedError, match="Payload"):
        planner.validate_joint_trajectory(torch.zeros(1, 2, 1), control_part="arm")
    with pytest.raises(NotImplementedError, match="immutable"):
        planner.update_dynamic_obstacles({"x": torch.eye(4)[None]})
    with pytest.raises(ValueError, match="batch"):
        planner._validate_query_backend("arm", 2)
    planner.close()
    with pytest.raises(RuntimeError, match="closed"):
        planner._validate_query_backend("arm", 1)


@pytest.mark.parametrize(
    "trace", _SOURCE["traces"], ids=lambda t: t["route"] + str(t["incoming"])
)
def test_owned_factory_preserves_source_options_grid_and_result_lifetime(
    monkeypatch, trace
):
    _cpu_constructor(monkeypatch)
    snapshot = _snapshot(payload=trace["route"] == "drawer_transport")
    cfg = _cfg()
    records = []
    native_buffers = []
    monkeypatch.setattr(
        cp.CuroboPlanner,
        "_load_runtime_robot_config",
        staticmethod(lambda path: deepcopy(_CONFIG)),
    )

    def plan(self, target_states, options):
        record = {
            key: getattr(options, key)
            for key in ("max_velocity", "max_acceleration", "control_part")
        }
        record["start_qpos"] = options.start_qpos.tolist()
        records.append(record)
        samples = torch.tensor([[0.1], [0.2], [0.3], [0.3]])
        native_buffers.append(samples)
        backend = _backend()
        backend.planner.close = lambda: samples.fill_(99)
        self._backend_cache[("arm", 1, False, MoveType.EEF_MOVE)] = backend
        assert (
            self._query_runtime_robot_config("ignored") == trace["runtime_robot_config"]
        )
        return self._assemble_result(
            [[samples]],
            [[torch.tensor([0.0, 0.04, 0.04, 0.04])]],
            torch.tensor([[0.1]]),
            torch.tensor([True]),
            1,
            1,
        )

    monkeypatch.setattr(cp.CuroboPlanner, "plan", plan)
    with curobo_motion_generator(cfg, query_snapshot=snapshot) as generator:
        for name, expected_cfg in trace["cfg"].items():
            actual = (
                getattr(cfg.world, name[6:])
                if name.startswith("world_")
                else getattr(cfg, name)
            )
            assert actual == expected_cfg
        options = MotionGenOptions(
            **trace["motion_options"],
            start_qpos=torch.tensor([[0.1]]),
            control_part="arm",
        )
        result = generator.generate(
            [PlanState.from_qpos(torch.tensor([[0.3]]))], options
        )
        expected = result.positions.clone()
        _, compiled = to_full_robot_trajectory(
            result,
            base_qpos=snapshot.full_qpos,
            joint_ids=[0],
            env_ids=torch.tensor([0]),
            control_dt=0.04,
        )
    assert records == [trace["plan_options"]]
    assert result.positions.shape[1] == trace["output_count"]
    torch.testing.assert_close(result.positions, expected)
    assert torch.all(native_buffers[0] == 99)
    torch.testing.assert_close(
        compiled.positions,
        torch.tensor(trace["control_grid"]["positions"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        compiled.dt, torch.tensor(trace["control_grid"]["dt"]), rtol=0, atol=0
    )


def test_owned_factory_detaches_and_closes_when_the_caller_raises(monkeypatch):
    _cpu_constructor(monkeypatch)
    events = _managers(monkeypatch)
    closed = []
    with pytest.raises(RuntimeError, match="phase failed"):
        with curobo_motion_generator(_cfg(), query_snapshot=_snapshot()) as generator:
            backend = _backend(_native_models())
            backend.planner.close = lambda: closed.append("closed")
            generator.planner._attach_payload(backend)
            generator.planner._backend_cache[("arm", 1, False, MoveType.EEF_MOVE)] = (
                backend
            )
            raise RuntimeError("phase failed")
    assert [event[1] for event in events if event[0] == "detach"] == [
        "kinematics",
        "optimizer_alias",
        "metrics",
    ]
    assert closed == ["closed"]


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("contract", ["start", "world_capability"])
def test_public_query_facade_uses_coherent_default_start_and_world_capability(
    monkeypatch, observed, contract
):
    robot, _ = _cpu_constructor(monkeypatch)
    live_reads = []

    def live_qpos(*, name):
        live_reads.append(name)
        return torch.tensor([[0.7]])

    robot.get_qpos = live_qpos
    starts = []
    monkeypatch.setattr(cp.CuroboPlanner, "_get_backend", lambda *args: _backend())

    def segments(self, targets, start, backends, options, pose):
        starts.append(start.clone())
        samples = torch.stack((start[0], targets[0].qpos[0]))
        return self._assemble_result(
            [[samples]],
            [[torch.tensor([0.0, 0.04])]],
            start,
            torch.tensor([True]),
            1,
            1,
        )

    monkeypatch.setattr(cp.CuroboPlanner, "_plan_segments", segments)
    snapshot = _snapshot(payload=False) if observed else None
    with curobo_motion_generator(_cfg(), query_snapshot=snapshot) as generator:
        if contract == "world_capability":
            assert generator.supports_dynamic_collision_world is (not observed)
            return
        result = generator.generate(
            [PlanState.from_qpos(torch.tensor([[0.9]]))],
            MotionGenOptions(control_part="arm"),
        )
        torch.testing.assert_close(
            result.positions[:, 0], torch.tensor([[0.1 if observed else 0.7]])
        )
    torch.testing.assert_close(starts[0], torch.tensor([[0.1 if observed else 0.7]]))
    assert live_reads == ([] if observed else ["arm"])


def test_observed_batch_snapshot_frames_locks_and_rejections(monkeypatch):
    """Batch observations own their rows; model locks cannot silently use row zero."""
    from embodichain.lab.sim.motion.planners.curobo.payload import _query_robot_config

    robot, _ = _cpu_constructor(monkeypatch)
    robot.num_instances = 2
    q = torch.tensor([[0.1, 0.9], [0.2, 0.9]])
    frames = torch.eye(4).repeat(2, 1, 1)
    frames[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    frames[1, 0, 3] = 3
    snapshot = CuroboQuerySnapshot(full_qpos=q, world_to_base=frames)
    planner = cp.CuroboPlanner(_cfg(), query_snapshot=snapshot)
    try:
        planner._validate_query_backend("arm", 2)
        goals = torch.eye(4).repeat(2, 1, 1)
        goals[:, 0, 3] = 1
        projected = planner._sim_world_to_curobo_base_pose(goals, _backend())
        torch.testing.assert_close(projected, frames @ goals)
        assert _query_robot_config(_CONFIG, snapshot, robot.joint_names)["robot_cfg"][
            "kinematics"
        ]["lock_joints"] == {"passive": pytest.approx(0.9)}
        q[1, 1] = 0.8
        frames[1, 0, 3] = 9
        assert snapshot.full_qpos[1, 1] == pytest.approx(0.9)
        assert snapshot.world_to_base[1, 0, 3] == 3
        with pytest.raises(ValueError, match="passive joint locks"):
            _query_robot_config(
                _CONFIG, replace(snapshot, full_qpos=q), robot.joint_names
            )
        with pytest.raises(ValueError, match="row order"):
            replace(snapshot, env_ids=(1, 0))
        with pytest.raises(ValueError, match="payload"):
            replace(snapshot, payload=_snapshot().payload)
        with pytest.raises(ValueError, match="batch"):
            replace(snapshot, world_to_base=frames[:1])
        invalid = frames.clone()
        invalid[1, 0, 0] = 2
        with pytest.raises(ValueError, match="rigid"):
            replace(snapshot, world_to_base=invalid)
        robot.num_instances = 4
        with pytest.raises(ValueError, match="num_instances"):
            planner._validate_query_backend("arm", 2)
    finally:
        planner.close()


def test_shared_world_snapshot_checks_every_projected_row():
    """Environment offsets cancel in a shared world; a different layout must fail."""
    frames = torch.eye(4).repeat(2, 1, 1)
    frames[1, 0, 3] = -4
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 4
    world = snapshot_collision_sources({"box": ()}, {"box": poses}, frames)
    torch.testing.assert_close(world["box"].pose, torch.eye(4)[None])
    poses[1, 1, 3] = 0.2
    with pytest.raises(ValueError, match="worlds must be identical"):
        snapshot_collision_sources({"box": ()}, {"box": poses}, frames)
