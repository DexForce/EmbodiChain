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

"""Independent collision and multi-stage branch-connection oracles."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.lab.sim.scene.feasibility import (
    CollisionBody,
    NewtonCollisionChecker,
    FeasibilityBudget,
    SceneRobotFeasibilityCfg,
    TaskPhase,
    TaskOperationMode,
    PhaseTarget,
    plan_task_mode,
    UnsupportedFeasibilityError,
)


def test_newton_self_interarm_payload_and_phase_contacts():
    pytest.importorskip("newton")
    bodies = (
        CollisionBody("left", "left", "box", (0.2, 0.2, 0.2), robot=True),
        CollisionBody("right", "right", "box", (0.2, 0.2, 0.2), robot=True),
        CollisionBody("payload", "payload", "sphere", (0.1,)),
        CollisionBody("obstacle", "obstacle", "box", (0.2, 0.2, 0.2)),
    )
    checker = NewtonCollisionChecker(bodies)

    def pose(x):
        p = np.eye(4)
        p[0, 3] = x
        return p

    frames = {
        "left": pose(0),
        "right": pose(1),
        "payload": pose(2),
        "obstacle": pose(3),
    }
    assert checker.check(frames)[0].status == "passed"
    frames["right"] = pose(0.1)
    assert checker.check(frames)[0].status == "failed"
    assert (
        checker.check(frames, allowed_contacts=(("left", "right"),))[0].status
        == "passed"
    )
    frames["right"] = pose(1)
    frames["payload"] = pose(3)
    assert checker.check(frames, moving_objects=("payload",))[0].status == "failed"
    with pytest.raises(UnsupportedFeasibilityError, match="Missing"):
        checker.check({"left": pose(0)})


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_batched_geometry_matches_scalar_rows_and_rebinds_permissions(device):
    bodies = (
        CollisionBody("arm", "arm", "box", (0.2, 0.2, 0.2), robot=True),
        CollisionBody("payload", "payload", "sphere", (0.1,)),
        CollisionBody("obstacle", "obstacle", "box", (0.2, 0.2, 0.2)),
        CollisionBody("ground", "ground", "halfspace", (0, 0, 1, 0)),
    )
    frames = {
        name: np.repeat(np.eye(4)[None], 3, axis=0)
        for name in ("arm", "payload", "obstacle", "ground")
    }
    for name, x in (("arm", 0.0), ("payload", 1.0), ("obstacle", 2.0)):
        frames[name][:, 0, 3] = x
        frames[name][:, 2, 3] = 0.5
    frames["payload"][1, 0, 3] = 0.1
    frames["payload"][2, 2, 3] = -0.1
    checker = NewtonCollisionChecker(bodies, device=device)
    try:
        for allowed in ((), (("arm", "payload"),)):
            expected, pairs = [], []
            for row in range(3):
                report, _ = checker.check(
                    {key: value[row] for key, value in frames.items()},
                    allowed_contacts=allowed,
                    moving_objects=("payload",),
                )
                expected.append(report.status)
                pairs.append(checker.last_pair)
            result = checker.check_batch(
                frames, allowed_contacts=allowed, moving_objects=("payload",)
            )
            assert [report.status for report in result] == expected
            assert checker.last_batch_pairs == tuple(pairs)
            selected = checker.check_batch(
                frames,
                allowed_contacts=allowed,
                moving_objects=("payload",),
                env_ids=(2, 0),
            )
            assert [report.status for report in selected] == [expected[2], expected[0]]
        assert expected == ["passed", "passed", "failed"]
        # Changing permissions cannot reuse the previous mask, and invalid row
        # IDs or malformed physical transforms fail before querying geometry.
        assert (
            checker.check_batch(frames, moving_objects=("payload",))[1].status
            == "failed"
        )
        with pytest.raises(ValueError, match="unique"):
            checker.check_batch(frames, env_ids=(1, 1))
        assert checker.last_batch_pairs == ()
        malformed = {key: value.copy() for key, value in frames.items()}
        malformed["arm"][0, 3, 0] = 1
        with pytest.raises(ValueError, match="rigid"):
            checker.check_batch(malformed)
    finally:
        checker.close()


def test_phase_branches_must_connect_and_dual_targets_stay_synchronized():
    identity = np.eye(4).tolist()
    mode = TaskOperationMode(
        "handover",
        (
            TaskPhase("lift", (PhaseTarget("left", identity),)),
            TaskPhase(
                "exchange",
                (PhaseTarget("left", identity), PhaseTarget("right", identity)),
                dependencies=("lift",),
            ),
        ),
    )

    class Branches:
        initial_qpos = np.zeros(2)
        joint_limits = np.array([[-3, 3], [-3, 3]])
        joint_names = ("left", "right")

        def solutions(self, phase, start, entities, budget):
            budget.charge("ik")
            if phase.phase_id == "lift":
                return (np.array([1.0, 0]), np.array([-1.0, 0]))
            assert {t.control_part for t in phase.targets} == {"left", "right"}
            return (np.array([-2.0, 2]),)

        def connect(self, start, goal, phase, entities, budget):
            budget.charge("planning")
            if phase.phase_id == "exchange" and start[0] > 0:
                return None
            return np.stack((start, goal))

        def resulting_poses(self, q, phase, entities):
            return entities

    result = plan_task_mode(
        mode, Branches(), {}, FeasibilityBudget(SceneRobotFeasibilityCfg())
    )
    assert result.accepted
    assert result.paths[0][-1] == pytest.approx((-1, 0))

    class Blocked(Branches):
        def connect(self, *args):
            return None

    result = plan_task_mode(
        mode, Blocked(), {}, FeasibilityBudget(SceneRobotFeasibilityCfg())
    )
    assert not result.accepted and result.checks[0].stage == "motion"


def test_robot_adapter_uses_hypothetical_roots_and_checks_joint_tcp_conflicts():
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim.scene.feasibility import (
        RobotFeasibilityBackend,
        FeasibilityCheck,
        HeldObject,
    )

    def pose(z):
        value = np.eye(4)
        value[2, 3] = z
        return value

    class Transform:
        def __init__(self, z):
            self.value = torch.tensor(pose(float(z)), dtype=torch.float32)[None]

        def get_matrix(self):
            return self.value

    class Chain:
        def get_joint_parameter_names(self):
            return ["lift", "arm"]

        def forward_kinematics(self, q):
            return {
                "base": Transform(0),
                "arm_base": Transform(q[0, 0]),
                "tcp": Transform(q.sum()),
            }

    class Solver:
        dof = 1

        def __init__(self, name):
            self.root_link_name = "base" if name == "torso" else "arm_base"

        def get_fk(self, qpos):
            return torch.tensor(pose(float(qpos[0, 0])), dtype=torch.float32)[None]

    class RobotPort:
        device = torch.device("cpu")
        uid = "r"
        joint_names = ["lift", "arm"]
        pk_chain = Chain()
        cfg = SimpleNamespace(workspace_cfg={})
        mimic_ids = []
        mimic_parents = []
        mimic_multipliers = []
        mimic_offsets = []

        def get_qpos(self):
            return torch.zeros(1, 2)

        def get_qpos_limits(self):
            return torch.tensor([[[-2.0, 2.0], [-2.0, 2.0]]])

        def get_local_pose(self, **kwargs):
            return torch.eye(4)[None]

        def get_link_pose(self, **kwargs):
            return torch.eye(4)[None]

        def get_solver(self, name):
            return Solver(name)

        def get_joint_ids(self, name):
            return [0] if name == "torso" else [1]

        def compute_ik(self, pose, **kwargs):
            return torch.tensor([True]), pose[:, 2, 3, None].clone()

        def compute_fk(self, qpos, **kwargs):
            return Solver("arm").get_fk(qpos)

    class CollisionPort:
        def check(self, *args, **kwargs):
            return FeasibilityCheck("collision", "passed"), 0.1

    cfg = SceneRobotFeasibilityCfg(ik_seeds=1)
    backend = RobotFeasibilityBackend(RobotPort(), CollisionPort(), cfg, {})
    conflict = TaskOperationMode(
        "conflict",
        (
            TaskPhase(
                "move", (PhaseTarget("arm", pose(1)),), joint_targets=(("arm", 0.0),)
            ),
        ),
    )
    with pytest.raises(ValueError, match="same joints"):
        plan_task_mode(conflict, backend, {}, FeasibilityBudget(cfg))
    mode = TaskOperationMode(
        "lift_then_arm",
        (
            TaskPhase("lift", joint_targets=(("lift", 1.0),)),
            TaskPhase("arm", (PhaseTarget("arm", pose(1)),), dependencies=("lift",)),
        ),
    )
    result = plan_task_mode(mode, backend, {}, FeasibilityBudget(cfg))
    assert result.accepted and result.paths[-1][-1] == pytest.approx((1.0, 0.0))
    held = TaskPhase("carry", held_objects=(HeldObject("payload", "arm"),))
    assert backend.frame_poses(np.array([1.0, 0.0]), held, {})[
        "payload"
    ] == pytest.approx(pose(1.0))
    simultaneous = TaskOperationMode(
        "synchronized",
        (
            TaskPhase(
                "both", (PhaseTarget("arm", pose(1)), PhaseTarget("torso", pose(1)))
            ),
        ),
    )
    result = plan_task_mode(simultaneous, backend, {}, FeasibilityBudget(cfg))
    assert result.accepted and result.paths[-1][-1] == pytest.approx((1.0, 0.0))


def test_robot_adapter_projects_mimic_follower_for_collision():
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim.scene.feasibility import RobotFeasibilityBackend

    pytest.importorskip("newton")

    class Transform:
        def __init__(self, x, y):
            self.value = torch.eye(4)[None]
            self.value[0, 0, 3] = x
            self.value[0, 1, 3] = y

        def get_matrix(self):
            return self.value

    class Chain:
        def get_joint_parameter_names(self):
            return ["leader", "follower"]

        def forward_kinematics(self, q):
            return {
                "leader": Transform(q[0, 0], 1.0),
                "follower": Transform(q[0, 1], 0.0),
            }

    class RobotPort:
        device = torch.device("cpu")
        uid = "r"
        joint_names = ["leader", "follower"]
        pk_chain = Chain()
        cfg = SimpleNamespace(workspace_cfg={})
        mimic_ids = [1]
        mimic_parents = [0]
        mimic_multipliers = [1.0]
        mimic_offsets = [0.0]

        def get_qpos(self):
            return torch.zeros(1, 2)

        def get_qpos_limits(self):
            return torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

        def get_local_pose(self, **kwargs):
            return torch.eye(4)[None]

    world = np.eye(4)
    world[0, 3] = 1.0
    bodies = (
        CollisionBody("leader", "r/leader", "sphere", (0.1,), robot=True),
        CollisionBody("follower", "r/follower", "sphere", (0.1,), robot=True),
        CollisionBody("wall", "wall", "sphere", (0.1,)),
    )
    cfg = SceneRobotFeasibilityCfg(max_rrt_nodes=1)
    backend = RobotFeasibilityBackend(
        RobotPort(), NewtonCollisionChecker(bodies), cfg, {"wall": world}
    )
    mode = TaskOperationMode(
        "close", (TaskPhase("close", joint_targets=(("leader", 1.0),)),)
    )
    result = plan_task_mode(mode, backend, {"wall": world}, FeasibilityBudget(cfg))
    assert not result.accepted
    phase = TaskPhase("observe")
    frames = backend.frame_poses(np.array([1.0, 0.0]), phase, {}, measured=True)
    assert frames["r/follower"][0, 3] == 0.0
    frames = backend.frame_poses(np.array([1.0, 0.0]), phase, {})
    assert frames["r/follower"][0, 3] == 1.0
    # Measured limit excursions are allowed only at the observed first point;
    # planned/controller targets remain strictly within the actual limits.
    port = RobotPort()
    port.get_qpos = lambda: torch.full((1, 2), -5e-5)
    safe = RobotFeasibilityBackend(
        port, NewtonCollisionChecker(bodies), cfg, {"wall": world}
    )
    close = TaskOperationMode(
        "close", (TaskPhase("close", joint_targets=(("leader", 0.3),)),)
    )
    planned = plan_task_mode(close, safe, {"wall": world}, FeasibilityBudget(cfg))
    assert planned.accepted
    assert planned.paths[0][0] == pytest.approx((-5e-5, -5e-5))
    assert np.all(planned.paths[0][1:] >= 0)
    port.get_qpos = lambda: torch.full((1, 2), -0.001)
    unsafe = RobotFeasibilityBackend(
        port, NewtonCollisionChecker(bodies), cfg, {"wall": world}
    )
    assert not plan_task_mode(
        close, unsafe, {"wall": world}, FeasibilityBudget(cfg)
    ).accepted


def test_halfspace_uses_physical_support_and_includes_ground():
    pytest.importorskip("newton")
    checker = NewtonCollisionChecker(
        (
            CollisionBody("ground", "ground", "halfspace", (0.0, 0.0, 1.0, 0.0)),
            CollisionBody("robot", "robot", "box", (0.2, 0.2, 0.2), robot=True),
        )
    )
    pose = np.eye(4)
    pose[2, 3] = 0.2
    report, clearance = checker.check({"ground": np.eye(4), "robot": pose})
    assert report.status == "passed" and 0 <= clearance <= checker.detection_gap_m
    assert SceneRobotFeasibilityCfg(collision_device="cpu:0").collision_device == "cpu"
    pose[2, 3] = 0.05
    assert checker.check({"ground": np.eye(4), "robot": pose})[0].status == "failed"


@pytest.mark.parametrize("reverse", (False, True))
def test_declared_convex_contains_robot_even_without_surface_intersection(reverse):
    import trimesh

    pytest.importorskip("newton")
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    container = CollisionBody(
        "container",
        "container",
        "convex",
        vertices=tuple(map(tuple, mesh.vertices)),
        triangles=tuple(tuple(int(i) for i in f) for f in mesh.faces),
    )
    robot = CollisionBody("robot", "robot", "sphere", (0.1,), robot=True)
    bodies = (robot, container) if reverse else (container, robot)
    checker = NewtonCollisionChecker(bodies)
    assert (
        checker.check({"container": np.eye(4), "robot": np.eye(4)})[0].status
        == "failed"
    )


def test_triangle_mesh_is_explicitly_unsupported_instead_of_being_convexified():
    import trimesh

    pytest.importorskip("newton")
    mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=2.0, sections=24)
    shell = CollisionBody(
        "ring",
        "ring",
        "mesh",
        vertices=tuple(map(tuple, mesh.vertices)),
        triangles=tuple(tuple(int(i) for i in f) for f in mesh.faces),
    )
    with pytest.raises(UnsupportedFeasibilityError, match="Triangle meshes"):
        NewtonCollisionChecker(
            (shell, CollisionBody("robot", "robot", "sphere", (0.05,), robot=True))
        )


def test_contact_permissions_reject_ambiguous_frame_and_shape_ids():
    pytest.importorskip("newton")
    with pytest.raises(ValueError, match="unambiguous"):
        NewtonCollisionChecker(
            (
                CollisionBody("arm/0", "arm", "sphere", (0.1,), robot=True),
                CollisionBody("other", "arm/0", "sphere", (0.1,)),
            )
        )


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_newton_clearance_cap_masks_and_owned_query_lifetime(device):
    checker = NewtonCollisionChecker(
        (
            CollisionBody("arm", "arm", "sphere", (0.1,), robot=True),
            CollisionBody("payload", "payload", "box", (0.2, 0.2, 0.2)),
        ),
        device=device,
        clearance_m=0.02,
    )
    fixed = np.eye(4)
    target = np.eye(4)
    target[0, 3] = 1.0
    frames = {"arm": fixed, "payload": target}
    before = {k: v.copy() for k, v in frames.items()}
    report, clearance = checker.check(frames)
    assert report.status == "passed" and 0.02 <= clearance <= checker.detection_gap_m
    target[0, 3] = 0.21
    assert checker.check(frames)[0].status == "failed"
    assert (
        checker.check(frames, allowed_contacts=(("arm", "payload"),))[0].status
        == "passed"
    )
    assert checker.check(frames)[0].status == "failed"
    target[0, 3] = 1.0
    assert checker.check(frames)[0].status == "passed"
    assert all(np.array_equal(frames[k], v) for k, v in before.items())
    assert checker.metadata["clearance_scope"] == "capped_contact_band"
    if device.startswith("cuda"):
        assert checker.metadata["cuda_graph"] == "captured"
    checker.close()
    checker.close()
    with pytest.raises(RuntimeError, match="closed"):
        checker.check(frames)


def test_newton_saturated_contact_buffer_cannot_accept_a_configuration():
    checker = NewtonCollisionChecker(
        (
            CollisionBody("a", "a", "box", (0.2, 0.2, 0.2), robot=True),
            CollisionBody("b", "b", "box", (0.2, 0.2, 0.2)),
        ),
        max_contacts=1,
    )
    with pytest.raises(UnsupportedFeasibilityError, match="overflowed"):
        checker.check({"a": np.eye(4), "b": np.eye(4)})
    checker.close()


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_newton_rebased_coordinates_preserve_overlapping_spheres(device):
    checker = NewtonCollisionChecker(
        (
            CollisionBody("a", "a", "sphere", (0.1,), robot=True),
            CollisionBody("b", "b", "sphere", (0.1,)),
        ),
        device=device,
    )
    for a, b in (
        (np.array([1e40, 0.0, 0.0]), np.array([1e40, 0.0, 0.0])),
        (
            np.array([105.60602987818785, 99.58739128249081, 97.52718946961653]),
            np.array([105.75440541212491, 99.68305045067996, 97.62117683297595]),
        ),
    ):
        assert np.linalg.norm(b - a) < 0.2
        first, second = np.eye(4), np.eye(4)
        first[:3, 3], second[:3, 3] = a, b
        assert checker.check({"a": first, "b": second})[0].status == "failed"
    checker.close()


def test_newton_invalid_composition_or_query_precision_is_unsupported():
    local = np.eye(4)
    local[0, 3] = 1e308
    checker = NewtonCollisionChecker(
        (
            CollisionBody("floor", "floor", "halfspace", (0, 0, 1, 0)),
            CollisionBody("r", "r", "sphere", (0.1,), local_pose=local, robot=True),
        ),
    )
    frame = np.eye(4)
    frame[0, 3], frame[2, 3] = 1e308, -1
    with pytest.raises(UnsupportedFeasibilityError, match="composed"):
        checker.check({"floor": np.eye(4), "r": frame})
    checker.close()
    checker = NewtonCollisionChecker(
        (
            CollisionBody("a", "a", "sphere", (0.1,), robot=True),
            CollisionBody("b", "b", "sphere", (0.1,)),
        ),
    )
    frame[0, 3] = 1e40
    with pytest.raises(UnsupportedFeasibilityError, match="precision"):
        checker.check({"a": np.eye(4), "b": frame})
    checker.close()
    for radius in (1e40, 1e-50):
        with pytest.raises(UnsupportedFeasibilityError):
            NewtonCollisionChecker(
                (CollisionBody("s", "s", "sphere", (radius,), robot=True),)
            )


def test_newton_translated_halfspace_preserves_small_shape_support():
    local = np.eye(4)
    local[2, 3] = 0.05
    checker = NewtonCollisionChecker(
        (
            CollisionBody("floor", "floor", "halfspace", (0, 0, 1, 0)),
            CollisionBody(
                "sphere", "sphere", "sphere", (0.1,), local_pose=local, robot=True
            ),
        ),
    )
    for height in (0.0, 1e16, 1e40):
        pose = np.eye(4)
        pose[2, 3] = height
        assert checker.check({"floor": pose, "sphere": pose})[0].status == "failed"
    checker.close()


def test_robot_rrt_routes_around_blockage_and_exhausts_a_complete_barrier():
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim.scene.feasibility import RobotFeasibilityBackend

    pytest.importorskip("newton")

    class Transform:
        def __init__(self, q):
            self.value = torch.eye(4)[None]
            self.value[0, :2, 3] = q

        def get_matrix(self):
            return self.value

    class Chain:
        def get_joint_parameter_names(self):
            return ["x", "y"]

        def forward_kinematics(self, q):
            return {"tcp": Transform(q[0])}

    class RobotPort:
        device = "cpu"
        uid = "r"
        joint_names = ["x", "y"]
        pk_chain = Chain()
        cfg = SimpleNamespace(workspace_cfg={})
        mimic_ids = []
        mimic_parents = []
        mimic_multipliers = []
        mimic_offsets = []

        def get_qpos(self):
            return torch.tensor([[-0.8, 0.0]])

        def get_qpos_limits(self):
            return torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])

        def get_local_pose(self, **kwargs):
            return torch.eye(4)[None]

        def get_solver(self, name):
            return SimpleNamespace(dof=2)

        def get_joint_ids(self, name):
            return [0, 1]

    identity = np.eye(4)
    phase = TaskPhase("move", (PhaseTarget("gantry", identity),))
    cfg = SceneRobotFeasibilityCfg(max_motion_samples=8192, max_rrt_nodes=256)

    def backend(height):
        checker = NewtonCollisionChecker(
            (
                CollisionBody("tool", "r/tcp", "sphere", (0.05,), robot=True),
                CollisionBody("wall", "wall", "box", (0.4, height, 0.2)),
            )
        )
        return RobotFeasibilityBackend(
            RobotPort(), checker, cfg, {"wall": identity}, seed=7
        )

    start, goal = np.array([-0.8, 0.0]), np.array([0.8, 0.0])
    path = backend(0.8).connect(
        start, goal, phase, {"wall": identity}, FeasibilityBudget(cfg)
    )
    assert path is not None and np.allclose(path[[0, -1]], np.stack((start, goal)))
    # Independent sphere-to-axis-aligned-box distance at each published sample.
    gaps = np.maximum(np.abs(path) - np.array([0.2, 0.4]), 0.0)
    assert np.all(np.linalg.norm(gaps, axis=1) >= 0.05 - 1e-6)
    assert np.max(np.abs(path[:, 1])) > 0.45
    assert (
        backend(3.0).connect(
            start, goal, phase, {"wall": identity}, FeasibilityBudget(cfg)
        )
        is None
    )
