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

"""Condition collection reuses canonical raw history and exact runtime owners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers import ObservationManager
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationFrame,
    ObservationSample,
)
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    EnvironmentStepClock,
    BufferedGymCommandSink,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout, _write_raw_rollout
from embodichain.lab.task_program import decode_task_program, TaskProgramCompiler
from embodichain.lab.task_program.integrations import SimulationTaskProgramRegistration
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
    SimulationRigidObjectBinding,
    SimulationArticulationBinding,
)
from embodichain.lab.task_program.integrations.simulation.observations import (
    SimulationStateObservationCollector,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
)

from .test_simulation_environment import (
    _Robot,
    _RigidObject,
    _Simulation,
    _profile_binding,
)


def _collector(*, history=True, invalid=False, timestamp=True, articulated=False):
    robot, cube = _Robot(), _RigidObject()
    cube.uid = "cube_native"
    sim = _Simulation(robot, {cube.uid: cube})
    sim.simulation_time = 0.04
    sim.articulations = {}
    sim.get_articulation = sim.articulations.get
    if articulated:
        sim.articulations["drawer_native"] = SimpleNamespace(
            uid="drawer_native",
            joint_names=("slide",),
            get_qpos=lambda: torch.tensor([[0.1], [0.2], [0.3]]),
        )
    binding = SimulationSceneBinding(
        registry_id="scene",
        rigid_objects=(
            SimulationRigidObjectBinding(entity_id="object", simulation_uid=cube.uid),
        ),
    )
    if articulated:
        binding = SimulationSceneBinding(
            registry_id="scene",
            rigid_objects=binding.rigid_objects,
            articulations=(SimulationArticulationBinding("drawer", "drawer_native"),),
        )
    registration = SimulationTaskProgramRegistration(binding, _profile_binding())
    compiled = TaskProgramCompiler(binding.declare()).compile(
        decode_task_program(
            {
                "program_id": "state",
                "targets": {},
                "integration": {
                    "robot_profile": "robot_profile",
                    "scene_registry": "scene",
                    "runtime_preset": "safe",
                },
                "program": {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "object"},
                },
            }
        )
    )
    clock = EnvironmentStepClock(0.04)
    port = SimulationSegmentPolicyPort(sim, robot, binding, step_dt=0.04)
    port.bind_clock(clock)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(
            SimpleNamespace(current_qpos=lambda ids: torch.zeros(len(ids), 1))
        ),
        clock,
    )
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda **kw: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        sink,
        clock,
        post_policy_port=port,
        isolate_policy_sessions=True,
    )
    context = EpisodeContext(3)
    context._begin_episode([0, 1, 2])
    context._finish_initialization([0, 1, 2])
    env = SimpleNamespace(
        sim=sim, robot=robot, num_envs=3, episode_context=context, _closed=False
    )
    manager = ObservationManager({}, env)
    if history:
        storage = TensorDict({}, batch_size=[3])
        _init_raw_rollout(storage, context, {}, torch.zeros(3, 2, 1))
        manager.bind_rollout_storage(storage)
        descriptor = ObservationDescriptor(
            cube.uid,
            "pose",
            "arena",
            "xyz_xyzw",
            ("m", "unitless"),
            (7,),
            torch.float32,
        )
        for slot in [0, 1]:
            values = torch.zeros(3, 7)
            values[:, 0] = slot
            values[:, 6] = 1
            mask = torch.tensor([True, not (invalid and slot == 1), True])
            errors = (
                None,
                "source unavailable" if invalid and slot == 1 else None,
                None,
            )
            frame = ObservationFrame(
                None,
                {"pose": ObservationSample(descriptor, values, mask, errors)},
                context.episode_ids,
                slot,
                timestamp=slot * 0.04 if timestamp or slot == 1 else None,
                clock_domain="simulation_time",
                episode_owner=context,
            )
            manager._bind_schema(frame)
            _write_raw_rollout(
                storage,
                context,
                frame,
                **(
                    {"field_ids": ("pose",), "env_ids": [0, 1, 2]}
                    if slot == 0
                    else {"action": torch.zeros(3, 1)}
                ),
            )
    collector = SimulationStateObservationCollector(
        env, bridge, registration, observations=manager if history else None
    )
    return collector, env, bridge, cube


def test_current_state_remains_available_without_recording_and_rejects_replaced_native_owner():
    collector, env, _, cube = _collector(history=False)
    packet = collector.capture_current_rigid_pose("object")
    assert packet["state_acquisition"]["valid"] == [True, True, True]
    env.sim.rigid_objects[cube.uid] = _RigidObject()
    with pytest.raises(ValueError, match="native object owner"):
        collector.capture_current_rigid_pose("object")


@pytest.mark.parametrize("invalid", [False, True])
def test_recorded_state_preserves_raw_availability_and_never_reads_live_pose(
    monkeypatch, invalid
):
    collector, env, bridge, cube = _collector(invalid=invalid)
    bridge.clock.advance_after_env_step()
    monkeypatch.setattr(
        cube,
        "get_local_pose",
        lambda **kw: pytest.fail("recorded state sampled a live pose"),
    )
    packet = collector.capture_recorded_rigid_pose("object", [1, 1, 1])
    assert packet["boundary_frames"] == [1, 1, 1]
    assert packet["state_acquisition"]["valid"] == [True, not invalid, True]
    assert packet["state_acquisition"]["values"][0][0][3] == 1.0
    assert packet["recorded_acquisition"]["timestamps"] == [0.04, 0.04, 0.04]
    assert packet["recorded_acquisition"]["observation_revisions"] == [1, 1, 1]
    if invalid:
        assert packet["state_acquisition"]["values"][1] is None
        assert (
            packet["state_acquisition"]["acquisition_errors"][1] == "source unavailable"
        )
    env.episode_context._begin_episode([0])
    with pytest.raises(ValueError, match="another episode"):
        collector.capture_recorded_rigid_pose("object", [1, 1, 1])


def test_recorded_state_requires_explicit_canonical_history():
    collector, _, _, _ = _collector(history=False)
    with pytest.raises(RuntimeError, match="Observation history"):
        collector.capture_recorded_rigid_pose("object", [1, 1, 1])


@pytest.mark.parametrize("kind", ["current_rigid", "current_joint", "recorded_rigid"])
def test_state_capture_rejects_native_owner_replaced_during_read(monkeypatch, kind):
    joint = kind == "current_joint"
    collector, env, bridge, cube = _collector(articulated=joint)
    bridge.clock.advance_after_env_step()
    table = env.sim.articulations if joint else env.sim.rigid_objects
    native = table["drawer_native"] if joint else cube

    def replace_owner():
        table[native.uid] = SimpleNamespace(uid=native.uid)

    if kind == "recorded_rigid":
        read = collector._observations.read_history

        def read_history(*args, **kwargs):
            snapshot = read(*args, **kwargs)
            replace_owner()
            return snapshot

        monkeypatch.setattr(collector._observations, "read_history", read_history)
        capture = lambda: collector.capture_recorded_rigid_pose("object", [1, 1, 1])
    else:
        read = native.get_qpos if joint else native.get_local_pose

        def read_current(**kwargs):
            values = read(**kwargs)
            replace_owner()
            return values

        monkeypatch.setattr(
            native, "get_qpos" if joint else "get_local_pose", read_current
        )
        capture = (
            (lambda: collector.capture_current_joint_position("drawer", "slide"))
            if joint
            else (lambda: collector.capture_current_rigid_pose("object"))
        )

    with pytest.raises(ValueError, match="native object owner"):
        capture()


def test_recorded_state_without_source_timestamp_stays_unavailable_per_row():
    collector, _, bridge, _ = _collector(timestamp=False)
    bridge.clock.advance_after_env_step()
    packet = collector.capture_recorded_rigid_pose("object", [1, 0, 1])
    assert packet["recorded_acquisition"]["timestamps"] == [0.04, None, 0.04]
    assert packet["state_acquisition"]["valid"] == [True, False, True]
    assert packet["state_acquisition"]["values"][1] is None
    assert packet["state_acquisition"]["acquisition_errors"][0] is None
    assert "timestamp" in packet["state_acquisition"]["acquisition_errors"][1]
    assert packet["state_acquisition"]["acquisition_errors"][2] is None
