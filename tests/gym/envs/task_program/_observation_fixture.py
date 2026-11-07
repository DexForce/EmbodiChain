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

"""Actual raw manager, policy bridge and native-shaped borrowed inputs."""

from __future__ import annotations

from types import SimpleNamespace as NS
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
from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations import SimulationTaskProgramRegistration
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
    SimulationRigidObjectBinding,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
)
from .test_simulation_environment import (
    _Robot,
    _RigidObject,
    _profile_binding,
    _motion_generator,
)


def observation_host(*, capacity=130, recorded_cfg=None, cargo_cfg=None):
    robot = _Robot()
    robot.qpos = robot.qpos[:1]
    robot.target_qpos = robot.target_qpos[:1]
    robot.num_instances = 1
    bodies = {}
    for name in ("tray", "cargo", "tool"):
        body = _RigidObject()
        body.pose = body.pose[:1]
        body.uid, body.body_type, body.num_instances = name, "dynamic", 1
        if name != "tray":
            body.pose[0, 2, 3] = 0.06
        bodies[name] = body
    revision = [0]
    physics = NS(_get_contact_data_revision=lambda: revision[0])
    sim = NS(
        num_envs=1,
        simulation_time=0.1,
        physics_backend="default",
        is_default_backend=True,
        spawn_result=NS(backend="dexsim", topology_revision=5),
        get_robot=lambda uid: robot if uid == robot.uid else None,
        get_rigid_object=bodies.get,
        get_articulation=lambda uid: None,
        get_world=lambda: NS(get_physics_scene=lambda: physics),
    )
    binding = SimulationSceneBinding(
        registry_id="observation_scene",
        rigid_objects=tuple(
            SimulationRigidObjectBinding(name, name) for name in bodies
        ),
    )
    registration = SimulationTaskProgramRegistration(
        binding,
        _profile_binding(),
        recorded_contacts_cfg=recorded_cfg,
        cargo_envelope_cfg=cargo_cfg,
    )
    program = TaskProgramCompiler(binding.declare()).compile(
        decode_task_program(
            {
                "program_id": "physical_facts",
                "targets": {},
                "integration": {
                    "robot_profile": "robot_profile",
                    "scene_registry": "observation_scene",
                    "runtime_preset": "safe",
                },
                "program": {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "cargo"},
                },
            }
        )
    )
    clock = EnvironmentStepClock(0.04)
    port = SimulationSegmentPolicyPort(
        sim, robot, binding, settle_presets=registration.settle_presets, step_dt=0.04
    )
    port.bind_clock(clock)
    runtime = NS(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    bridge = TaskProgramDemoBridge(
        program,
        runtime,
        BufferedGymCommandSink(
            RuntimeCommandFrameEncoder(NS(current_qpos=lambda ids: robot.get_qpos())),
            clock,
        ),
        clock,
        post_policy_port=port,
        isolate_policy_sessions=False,
    )
    context = EpisodeContext(1)
    context._begin_episode([0])
    context._finish_initialization([0])
    env = NS(
        sim=sim,
        robot=robot,
        num_envs=1,
        step_dt=0.04,
        episode_context=context,
        active_task_program_bridge=bridge,
        _closed=False,
    )
    manager = ObservationManager({}, env)
    env.observation_manager = manager
    storage = TensorDict({}, batch_size=[1])
    _init_raw_rollout(storage, context, {}, torch.zeros(1, capacity, robot.dof))
    manager.bind_rollout_storage(storage)
    reads = []
    original = manager.read_history_window

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return original(*args, **kwargs)

    manager.read_history_window = read

    def write(*, invalid=None, timestamp=True, seed=False):
        fields = {}
        for name, body in bodies.items():
            values = torch.zeros(1, 7)
            values[:, :3] = body.pose[:, :3, 3]
            values[:, 6] = 1
            good = name != invalid
            fields[name] = ObservationSample(
                ObservationDescriptor(
                    name,
                    "pose",
                    "arena",
                    "xyz_xyzw",
                    ("m", "unitless"),
                    (7,),
                    torch.float32,
                ),
                values,
                torch.tensor([good]),
                (None if good else "raw source unavailable",),
            )
        slot = clock.step_index
        frame = ObservationFrame(
            None,
            fields,
            context.episode_ids,
            slot,
            timestamp=sim.simulation_time if timestamp else None,
            clock_domain="simulation_time",
            episode_owner=context,
        )
        manager._bind_schema(frame)
        _write_raw_rollout(
            storage,
            context,
            frame,
            **(
                {"field_ids": tuple(manager.field_ids.values()), "env_ids": [0]}
                if seed
                else {"action": torch.zeros(1, robot.dof)}
            ),
        )

    write(seed=True)
    actors = {name: i + 1 for i, name in enumerate(bodies)}
    hooks = {}
    fetches = [0]
    data = {
        "is_valid": torch.tensor([[True]]),
        "user_ids": torch.tensor([[[actors["cargo"], actors["tray"]]]]),
        "normal": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "position": torch.zeros(1, 1, 3),
        "impulse": torch.ones(1, 1),
    }

    def sample():
        fetches[0] += 1
        return data

    def add(name, observer):
        if name in hooks:
            raise ValueError("Observer name already owned")
        hooks[name] = observer

    def remove(name, *, expected):
        if hooks.get(name) is expected:
            hooks.pop(name)

    sensor = NS(
        num_instances=1,
        cfg=NS(
            rigid_uid_list=list(bodies),
            articulation_cfg_list=[],
            filter_need_both_actor=False,
            max_contacts_per_env=1024,
        ),
        contact_capabilities=NS(geometry=True, impulse=True),
        dropped_contacts=0,
        get_data=sample,
        get_actor_ids=lambda uid: torch.tensor([[actors[uid]]]),
        add_sampling_observer=add,
        remove_sampling_observer=remove,
    )
    sim.get_sensor = lambda uid: sensor if uid == "contacts" else None
    env.sensors = {"contacts": sensor}

    def advance(*, invalid=None, timestamp=True):
        for observer in tuple(hooks.values()):
            observer.begin_control_step()
        for _ in range(2):
            revision[0] += 1
            sim.simulation_time += 0.02
            for observer in tuple(hooks.values()):
                token = observer.before_sample()
                observer.after_sample(0.02, token)
        clock.advance_after_env_step()
        write(invalid=invalid, timestamp=timestamp)

    return NS(
        env=env,
        bridge=bridge,
        registration=registration,
        manager=manager,
        storage=storage,
        bodies=bodies,
        robot=robot,
        sensor=sensor,
        hooks=hooks,
        fetches=fetches,
        advance=advance,
        reads=reads,
        motion=lambda: _motion_generator(robot),
    )
