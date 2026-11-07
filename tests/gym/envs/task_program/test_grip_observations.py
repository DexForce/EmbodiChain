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

"""Fresh grip ownership through the standard factory, without native construction."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    BufferedGymCommandSink,
    EnvironmentStepClock,
    RuntimeCommandFrameEncoder,
    TaskProgramDemoBridge,
)
from embodichain.lab.gym.utils.gym_utils import (
    _close_raw_rollout,
    _init_raw_rollout,
    _write_raw_rollout,
)
from embodichain.lab.sim.atomic_skills import (
    AtomicSkillEngine,
    CARTESIAN_POSE_CAPABILITY,
)
from embodichain.lab.sim.atomic_skills.grip_evidence import GripEvidenceObservationPort
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations import (
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    SimulationTaskProgramFactory,
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.integrations.simulation.grip_evidence import (
    NativeGripEvidenceCfg,
    NativeGripEvidenceFactory,
)
from embodichain.lab.task_program.integrations.simulation.observations import (
    SimulationStateObservationCollector,
)
from embodichain.lab.task_program.semantics import EffectAssurance, SkillPolicyPreset
from embodichain.lab.task_program.semantics.evidence import (
    EffectEvidenceCollector,
    EffectEvidenceProviderRegistry,
)


def _case(monkeypatch, *, configured=True):
    """Use actual public factories with CPU registries and deterministic contacts."""
    count = 2
    updates, captures, providers, scope_reads = [], [], [], []
    runtime_scope = {"current": True}
    data = dict(
        is_valid=torch.ones(count, 2, dtype=torch.bool),
        user_ids=torch.tensor([[[1, 2], [1, 3]], [[11, 12], [11, 13]]]),
        normal=torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]] * count),
        position=torch.zeros(count, 2, 3),
        impulse=torch.full((count, 2), 0.1),
    )
    sensor = SimpleNamespace(
        num_instances=count,
        contact_capabilities=SimpleNamespace(geometry=True, impulse=True),
        dropped_contacts=0,
        update=lambda: updates.append(1),
        get_data=lambda: data,
        get_actor_ids=lambda uid, names=None: (
            torch.tensor([[1], [11]])
            if names is None
            else torch.tensor([[2, 3], [12, 13]])
        ),
    )
    sensors = {}

    def add_sensor(cfg):
        sensor.cfg = cfg
        sensors[cfg.uid] = sensor
        return sensor

    robot = SimpleNamespace(
        uid="robot",
        device=torch.device("cpu"),
        dof=1,
        num_instances=count,
        control_parts={"arm": object(), "hand": object()},
        get_joint_ids=lambda name: [0],
        get_solver=lambda name: object(),
        get_qpos=lambda **kwargs: torch.zeros(count, 1),
        get_qvel=lambda **kwargs: torch.zeros(count, 1),
        compute_fk=lambda **kwargs: torch.eye(4).repeat(count, 1, 1),
    )
    body = SimpleNamespace(uid="native_cube")
    bodies = {body.uid: body}
    simulation = SimpleNamespace(
        num_envs=count,
        simulation_time=0.08,
        spawn_result=SimpleNamespace(topology_revision=1),
        get_robot=lambda uid: robot if uid == robot.uid else None,
        get_rigid_object=bodies.get,
        get_sensor=sensors.get,
        add_sensor=add_sensor,
    )
    scene = SimulationSceneBinding(
        "scene", rigid_objects=(SimulationRigidObjectBinding("cube", body.uid),)
    )
    profile = SimulationRobotSkillProfileBinding(
        "profile",
        resources=(
            ControlPartResourceBinding(
                "primary",
                endpoints=(
                    ControlPartEndpointBinding(
                        "motion", "arm", frozenset({CARTESIAN_POSE_CAPABILITY})
                    ),
                ),
            ),
        ),
        presets=(
            SkillPolicyPreset(
                "default",
                skill_option_templates={},
                effect_assurance=EffectAssurance.PROJECTED,
            ),
        ),
        default_preset="default",
    )
    native_factory = NativeGripEvidenceFactory(
        NativeGripEvidenceCfg(
            sensor_uid="contacts",
            object_bindings=(("cube", body.uid),),
            pad_bindings=(("hand", ("first_pad", "second_pad")),),
        )
    )
    registration = SimulationTaskProgramRegistration(
        scene,
        profile,
        control_part_evidence_factory=native_factory if configured else None,
    )
    create = NativeGripEvidenceFactory.create

    def retain_provider(self, **kwargs):
        provider = create(self, **kwargs)
        capture = provider.capture_grip_observation

        def retain_capture(*args, **kwargs):
            batch, raw = capture(*args, **kwargs)
            captures.append((batch.snapshot(), deepcopy(raw)))
            return batch, raw

        provider.capture_grip_observation = retain_capture
        providers.append(provider)
        return provider

    monkeypatch.setattr(NativeGripEvidenceFactory, "create", retain_provider)
    generator = MagicMock(spec=MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(planner_type="test"))
    generator.collision_world_info = None
    generator.dynamic_collision_entity_ids = ()
    generator.collision_world_entity_ids = ()
    generator.supports_dynamic_collision_world = False
    generator.collision_world_batch_mode = None
    factory = SimulationTaskProgramFactory(
        simulation,
        robot,
        registration,
        step_dt=0.04,
        motion_generator_factory=lambda: generator,
    )
    clock = EnvironmentStepClock(0.04)
    context = EpisodeContext(count)
    context._begin_episode([0, 1])
    context._finish_initialization([0, 1])

    def require_scope():
        scope_reads.append(1)
        if not runtime_scope["current"]:
            raise ValueError("Runtime invocation scope replaced.")

    engine = AtomicSkillEngine(generator, load_builtins=False)
    registry = factory.create_scene_registry()
    observation = factory.create_planning_observation_provider(
        scene_registry=registry,
        engine=engine,
        clock=clock,
        runtime_scope_guard=require_scope,
    )
    wrapped = tuple(
        factory.create_effect_evidence_providers(
            scene_registry=registry, engine=engine, observation_provider=observation
        )
    )
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *args, **kwargs: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
        evidence_collector=EffectEvidenceCollector(
            EffectEvidenceProviderRegistry(wrapped)
        ),
    )
    program = TaskProgramCompiler(scene.declare()).compile(
        decode_task_program(
            dict(
                program_id="grip-observation",
                targets={},
                integration=dict(
                    robot_profile="profile",
                    scene_registry="scene",
                    runtime_preset="default",
                ),
                program=dict(
                    kind="segment",
                    name="observe",
                    steps=dict(kind="invoke", call=dict(kind="pick", object="cube")),
                ),
            )
        )
    )
    bridge = TaskProgramDemoBridge(
        program,
        runtime,
        BufferedGymCommandSink(RuntimeCommandFrameEncoder(observation), clock),
        clock,
        post_policy_port=factory.segment_policy_port,
        isolate_policy_sessions=True,
    )
    bridge.post_policy_port.bind_clock(clock)
    env = SimpleNamespace(
        sim=simulation, robot=robot, episode_context=context, _closed=False
    )
    observations = ObservationManager({}, env)
    storage = TensorDict({}, batch_size=[count])
    _init_raw_rollout(storage, context, {}, torch.zeros(count, 2, 1))
    observations.bind_rollout_storage(storage)
    collector = SimulationStateObservationCollector(
        env, bridge, registration, observations=observations
    )
    descriptor = ObservationDescriptor(
        "probe", "boundary", "sensor", "scalar", ("s",), (1,), torch.float32
    )
    for slot in range(3):
        sample = ObservationSample(
            descriptor,
            torch.full((count, 1), float(slot)),
            torch.ones(count, dtype=torch.bool),
            (None, None),
        )
        frame = ObservationFrame(
            None,
            {"boundary": sample},
            context.episode_ids,
            slot,
            slot * 0.04,
            "simulation_time",
            context,
        )
        observations._bind_schema(frame)
        _write_raw_rollout(
            storage,
            context,
            frame,
            **(
                {"field_ids": ("boundary",), "env_ids": [0, 1]}
                if slot == 0
                else {
                    "action": torch.zeros(count, 1),
                    "active": torch.tensor([True, slot == 1]),
                }
            ),
        )
        if slot:
            clock.advance_after_env_step()
    scope_reads.clear()
    return SimpleNamespace(
        collector=collector,
        env=env,
        bridge=bridge,
        factory=factory,
        storage=storage,
        sensor=sensor,
        sensors=sensors,
        bodies=bodies,
        providers=providers,
        wrapped=wrapped,
        captures=captures,
        updates=updates,
        scope_reads=scope_reads,
        runtime_scope=runtime_scope,
    )


def _capture(case):
    return case.collector.capture_current_grip(
        "cube", "hand", observation_revision=9, frame_indices=(2, 1)
    )


@pytest.mark.parametrize("unavailable", [False, True])
def test_standard_factory_port_preserves_one_exact_owned_grip_query(
    monkeypatch, unavailable
):
    case = _case(monkeypatch)
    if unavailable:
        case.sensor.get_data()["normal"][1, 0, 0] = float("nan")
    assert isinstance(case.wrapped[0], GripEvidenceObservationPort)
    result = _capture(case)
    batch, raw = case.captures[0]
    assert {key: result[key] for key in raw} == raw
    assert result["boundary_frames"] == [2, 1] and result["episode_ids"] == [0, 0]
    assert result["query"]["query_index"] == 1
    assert result["query"]["timestamp"] == case.bridge.clock.now()
    assert result["query"]["observation_revision"] == 9
    assert (
        result["timestamp_scope"] == "collection_query_clock_not_native_impulse_refresh"
    )
    acquisition = result["grip_acquisition"]
    assert acquisition["values"] == batch.values.tolist()
    assert acquisition["valid"] == [True, not unavailable] == batch.valid.tolist()
    assert acquisition["acquisition_errors"] == list(batch.acquisition_errors)
    assert acquisition["clock_domain"] == "environment_step"
    assert acquisition["revision_kind"] == "semantic_query_revision"
    assert (
        case.updates == [1]
        and case.providers[0].grip_observer.cumulative_query_count == 1
    )
    assert case.scope_reads == [1, 1]
    original = deepcopy(raw)
    result["query"]["timestamp"] = -1
    case.sensor.get_data()["normal"].zero_()
    assert (
        case.providers[0].grip_observer.snapshot_latest_query(
            object_id="cube", control_part="hand", query_index=1
        )
        == original
    )
    assert case.updates == [1]


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize(
    "damage",
    ["closed", "runtime", "episode", "robot", "sensor", "object", "registry", "clock"],
)
def test_factory_port_rejects_replaced_scope_before_and_after_acquisition(
    monkeypatch, damage, when
):
    case = _case(monkeypatch)

    def replace_owner():
        if damage == "closed":
            case.factory.close_runtime_resources()
        elif damage == "runtime":
            case.runtime_scope["current"] = False
        elif damage == "episode":
            case.env.episode_context._begin_episode([0])
        elif damage == "robot":
            case.env.sim.get_robot = lambda uid: object()
        elif damage == "sensor":
            case.sensors["contacts"] = object()
        elif damage == "object":
            case.bodies["native_cube"] = object()
        elif damage == "registry":
            case.bridge._runtime.evidence_collector = object()
        else:
            case.bridge.post_policy_port.bind_clock(EnvironmentStepClock(0.04))

    if when == "before":
        replace_owner()
    else:

        def update():
            case.updates.append(1)
            replace_owner()

        case.sensor.update = update
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        _capture(case)
    assert len(case.updates) == (when == "during")
    assert case.providers[0].grip_observer.cumulative_query_count == (when == "during")


@pytest.mark.parametrize("damage", ["unconfigured", "boundary", "closed"])
def test_fresh_grip_requires_declared_capability_and_exact_recorded_boundary(
    monkeypatch,
    damage,
):
    case = _case(monkeypatch, configured=damage != "unconfigured")
    if damage == "unconfigured":
        assert not any(
            isinstance(provider, GripEvidenceObservationPort)
            for provider in case.wrapped
        )
        error_type, reason = ValueError, "explicitly configured"
    elif damage == "boundary":
        case.storage["raw_steps"][1] = 0
        error_type, reason = ValueError, "current boundary"
    else:
        _close_raw_rollout(case.storage)
        error_type, reason = RuntimeError, "after close"
    with pytest.raises(error_type, match=reason):
        _capture(case)
    assert case.updates == []
