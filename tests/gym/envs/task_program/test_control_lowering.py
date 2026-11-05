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
"""Source controls retain explicit identity, geometry and mass lineage."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionTarget,
    MoveJointsOptions,
    MoveEndEffectorOptions,
    PressOptions,
)
from embodichain.lab.sim.scene.press_inputs import PressBinding
from embodichain.lab.task_program.semantics import SceneArticulationRef
from embodichain.lab.task_program.integrations.simulation import (
    control_lowering as subject,
)
from embodichain.lab.task_program.integrations.extensions import (
    declare_registered_semantic_lowerer_factory,
)


def _assembly(monkeypatch, variant="calibrated"):
    binding = PressBinding(
        "button",
        "joint",
        "cap",
        "base",
        "source.usda",
        "a" * 64,
        "/mesh",
        1.0,
        (0.0, 0.0, -1.0),
        (0.0, 0.01),
        (0.0, 0.0, 0.0),
        0.0,
        0.01,
    )
    art = SimpleNamespace(
        uid="button",
        cfg=SimpleNamespace(
            fpath="source.usda",
            root_props=SimpleNamespace(fixed_base=True),
            body_scale=[1.0, 1.0, 1.0],
        ),
        joint_names=["joint"],
        get_qpos=lambda **kwargs: torch.zeros(1, 1),
        get_parent_joint_chain=lambda _: [
            SimpleNamespace(
                joint_type="prismatic",
                name="joint",
                parent_link_name="base",
                joint_limits=(0.0, 0.01),
            )
        ],
        get_link_pose=lambda *a, **k: torch.eye(4)[None],
    )
    robot = object()
    operation = SimpleNamespace(
        family="press",
        entity_id="cap",
        robot=robot,
        articulation=art,
        binding=binding,
        geometry=SimpleNamespace(motion_part="arm", finger_link_names=("finger",)),
        route_payload={"variant": variant, "extra_press_distance": 0.01},
    )
    active = [True]

    def lookup(key):
        if key != "operation" or not active[0]:
            raise ValueError("Operation scope expired")
        return operation

    services = SimpleNamespace(control_operation=lookup)
    sim = SimpleNamespace(num_envs=1, get_articulation=lambda _: art)
    registry = SimpleNamespace(
        lookup=lambda *a, **k: SimpleNamespace(
            parent=SceneArticulationRef("button"), native_name="cap"
        )
    )
    engine = SimpleNamespace(robot=robot, device="cpu")
    monkeypatch.setattr(subject, "inspect_press", lambda *a, **k: binding)
    endpoint = SimpleNamespace(
        task_state_key="hand",
        require_target=lambda _: JointPositionTarget("arm", (0, 1)),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            action_binding=SimpleNamespace(endpoint=lambda *a: endpoint)
        )
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.tensor([[0.2, 0.4, 0.8]])),
        task=SimpleNamespace(
            get_held_object=lambda _: None, coordinated_held_objects={}
        ),
    )
    return (
        dict(
            simulation=sim,
            robot=robot,
            scene_registry=registry,
            engine=engine,
            runtime_services=services,
        ),
        operation,
        active,
        context,
        bound,
    )


def test_press_prepare_preserves_observed_arm_and_checks_actual_scope(monkeypatch):
    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    factory = subject.PressPrepareFactory(
        subject.PressCallDeclaration("operation", "cap")
    )
    assert (
        declare_registered_semantic_lowerer_factory(factory).call_id
        == "gen_sim.press_prepare"
    )
    lowerer = factory.create_with_runtime_services(**kwargs)
    second = factory.create_with_runtime_services(**kwargs)
    assert second is not lowerer
    call = SimpleNamespace(arguments={"object": "button"})
    result = lowerer.lower(
        call, context=context, bound=bound, option_template=MoveJointsOptions()
    )
    assert result.skill_options is None
    assert torch.equal(result.goal.target, torch.tensor([[0.2, 0.4]]))
    active[0] = False
    with pytest.raises(ValueError, match="scope expired"):
        lowerer.lower(
            call, context=context, bound=bound, option_template=MoveJointsOptions()
        )


def test_press_factory_rejects_unscoped_or_changed_declarations(monkeypatch):
    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    factory = subject.PressFactory(subject.PressCallDeclaration("operation", "cap"))
    with pytest.raises(ValueError, match="actual runtime"):
        factory.create(**{k: v for k, v in kwargs.items() if k != "runtime_services"})
    operation.route_payload["extra_press_distance"] = 0.004
    with pytest.raises(ValueError, match="provenance"):
        factory.create_with_runtime_services(**kwargs)
    with pytest.raises(ValueError, match="Approach-only"):
        subject.PressFactory(
            subject.PressCallDeclaration("operation", "cap", "approach_only")
        )


def test_press_approach_keeps_exact_source_standoff(monkeypatch):
    kwargs, operation, active, context, bound = _assembly(monkeypatch, "approach_only")
    lowerer = subject.PressApproachFactory(
        subject.PressCallDeclaration("operation", "cap", "approach_only")
    ).create_with_runtime_services(**kwargs)
    result = lowerer.lower(
        SimpleNamespace(arguments={"object": "button"}),
        context=context,
        bound=bound,
        option_template=MoveEndEffectorOptions(),
    )
    assert result.skill_options is None
    assert torch.allclose(result.goal.xpos[:, :3, 3], torch.tensor([[0.0, 0.0, 0.08]]))
    context.task.get_held_object = lambda _: "held"
    with pytest.raises(ValueError, match="empty hand"):
        lowerer.lower(
            SimpleNamespace(arguments={"object": "button"}),
            context=context,
            bound=bound,
            option_template=MoveEndEffectorOptions(),
        )


def test_press_goal_uses_native_tip_and_preserves_profile_options(monkeypatch):
    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    context.batch_size = 1
    cfg = SimpleNamespace(
        motion_part="arm", hand_part="gripper", open_qpos=(0.8,), closed_qpos=(0.0,)
    )
    kwargs["engine"].hand_geometry_cfgs = {"gripper": cfg}
    motion = bound.binding.action_binding.endpoint("primary", "motion")
    hand = SimpleNamespace(
        require_target=lambda _: JointPositionTarget("gripper", (2,)),
        joint_positions=lambda command, **kw: torch.tensor(
            [[0.8 if command == subject.OPEN_COMMAND else 0.0]]
        ),
    )
    bound.binding.action_binding.endpoint = lambda role, kind: (
        motion if kind == "motion" else hand
    )
    calls = []

    def tip(robot, calibration, qpos, *, finger_link_names):
        calls.append((robot, calibration, finger_link_names))
        return 0.04

    monkeypatch.setattr(subject, "closed_native_hand_tip_offset", tip)
    lowerer = subject.PressFactory(
        subject.PressCallDeclaration("operation", "cap")
    ).create_with_runtime_services(**kwargs)
    options = PressOptions(press_distance=0.02)
    result = lowerer.lower(
        SimpleNamespace(arguments={"object": "button"}),
        context=context,
        bound=bound,
        option_template=options,
    )
    assert result.skill_options is None
    assert result.goal.semantics.geometry == {}
    assert result.goal.semantics.affordance.press_position == pytest.approx(
        (0.0, 0.0, 0.04)
    )
    assert calls == [(kwargs["robot"], cfg, ("finger",))]
    assert result.goal.target_pose.entity_id == "cap"
    with pytest.raises(ValueError, match="command distance"):
        lowerer.lower(
            SimpleNamespace(arguments={"object": "button"}),
            context=context,
            bound=bound,
            option_template=PressOptions(press_distance=0.03),
        )


def test_operation_suffix_is_type_declared_and_preserves_descriptor(monkeypatch):
    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    base = subject.PressPrepareFactory(subject.PressCallDeclaration("operation", "cap"))
    factory = subject.specialize_control_factory(base, "operation")
    assert type(factory).call_id == "gen_sim.press_prepare.operation"
    assert (
        declare_registered_semantic_lowerer_factory(factory).call_id
        == type(factory).call_id
    )
    lowerer = factory.create_with_runtime_services(**kwargs)
    assert type(lowerer).call_id == type(factory).call_id
    assert type(lowerer).target_descriptor == base.target_descriptor
    assert type(subject.specialize_control_factory(base, "operation")) is type(factory)
    with pytest.raises(ValueError, match="match its declared operation"):
        subject.specialize_control_factory(base, "other")


@pytest.mark.parametrize(
    "variant, expected_z, rotation",
    [
        ("source", 0.168, True),
        ("centroid", 0.22, True),
        ("closed_tip", 0.158, True),
        ("no_twist", 0.168, False),
    ],
)
def test_twist_grounding_preserves_typed_target_and_source_contact(
    monkeypatch, variant, expected_z, rotation
):
    from dataclasses import asdict
    from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding

    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    binding = RevoluteControlBinding(
        "button",
        "joint",
        "cap",
        "base",
        "a" * 64,
        1.0,
        (0.0, 0.0, 1.0),
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.2),
        0.04,
        0.03,
        (-1.0, 1.0),
        1,
        0.4,
        calibration_sha256="b" * 64,
    )
    declaration = subject.TwistCallDeclaration("operation", "cap", variant)
    operation.family, operation.binding = "twist", binding
    operation.route_payload = asdict(declaration)
    context.batch_size = 1
    robot = SimpleNamespace(
        get_link_physical_attr=lambda names: [
            SimpleNamespace(rest_offset=0.002),
            SimpleNamespace(rest_offset=0.003),
        ]
    )
    operation.robot = kwargs["robot"] = kwargs["engine"].robot = robot
    operation.articulation.get_link_physical_attr = lambda name: [
        SimpleNamespace(rest_offset=0.001)
    ]
    monkeypatch.setattr(subject, "inspect_revolute_control", lambda *a, **k: binding)
    cfg = SimpleNamespace(
        motion_part="arm",
        hand_part="gripper",
        open_qpos=(0.8,),
        closed_qpos=(0.0,),
        pad_link_names=("left_pad", "right_pad"),
    )
    kwargs["engine"].hand_geometry_cfgs = {"gripper": cfg}
    motion = bound.binding.action_binding.endpoint("primary", "motion")
    hand = SimpleNamespace(
        require_target=lambda _: JointPositionTarget("gripper", (2,)),
        joint_positions=lambda command, **kw: torch.tensor(
            [[0.8 if command == subject.OPEN_COMMAND else 0.0]]
        ),
    )
    bound.binding.action_binding.endpoint = lambda role, kind: (
        motion if kind == "motion" else hand
    )
    widths = []

    class Calibration:
        def __init__(self, *args, **kwargs):
            pass

        def fit_opening(self, width):
            widths.append(width)
            return 0.04, torch.tensor([[0.25]])

        def closed_tip_offset(self):
            return 0.05

    monkeypatch.setattr(subject, "HandCollisionTipCalibration", Calibration)
    startup = []
    operation.start_startup = lambda: startup.append(True)
    prepare = subject.TwistPrepareFactory(declaration).create_with_runtime_services(
        **kwargs
    )
    prepared = prepare.lower(
        SimpleNamespace(arguments={"object": "button", "setting": 1}),
        context=context,
        bound=bound,
        option_template=MoveJointsOptions(),
    )
    assert startup == [True]
    assert torch.equal(prepared.goal.target, torch.tensor([[0.2, 0.4]]))
    lowerer = subject.TwistFactory(declaration).create_with_runtime_services(**kwargs)
    options = subject.TwistOptions(
        rotation_enabled=rotation,
        contact_bite_depth=0.008,
        max_twist_angle=declaration.chunk_angle,
        contact_settle_steps=5,
    )
    call = SimpleNamespace(arguments={"object": "button", "setting": 1})
    result = lowerer.lower(call, context=context, bound=bound, option_template=options)
    assert result.skill_options is None and result.goal.semantics.geometry == {}
    assert result.goal.contact_reference == pytest.approx((0.0, 0.0, expected_z))
    assert result.goal.joint_target.position == 0.4
    assert result.goal.joint_target.articulation_id == "button"
    assert widths == pytest.approx([0.038])
    assert torch.equal(
        result.control_overrides.endpoints["primary"]["grasp"][
            subject.GRASP_COMMAND
        ].positions,
        torch.tensor([[0.25]]),
    )
    with pytest.raises(ValueError, match="explicit bounded declaration"):
        lowerer.lower(
            call, context=context, bound=bound, option_template=subject.TwistOptions()
        )


def test_native_control_assembly_uses_exact_sensor_and_fresh_operation(monkeypatch):
    from unittest.mock import MagicMock
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensor
    from embodichain.lab.sim.scene import press_inputs
    from embodichain.lab.task_program.integrations.simulation import control_assembly

    kwargs, operation, active, context, bound = _assembly(monkeypatch)
    b = operation.binding
    robot = SimpleNamespace(uid="robot", num_instances=1, link_names=("finger",))
    operation.articulation.num_instances = 1
    sensor = MagicMock(spec=ContactSensor)
    sensor.uid, sensor.num_instances = "contact", 1
    sensor.cfg = SimpleNamespace(
        max_contacts_per_env=512,
        filter_need_both_actor=True,
        rigid_uid_list=[],
        articulation_cfg_list=[
            SimpleNamespace(articulation_uid=uid, link_name_list=[])
            for uid in ("robot", "button")
        ],
    )
    sensor.get_actor_ids.side_effect = lambda uid, names=None: (
        torch.tensor([[1, 2]]) if uid == "button" else torch.tensor([[3]])
    )
    simulation = SimpleNamespace(
        num_envs=1,
        is_default_backend=True,
        get_robot=lambda uid: robot,
        get_articulation=lambda uid: operation.articulation,
        get_sensor=lambda uid: sensor,
        sim_config=SimpleNamespace(physics_dt=0.004),
        simulation_time=0.0,
        spawn_result=SimpleNamespace(topology_revision=7),
    )
    declaration = control_assembly.ControlOperationDeclaration(
        subject.PressCallDeclaration("operation", "cap"),
        b,
        "contact",
        "gripper",
        ("finger",),
        "right",
        moving_mass=0.1,
    )
    monkeypatch.setattr(press_inputs, "inspect_press", lambda *args, **kw: b)
    created = []

    class Runtime:
        def __init__(self, *args, **kw):
            self.args, self.kw, self.closed = args, kw, False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(control_assembly, "ControlOperationRuntime", Runtime)
    assembly_kwargs = dict(
        simulation=simulation,
        robot=robot,
        sensor_suite={"contact": sensor},
        hand_geometry_cfgs={
            "gripper": SimpleNamespace(hand_part="gripper", motion_part="arm")
        },
        program_fingerprint="program",
        episode_id="episode",
        invocation_ids=(),
        control_dt=0.04,
        validate_scope=lambda: None,
        validate_observation=lambda context: None,
    )
    simulation.num_envs = 2
    with pytest.raises(ValueError, match="managed B1 robot"):
        control_assembly.assemble_control_operation(declaration, **assembly_kwargs)
    simulation.num_envs = 1
    assert not created
    first = control_assembly.assemble_control_operation(declaration, **assembly_kwargs)
    second = control_assembly.assemble_control_operation(declaration, **assembly_kwargs)
    assert first.operation is not second.operation
    assert first.operation.kw["observer_id"] != second.operation.kw["observer_id"]
    assert first.operation.args[4].sensor is sensor
    stamp = first.operation.args[4].source_stamp()
    assert stamp.topology_revision == 7
    simulation.spawn_result.topology_revision = 8
    changed = first.operation.args[4].source_stamp()
    assert changed.topology_revision == 8
    assert "Contact actor topology changed; rebind the sampler." in changed.errors
    simulation.spawn_result.topology_revision = 7
    assert first.operation.args[5].hand == (3,)
    assert first.operation.kw["program_fingerprint"] == "program"
    first.close()
    assert first.operation.closed and not second.operation.closed
    sensor.cfg.filter_need_both_actor = False
    with pytest.raises(ValueError, match="source query"):
        control_assembly.assemble_control_operation(declaration, **assembly_kwargs)
    sensor.cfg.filter_need_both_actor = True
    with pytest.raises(ValueError, match="ordinary substep-suite"):
        control_assembly.assemble_control_operation(
            declaration, **{**assembly_kwargs, "sensor_suite": {}}
        )


@pytest.mark.parametrize("via_environment", [False, True])
def test_control_bridge_uses_one_clock_actual_services_and_episode_guard(
    monkeypatch, via_environment
):
    from dataclasses import asdict
    from unittest.mock import MagicMock
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensor
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from embodichain.lab.sim.scene import press_inputs
    from embodichain.lab.task_program.integrations.configured import (
        _decode_configured_task_program_integration,
    )
    from embodichain.lab.task_program.integrations.simulation.environment import (
        SimulationTaskProgramFactory,
    )
    from embodichain.lab.task_program.language.decoder import decode_task_program
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    _, source, _, _, _ = _assembly(monkeypatch)
    binding = asdict(source.binding)
    binding.pop("source_path")
    hand = dict(
        kind="sampled_parallel_jaw",
        motion_part="arm",
        hand_part="gripper",
        command_preset="hand",
        hand_root_link="hand",
        pad_link_names=["finger_a", "finger_b"],
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=10,
    )
    payload = {
        "scene": {
            "registry_id": "control_scene",
            "articulations": [{"entity_id": "button", "simulation_uid": "button"}],
            "links": [
                {
                    "entity_id": "cap",
                    "articulation_id": "button",
                    "native_link_name": "cap",
                }
            ],
        },
        "robot_profile": {
            "profile_id": "control_robot",
            "resources": [
                {
                    "resource_id": "manipulator",
                    "endpoints": [
                        {
                            "endpoint_id": "motion",
                            "control_part": "arm",
                            "capabilities": [
                                "motion.joint_position",
                                "motion.cartesian_pose",
                                "kinematics.forward",
                                "kinematics.batch_inverse",
                            ],
                        },
                        {
                            "endpoint_id": "grasp",
                            "control_part": "gripper",
                            "capabilities": ["interaction.grasp"],
                            "command_preset": "hand",
                        },
                    ],
                }
            ],
            "command_presets": [
                {
                    "preset_id": "hand",
                    "control_part": "gripper",
                    "commands": {"open": [0.8], "grasp": [0.0]},
                }
            ],
            "defaults": {
                key: {"primary": "manipulator"} for key in ("move_joints", "press")
            },
            "presets": [
                {
                    "preset_id": "safe",
                    "action_options": {
                        "gen_sim.press_prepare": {"kind": "move_joints"},
                        "gen_sim.press": {"kind": "press", "press_distance": 0.02},
                    },
                    "effect_assurance": "projected",
                }
            ],
            "default_preset": "safe",
        },
        "runtime_services": {
            "hand_geometries": {"gripper": hand},
            "control_operations": [
                {
                    "kind": "press",
                    "operation_id": "operation",
                    "entity_id": "cap",
                    "binding": binding,
                    "sensor_uid": "contact",
                    "grasp_target_id": "gripper",
                    "finger_link_names": ["finger_a", "finger_b"],
                    "arm_label": "right",
                    "moving_mass": 0.1,
                }
            ],
            "registered_semantic_lowerers": [
                {"kind": kind, "operation_id": "operation"}
                for kind in ("press_prepare", "press")
            ],
            "settle_presets": {
                phase: {
                    "kind": "press_control",
                    "entity": "cap",
                    "operation_id": "operation",
                    "phase": phase,
                }
                for phase in ("ready", "released")
            },
        },
    }
    payload["robot_profile"]["presets"][0].update(
        motion={
            "strategy": "ik_interp",
            "velocity_targets": "auto",
            "sample_count": 40,
            "dynamic_collision_mode": "off",
        },
        tracking={"kind": "timed"},
        recovery={"max_replans": 0, "max_action_retries": 0},
        workflow_recovery={"max_recovery_attempts": 0},
        runner={
            "minimum_cycle_time": 0.0,
            "hold_on_completion": False,
            "hold_during_effect_verification": False,
        },
        effect_monitors={},
    )
    registration = _decode_configured_task_program_integration(payload).registration
    robot = SimpleNamespace(
        uid="robot",
        device=torch.device("cpu"),
        num_instances=1,
        dof=3,
        joint_names=["j0", "j1", "j2"],
        link_names=["finger_a", "finger_b"],
        control_parts={"arm": ("j0", "j1"), "gripper": ("j2",)},
        get_qpos=lambda **kw: torch.tensor([[0.2, 0.4, 0.8]]),
        get_qvel=lambda **kw: torch.zeros(1, 3),
        get_qf=lambda **kw: torch.zeros(1, 3),
        get_joint_ids=lambda name: [0, 1] if name == "arm" else [2],
        get_solver=lambda name: object(),
    )
    art = source.articulation
    art.num_instances, art.link_names = 1, ["cap", "base"]
    art.get_root_pose = lambda **kw: torch.eye(4)[None]
    art.get_local_pose = art.get_root_pose
    art.get_qvel = lambda **kw: torch.zeros(1, 1)
    sensor = MagicMock(spec=ContactSensor)
    sensor.uid, sensor.num_instances = "contact", 1
    sensor.cfg = SimpleNamespace(
        max_contacts_per_env=512,
        filter_need_both_actor=True,
        rigid_uid_list=[],
        articulation_cfg_list=[
            SimpleNamespace(articulation_uid=uid, link_name_list=[])
            for uid in ("robot", "button")
        ],
    )
    sensor.get_actor_ids.side_effect = lambda uid, names=None: (
        torch.tensor([[1, 2]]) if uid == "button" else torch.tensor([[3, 4]])
    )
    hooks = {}
    sensor.add_sampling_observer.side_effect = lambda name, observer: hooks.__setitem__(
        name, observer
    )
    sensor.remove_sampling_observer.side_effect = lambda name, expected: hooks.pop(name)
    simulation = SimpleNamespace(
        num_envs=1,
        is_default_backend=True,
        get_robot=lambda uid: robot,
        get_articulation=lambda uid: art,
        get_rigid_object=lambda uid: None,
        get_sensor=lambda uid: sensor,
        sim_config=SimpleNamespace(physics_dt=0.004),
        simulation_time=0.0,
    )
    monkeypatch.setattr(press_inputs, "inspect_press", lambda *a, **kw: source.binding)
    generator = MagicMock(spec=MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(planner_type="test"))
    generator.collision_world_info = None
    generator.dynamic_collision_entity_ids = generator.collision_world_entity_ids = ()
    generator.supports_dynamic_collision_world = False
    generator.collision_world_batch_mode = None
    cfg = RobotHandGeometryCfg(
        motion_part="arm",
        hand_part="gripper",
        hand_root_link="hand",
        pad_link_names=("finger_a", "finger_b"),
        open_qpos=(0.8,),
        closed_qpos=(0.0,),
        sample_count=65,
        body_max_edge=0.01,
        pad_max_edge=0.004,
        max_subdivisions=10,
    )
    captures = []
    providers = []

    class Factory(SimulationTaskProgramFactory):
        def create_control_bridge_services(self, program, clock):
            services = super().create_control_bridge_services(program, clock)
            captures.append((clock, services))
            return services

        def create_planning_observation_provider(self, **kwargs):
            provider = super().create_planning_observation_provider(**kwargs)
            providers.append(provider)
            return provider

    episode = ["actual_episode_4"]
    environment = None
    common = dict(
        motion_generator_factory=lambda: generator,
        hand_geometry_cfgs={"gripper": cfg},
    )
    if via_environment:
        from embodichain.lab.gym.envs.episode_context import EpisodeContext

        episode_context = EpisodeContext(1)
        environment = SimpleNamespace(
            sim=simulation,
            robot=robot,
            step_dt=0.04,
            sensors={"contact": sensor},
            episode_context=episode_context,
        )
        factory = Factory.from_environment(
            environment, registration=registration, **common
        )
        episode[0] = "0"
    else:
        factory = Factory(
            simulation,
            robot,
            registration,
            step_dt=0.04,
            sensor_suite={"contact": sensor},
            episode_identity=lambda: episode[0],
            **common,
        )
    adapter = factory.create_adapter()
    program = decode_task_program(
        {
            "program_id": "controls",
            "targets": {},
            "integration": {
                "scene_registry": "control_scene",
                "robot_profile": "control_robot",
                "runtime_preset": "safe",
            },
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": phase,
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "registered",
                                "call_id": call_id,
                                "arguments": {"object": "button"},
                            },
                        },
                        "post": [
                            {"kind": "wait_stable", "entity": "cap", "preset": phase}
                        ],
                    }
                    for phase, call_id in (
                        ("ready", "gen_sim.press_prepare"),
                        ("released", "gen_sim.press"),
                    )
                ],
            },
        }
    )
    compiled = adapter.compile(program)
    with pytest.raises(ValueError, match="per-bridge"):
        factory.segment_policy_port.bind_clock(EnvironmentStepClock(0.04))
    with pytest.raises(ValueError, match="do not support expansion"):
        adapter.create_bridge(compiled, expansion_profile=object())
    assert not hooks
    if via_environment:
        with pytest.raises(ValueError, match="initialized actual episode"):
            adapter.create_bridge(compiled)
        episode_context._begin_episode([0])
        episode_context._finish_initialization([0])
    scope_open = [True]

    def validate_scope():
        if not scope_open[0]:
            raise RuntimeError("Actual host scope replaced")

    first = adapter.create_bridge(compiled, runtime_scope_guard=validate_scope)
    second = adapter.create_bridge(compiled)
    assert len(hooks) == 2
    assert first.clock is captures[0][0] and second.clock is captures[1][0]
    assert first.post_policy_port is first.validator_port is captures[0][1].policy_port
    assert (
        second.post_policy_port is second.validator_port is captures[1][1].policy_port
    )
    assert first.post_policy_port is not factory.segment_policy_port
    first_op = captures[0][1].runtime_context.control_operation("operation")
    assert first_op is not captures[1][1].runtime_context.control_operation("operation")
    assert first_op.observation_scope["episode_id"] == episode[0]
    first.close_runtime_resources()
    assert len(hooks) == 1
    # Final collectors may still use the actual port after control hooks close.
    context = first.post_policy_port.runtime_service_context(
        program_fingerprint=captures[0][1].runtime_context.program_fingerprint,
        episode_id=episode[0],
    )
    context.validate_scope()
    from embodichain.lab.sim.atomic_actions import TaskState

    final_observation = providers[0].observe(TaskState(batch_size=1, device="cpu"))
    assert torch.equal(final_observation.robot.qpos, robot.get_qpos())
    with pytest.raises(RuntimeError, match="closed"):
        context.control_operation("operation")
    scope_open[0] = False
    with pytest.raises(RuntimeError, match="Actual host scope replaced"):
        providers[0].current_qpos(torch.tensor([0]))
    scope_open[0] = True
    if via_environment:
        environment.episode_context = EpisodeContext(1)
        with pytest.raises(RuntimeError, match="episode owner was replaced"):
            providers[1].current_qpos(torch.tensor([0]))
        environment.episode_context = episode_context
        episode_context._begin_episode([0])
        episode_context._finish_initialization([0])
    episode[0] = "actual_episode_5"
    with pytest.raises(RuntimeError, match="replaced episode"):
        captures[1][1].runtime_context.control_operation("operation")
    from embodichain.lab.sim.atomic_actions import TaskState

    with pytest.raises(RuntimeError, match="replaced episode"):
        providers[1].observe(TaskState(batch_size=1, device="cpu"))
    second.close_runtime_resources()
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        providers[0].current_qpos(torch.tensor([0]))
    assert not hooks
