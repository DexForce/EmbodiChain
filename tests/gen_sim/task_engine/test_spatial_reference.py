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
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.configured import decode_task_lowerer
from embodichain.gen_sim.task_engine._task_program.spatial_reference import (
    SpatialReference,
    measure_relation,
)
from embodichain.gen_sim.task_engine._task_program.stability import (
    StabilityConstraint,
    TaskStabilityPort,
)
from embodichain.lab.sim.atomic_actions import PlaceOptions
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
    SimulationRigidObjectBinding,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
)
from embodichain.lab.task_program.language.schema import WaitStablePostCfg
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SceneObjectRef,
)


def test_live_link_is_shared_by_lowering_lookahead_and_stability():
    poses = {name: torch.eye(4).repeat(2, 1, 1) for name in ("object", "root", "link")}
    poses["root"][:, 0, 3] = 10.0
    poses["object"][:, 1, 3] = -0.2
    obj = SimpleNamespace(get_local_pose=lambda **kw: poses["object"].clone())
    art = SimpleNamespace(
        get_local_pose=lambda **kw: poses["root"].clone(),
        get_link_pose=lambda *a, **kw: poses["link"].clone(),
        get_qpos=lambda **kw: torch.zeros(2, 1),
        joint_names=("slide",),
        link_names=("drawer",),
    )
    sim = SimpleNamespace(
        get_rigid_object=lambda uid: obj, get_articulation=lambda uid: art
    )
    scene = SimulationSceneBinding(
        registry_id="links",
        rigid_objects=(SimulationRigidObjectBinding("cup", "cup"),),
        articulations=(SimulationArticulationBinding("cabinet", "cabinet"),),
        links=(
            SimulationArticulationLinkBinding("cabinet__drawer", "cabinet", "drawer"),
        ),
    )
    registry = scene.build(sim)
    robot = SimpleNamespace(
        compute_fk=lambda **kw: poses["object"].clone(),
        get_qpos=lambda **kw: torch.zeros(2, 1),
    )
    factory = decode_task_lowerer(
        {
            "kind": "place_relative",
            "routes": [
                {
                    "object_id": "cup",
                    "reference_entity_id": "cabinet",
                    "relation": "left_of",
                    "world_displacement": [0.0, -0.2, 0.0],
                    "reference_link_id": "cabinet__drawer",
                    "reference_part": "part_fixture",
                }
            ],
        },
        path="link",
    )
    lowerer = factory.create(
        simulation=sim,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    call = RegisteredSemanticCall(
        call_id="simulation.place_relative",
        arguments={
            "object": "cup",
            "reference": "cabinet",
            "relation": "left_of",
            "reference_part": "part_fixture",
        },
    )
    provider = registry.make_scene_provider(batch_size=2)
    env_ids = torch.tensor([0, 1])
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="cup"),
        object_to_eef=torch.eye(4).repeat(2, 1, 1),
    )
    options = PlaceOptions(preserve_current_object_orientation=True)
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            task_state_key="arm",
                            runtime_target=SimpleNamespace(
                                joint_ids=(0,), control_part="arm"
                            ),
                        )
                    }
                )
            }
        ),
        preset=SimpleNamespace(action_option_template=lambda _: options),
    )
    context = SimpleNamespace(
        scene=provider.snapshot(timestamp=0.0, env_ids=env_ids),
        task=SimpleNamespace(get_held_object=lambda _: held),
        robot=SimpleNamespace(qpos=torch.zeros(2, 1)),
        batch_size=2,
        env_ids=env_ids,
    )
    target = lowerer.lower(
        call, context=context, bound=bound, option_template=options
    ).goal.xpos
    lookahead = lowerer.pick_lookahead_targets(
        call, picked_object=SceneObjectRef("cup"), bound=bound, previous_target=None
    )[0].pose
    assert target.entity_id == lookahead.entity_id == "cabinet__drawer"
    poses["link"][1, 0, 3] = 0.15
    context.scene = provider.snapshot(timestamp=1.0, env_ids=env_ids)
    resolved = resolve_pose_goal(target, context, name="release")
    torch.testing.assert_close(resolved[:, 0, 3], torch.tensor([0.0, 0.15]))
    poses["object"][1, 0, 3] = 0.15
    cfg = StabilityConstraint(
        kind="placement",
        entity="cup",
        reference="cabinet__drawer",
        displacement=(0.0, -0.2, 0.0),
        duration=0.04,
        timeout=0.08,
    )
    policy = CompiledPostPolicy(
        cfg=WaitStablePostCfg(entity="cup", preset="gen_sim.link.stable"),
        entity=SceneObjectRef("cup"),
        source_path=("program", "items", 0, "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    port = TaskStabilityPort(
        Mock(), sim, robot, scene, {"gen_sim.link.stable": cfg}, step_dt=0.04
    )
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, True]
    measurement = port.post_policy_metadata(policy, segment=segment)["measurements"]
    assert measurement["reference"] == target.entity_id
    assert measurement["reference_pose"][1][0][3] == pytest.approx(0.15)
    poses["link"][1, 0, 3] += 0.1
    list(
        port.actions(
            policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
        )
    )
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    with pytest.raises(ValueError, match="reference part"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=call.call_id,
                arguments={**dict(call.arguments), "reference_part": "part_other"},
            ),
            context=context,
            bound=bound,
            option_template=options,
        )
    from embodichain.lab.task_program.semantics import SceneArticulationRef

    foreign = SimpleNamespace(
        resolve=registry.resolve,
        lookup=lambda *a, **kw: SimpleNamespace(parent=SceneArticulationRef("other")),
    )
    with pytest.raises(ValueError, match="different articulation"):
        factory.create(
            simulation=sim,
            robot=robot,
            scene_registry=foreign,
            engine=SimpleNamespace(robot=robot),
        )


@pytest.mark.parametrize(
    "relation",
    ["left_of", "right_of", "front_of", "behind", "front_left_of", "back_right_of"],
)
def test_side_envelopes_use_world_axes_not_link_rotation(relation):
    obj = torch.eye(4).repeat(2, 1, 1)
    reference = obj.clone()
    reference[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    directions = {
        "left_of": (0, -1),
        "right_of": (0, 1),
        "front_of": (-1, 0),
        "behind": (1, 0),
        "front_left_of": (-1, -1),
        "back_right_of": (1, 1),
    }
    obj[0, :2, 3] = torch.tensor(directions[relation]) * 0.2
    vertices = torch.tensor([[[-0.01, -0.01, 0], [0.01, 0.01, 0.02]]]).expand(2, -1, -1)
    accepted, _ = measure_relation(
        vertices, obj, vertices[0].numpy(), reference, relation
    )
    assert accepted == [True, False]


@pytest.mark.parametrize("recorded", [False, True])
def test_side_taskspec_observes_bound_link_and_tabletop(
    tmp_path, monkeypatch, recorded
):
    from .test_task_spec_runtime import _placement_evidence, _observe_placement
    from embodichain.gen_sim.task_engine.contracts import canonical_hash
    from embodichain.utils.utility import save_config
    from dataclasses import asdict
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _placement_record,
        _segments,
    )
    from embodichain.gen_sim.task_engine._task_program.services import (
        _RelativePlaceRoute,
    )
    from embodichain.gen_sim.task_engine._task_program import spatial_reference

    plan, contract, runtime, program, constraints, env, deployment = (
        _placement_evidence(tmp_path, relation="left_of")
    )
    registration = deployment.integration.registration
    obj = env.sim.get_rigid_object("cube")
    table = env.sim.get_rigid_object("region")
    obj.get_local_pose()[:, 1, 3] = -0.10
    env._traj_buffer["states"]["rigid_objects"]["cube"]["pose"][..., 1] = -0.10
    env._traj_buffer["states"]["rigid_objects"]["table"] = env._traj_buffer["states"][
        "rigid_objects"
    ]["region"]
    links = torch.eye(4)[None]
    art = SimpleNamespace(
        get_link_pose=lambda *a, **kw: links.clone(),
        cfg=SimpleNamespace(
            fpath="fixture",
            body_scale=[1.0] * 3,
            root_props=SimpleNamespace(fixed_base=True),
            init_local_pose=None,
            init_pos=[0.0] * 3,
            init_rot=[0.0] * 3,
        ),
    )
    env.sim.get_rigid_object = {"cube": obj, "table": table}.get
    env.sim.get_articulation = lambda _: art
    scene = registration.scene_binding
    scene.rigid_objects = (
        SimpleNamespace(entity_id="cube", simulation_uid="cube"),
        SimpleNamespace(entity_id="table", simulation_uid="table"),
    )
    scene.articulations = (
        SimpleNamespace(entity_id="region", simulation_uid="region"),
    )
    scene.links = (
        SimpleNamespace(
            entity_id="region__body", articulation_id="region", native_link_name="body"
        ),
    )
    registration.registered_semantic_lowerer_factories = (
        SimpleNamespace(
            call_id="simulation.place_relative",
            routes=(
                _RelativePlaceRoute(
                    "cube",
                    "region",
                    "left_of",
                    (0.0, -0.10, 0.0),
                    reference_link_id="region__body",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        spatial_reference,
        "resolve_reference",
        lambda *a: SpatialReference(
            "region",
            None,
            "body",
            np.array([[-0.02, -0.02, 0], [0.02, 0.02, 0.02]]),
            np.eye(4),
        ),
    )
    segments = _segments(runtime)
    boundary = next(reversed(segments.values()))
    place = next(
        n
        for n in plan.graph["nodes"]
        if n["call"].get("call_id") == "simulation.place_relative"
    )
    preset = f"gen_sim.{place['id']}.stable"
    cfg = StabilityConstraint(
        kind="placement",
        entity="cube",
        reference="region__body",
        displacement=(0.0, -0.10, 0.0),
    )
    deployment.integration.adapter_factory = SimpleNamespace(
        constraints=((preset, cfg),)
    )
    constraints["presets"][preset] = asdict(cfg)
    program["program"]["items"][2]["post"] = [
        {"kind": "wait_stable", "entity": "cube", "preset": preset}
    ]
    contract["program_digest"] = canonical_hash(program)
    contract["constraints_digest"] = canonical_hash(constraints)
    save_config(tmp_path / "task_program/constraints.json", constraints)
    prefix = ["program", "items", 2]
    boundary["metadata"].update(
        program_segment_source_path=prefix,
        post_policies=[
            {
                "kind": "wait_stable",
                "policy_index": 0,
                "source_path": [*prefix, "post", 0],
                "result_mask": [True],
                "result": {
                    "preset": preset,
                    "kind": "placement",
                    "required_seconds": cfg.duration,
                    "stable_seconds": [cfg.duration],
                    "accepted_mask": [True],
                    "measurements": {
                        "entity": "cube",
                        "reference": "region__body",
                        "object_pose": obj.get_local_pose().tolist(),
                        "reference_pose": links.tolist(),
                    },
                },
            }
        ],
    )
    member = next(iter(contract["coverage"]["lineage"].values()))[0]
    predicate = plan.task_spec.goal[0]
    result = _placement_record(
        env,
        deployment,
        plan.graph,
        contract,
        predicate,
        member,
        segments,
        boundary,
        1,
        recorded=recorded,
    )
    assert result["accepted"] == [True]
    assert result["measurements"]["reference_link_id"] == "region__body"
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _placement_measurement_matches,
    )

    assert _placement_measurement_matches(
        result, predicate, contract["coverage"]["roles"], 1
    )
    evaluation = _observe_placement(
        tmp_path, (plan, contract, runtime, program, constraints, env, deployment)
    )
    assert evaluation["certificate_ready"] is True
    assert evaluation["milestones"][0]["status"] == "satisfied"
    links[:, 1, 3] = -0.25
    result = _placement_record(
        env,
        deployment,
        plan.graph,
        contract,
        predicate,
        member,
        segments,
        boundary,
        1,
        recorded=recorded,
    )
    assert result["accepted"] == ([True] if recorded else [False])
    assert _placement_measurement_matches(
        result, predicate, contract["coverage"]["roles"], 1
    )
    if recorded:
        boundary["metadata"]["post_policies"][0]["result"]["measurements"][
            "reference"
        ] = "other__body"
        with pytest.raises(ValueError, match="exact accepted link"):
            _placement_record(
                env,
                deployment,
                plan.graph,
                contract,
                predicate,
                member,
                segments,
                boundary,
                1,
                recorded=True,
            )
