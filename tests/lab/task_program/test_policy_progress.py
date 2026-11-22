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

"""Policy progress reuses qualified rules without an expert executor."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from embodichain.lab.evaluation import (
    EvaluationSpec,
    EvaluationStatus,
    PredicateDefinition,
    PredicateMeasurement,
    PredicateSpec,
    builtin_predicates,
)
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers import ConditionCfg, RewardCfg, RLTaskCfg
from embodichain.lab.gym.envs.managers.conditions import program_stage_condition
from embodichain.lab.gym.envs.managers.rewards import program_stage_reward
from embodichain.lab.gym.envs.managers.reward_manager import RewardManager
from embodichain.lab.gym.envs.managers.termination_manager import TerminationManager
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationFrame,
    ObservationSample,
)
from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.semantics.integration import SceneManifest
from embodichain.lab.task_program.integrations.progress import TaskProgramProgress
from test_program_compiler import _scene_registry, _integration


def _setup(*, repeat=1, ids=(0, 1), count=2, gates=False, stage_names=None):
    calls = []
    registry = builtin_predicates()

    def measure(arguments, params):
        calls.append(1)
        value = arguments["height"]
        return PredicateMeasurement(
            value >= params["minimum"], torch.isfinite(value), {"height": value}
        )

    registry.register(
        "measured_height", PredicateDefinition(1, ("height",), dict, measure)
    )
    graph = registry.compile(
        EvaluationSpec(
            {
                "lifted": PredicateSpec(
                    "measured_height", {"height": "height"}, {"minimum": 0.1}
                )
            },
            ("lifted",),
        )
    )
    conditions = {"lifted": graph}
    segment = {
        "kind": "segment",
        "name": "lift",
        "instruction": "Lift the cube.",
        "steps": {
            "kind": "invoke",
            "call": {"kind": "pick", "object": "cube", "grasp": "cube_grasp"},
        },
        "validators": [{"kind": "condition", "ref": "lifted"}],
        "contract": {"outcomes": [{"kind": "validator", "ref": 0}]},
    }
    data = {
        "program_id": "policy_lift",
        "integration": _integration().to_dict(),
        "targets": {},
        "program": (
            segment
            if repeat == 1
            else {"kind": "repeat", "count": repeat, "body": segment}
        ),
    }
    if stage_names is not None:
        data["program"] = {
            "kind": "sequence",
            "items": [{**segment, "name": name} for name in stage_names],
        }
    scene, provider = _scene_registry()
    program = TaskProgramCompiler(
        SceneManifest.from_registry(scene), evaluation_conditions=conditions
    ).compile(decode_task_program(data), strict_contracts=True)
    owner = EpisodeContext(count)
    owner._begin_episode(range(count))
    progress = TaskProgramProgress(
        program,
        conditions,
        env_ids=ids,
        episode_ids=tuple(owner.episode_ids[i] for i in ids),
        final_rules=(
            {
                "final": registry.compile(
                    EvaluationSpec(
                        {
                            "final": PredicateSpec(
                                "measured_height",
                                {"height": "height"},
                                {"minimum": 0.3},
                            )
                        },
                        ("final",),
                    )
                )
            }
            if gates
            else None
        ),
        process_rules=(
            {
                "safe": registry.compile(
                    EvaluationSpec(
                        {
                            "safe": PredicateSpec(
                                "measured_height",
                                {"height": "height"},
                                {"minimum": 0.0},
                            )
                        },
                        ("safe",),
                    )
                )
            }
            if gates
            else None
        ),
    )
    progress._program_for_test = program
    env = SimpleNamespace(
        num_envs=len(ids), device=torch.device("cpu"), task_program_progress=progress
    )
    env.sim = SimpleNamespace()
    return env, owner, provider, calls, graph


def _frame(owner, values, revision):
    value = torch.tensor(values, dtype=torch.float32)
    descriptor = ObservationDescriptor(
        "height", "scalar", "arena", "scalar", (), (), value.dtype
    )
    return ObservationFrame(
        None,
        {
            "height": ObservationSample(
                descriptor, value, torch.isfinite(value), (None,) * len(value)
            )
        },
        owner.episode_ids,
        revision,
        episode_owner=owner,
    )


def test_one_criterion_serves_expert_reward_and_termination_without_query_updates():
    env, owner, provider, calls, graph = _setup()
    progress = env.task_program_progress
    reward = RewardManager(
        {
            "bonus": RewardCfg(
                func=program_stage_reward,
                weight=3.0,
                params={
                    "stage": "lift",
                    "mode": "first_achievement",
                    "scope": "occurrence",
                },
            )
        },
        env,
    )
    termination = TerminationManager(
        RLTaskCfg(
            success={
                "done": ConditionCfg(
                    func=program_stage_condition, params={"state": "achieved"}
                )
            }
        ),
        env,
    )
    frame = _frame(owner, [0.2, 0.0], 1)
    from embodichain.lab.evaluation import RuleSession
    from embodichain.lab.evaluation.observations import advance_observation
    from embodichain.lab.task_program.integrations.simulation.policies import (
        SimulationSegmentPolicyPort,
    )

    validator = next(progress._program_for_test.iter_segments()).validators[0]
    expert_session = RuleSession(
        validator.spec, env_ids=(0, 1), episode_ids=owner.episode_ids, scope="expert"
    )
    advance_observation(expert_session, frame, step=0, episode_owner=owner)
    port = object.__new__(SimulationSegmentPolicyPort)
    port._evaluation_active = torch.ones(2, dtype=torch.bool)
    port._env_ids = torch.arange(2)
    port._evaluation_specs = {"lifted": validator.spec}
    port._evaluation_sessions = {"lifted": expert_session}
    port._validator_results = {}
    expert = port._validate_condition(validator)
    progress.advance(frame, step=0, episode_owner=owner)
    termination.compute()
    before = len(calls)
    for _ in range(3):
        total, _ = reward.compute({}, torch.zeros(2, 1), {})
        assert total.tolist() == [3.0, 0.0]
        assert termination.task_state()[0].tolist() == expert.tolist()
        assert termination.terminated.tolist() == [True, False]
        progress.snapshot()["achieved"].zero_()
        progress.advance(frame, step=0, episode_owner=owner)
    assert len(calls) == before
    assert provider.calls == 0
    assert progress.snapshot()["achieved"].tolist() == [[True], [False]]


def test_revocation_reachievement_and_explicit_bonus_scopes():
    env, owner, _, calls, _ = _setup(repeat=2)
    p = env.task_program_progress
    for step, heights in enumerate(([0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [0.2, 0.2])):
        p.advance(_frame(owner, heights, step + 1), step=step, episode_owner=owner)
        if step == 0:
            assert p.reward(
                "lift", mode="first_achievement", scope="episode"
            ).tolist() == [1.0, 0.0]
            assert p.snapshot()["achieved"].tolist() == [[True, False], [False, False]]
        elif step == 1:
            assert p.reward("lift", mode="rising").tolist() == [0.0, 1.0]
        elif step == 2:
            assert p.reward(
                "lift", mode="first_achievement", scope="occurrence"
            ).tolist() == [1.0, 1.0]
            assert p.reward(
                "lift", mode="first_achievement", scope="episode"
            ).tolist() == [0.0, 0.0]
            assert p.reward("lift", mode="rising").tolist() == [2.0, 1.0]
        else:
            assert p.reward(
                "lift", mode="first_achievement", scope="occurrence"
            ).tolist() == [0.0, 0.0]
    # Repeated occurrences share the exact predicate node in each frame.
    assert len(calls) == 4
    with pytest.raises(ValueError, match="explicit"):
        p.reward("lift", mode="first_achievement")


@pytest.mark.parametrize("damage", ["missing", "stale", "foreign", "episode", "nan"])
def test_invalid_evidence_never_establishes_achievement(damage):
    env, owner, _, _, _ = _setup()
    p = env.task_program_progress
    p.advance(_frame(owner, [0.0, 0.0], 1), step=0, episode_owner=owner)
    frame = _frame(owner, [0.2, 0.2], 2)
    if damage == "missing":
        frame = replace(frame, fields={})
    elif damage == "stale":
        frame = replace(frame, revision=1)
    elif damage == "foreign":
        frame = replace(frame, episode_owner=EpisodeContext(2))
    elif damage == "episode":
        frame = replace(frame, episode_ids=(99, 99))
    else:
        frame = _frame(owner, [float("nan")] * 2, 2)
    p.advance(frame, step=1, episode_owner=owner)
    assert not p.snapshot()["achieved"].any()
    assert p.reward("lift", mode="first_achievement", scope="occurrence").tolist() == [
        0.0,
        0.0,
    ]
    assert p.condition()[1].tolist() == [False, False]


def test_noncontiguous_reset_preserves_peer_episode_and_trigger_state():
    env, owner, _, _, _ = _setup(ids=(7, 3, 5), count=8)
    p = env.task_program_progress
    p.advance(_frame(owner, [0.2] * 8, 1), step=0, episode_owner=owner)
    peer = p.snapshot()["achieved"][2].clone()
    owner._begin_episode([7, 3])
    p.reset([7, 3], [owner.episode_ids[7], owner.episode_ids[3]])
    assert torch.equal(p.snapshot()["achieved"][2], peer)
    p.advance(_frame(owner, [0.2] * 8, 2), step=1, episode_owner=owner)
    assert p.reward("lift", mode="first_achievement", scope="episode").tolist() == [
        1.0,
        1.0,
        0.0,
    ]


def test_foreign_episode_masks_only_its_row():
    env, owner, _, _, _ = _setup()
    frame = replace(
        _frame(owner, [0.2, 0.2], 1), episode_ids=(owner.episode_ids[0], 99)
    )
    env.task_program_progress.advance(frame, step=0, episode_owner=owner)
    assert env.task_program_progress.reward(
        "lift", mode="first_achievement", scope="episode"
    ).tolist() == [1.0, 0.0]
    assert env.task_program_progress.condition()[1].tolist() == [True, False]


def test_metric_reward_zero_when_its_field_disappears_but_other_metric_remains():
    from embodichain.lab.evaluation import BooleanRule

    env, owner, _, _, graph = _setup()
    registry = builtin_predicates()

    def measure(args, params):
        value = args["value"]
        return PredicateMeasurement(
            value > 0.1, torch.isfinite(value), {params["key"]: value}
        )

    registry.register("independent", PredicateDefinition(1, ("value",), dict, measure))
    both = registry.compile(
        EvaluationSpec(
            {
                "height": PredicateSpec(
                    "independent", {"value": "height"}, {"key": "height"}
                ),
                "clearance": PredicateSpec(
                    "independent", {"value": "clearance"}, {"key": "clearance"}
                ),
                "both": BooleanRule("all", ("height", "clearance")),
            },
            ("both",),
        )
    )
    # Rebind the compiled declaration's condition reference; its criteria are
    # independently registered, with separate observation availability.
    p = TaskProgramProgress(
        env.task_program_progress._program_for_test,
        {"lifted": both},
        env_ids=(0, 1),
        episode_ids=owner.episode_ids,
    )
    frame = _frame(owner, [0.2, 0.2], 1)
    sample = frame.fields["height"]
    complete = replace(
        frame,
        fields={
            "height": sample,
            "clearance": replace(
                sample, descriptor=replace(sample.descriptor, entity_uid="clearance")
            ),
        },
    )
    p.advance(complete, step=0, episode_owner=owner)
    assert p.reward(
        "lift", mode="metric", metric="clearance"
    ).tolist() == pytest.approx([0.2, 0.2])
    p.advance(replace(frame, revision=2), step=1, episode_owner=owner)
    assert p.reward("lift", mode="metric", metric="clearance").tolist() == [0.0, 0.0]


def test_measured_rl_success_retains_deployment_final_and_process_gates():
    env, owner, _, _, _ = _setup(gates=True)
    manager = TerminationManager(
        RLTaskCfg(
            success={
                "stage": ConditionCfg(
                    func=program_stage_condition, params={"state": "achieved"}
                )
            }
        ),
        env,
    )
    env.task_program_progress.advance(
        _frame(owner, [0.2, 0.4], 1), step=0, episode_owner=owner
    )
    manager.compute()
    assert manager.task_state()[0].tolist() == [False, True]
    env.task_program_progress.advance(
        _frame(owner, [0.4, -0.1], 2), step=1, episode_owner=owner
    )
    manager.compute()
    success, failure, _ = manager.task_state()
    assert success.tolist() == [True, False]
    assert failure.tolist() == [False, True]


def test_real_host_step_advances_before_consumers_and_seals_before_reset():
    from contextlib import nullcontext
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    env_stub, owner, provider, calls, _ = _setup()
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    env.cfg = SimpleNamespace(
        task_program_execution="policy",
        task_program=None,
        events=None,
        ignore_terminations=False,
        episode_context_in_info=True,
        seed=7,
    )
    env.cfg.rl_task = RLTaskCfg(
        success={
            "stage": ConditionCfg(
                func=program_stage_condition, params={"state": "achieved"}
            )
        }
    )
    env.task_program_progress = env_stub.task_program_progress
    env.episode_context = owner
    env._program_transition = 0
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), render_frame=lambda: nullcontext()
    )
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env.termination_manager = TerminationManager(env.cfg.rl_task, env)
    env._elapsed_steps = torch.zeros(2, dtype=torch.int32)
    env.max_episode_steps = 10
    env._demo_no_auto_reset = True
    env._demo_episode_metadata = [env._new_demo_episode_metadata(i) for i in range(2)]
    env._task_initial_states = [{}, {}]
    env._active_task_program_bridge = None
    env.physical_objective = None
    env.rollout_buffer = None
    env.episode_success_status = torch.zeros(2, dtype=torch.bool)
    env._advance_physics = lambda: None
    env._preprocess_action = lambda action: action
    env._step_action = lambda action: action
    env._postprocess_action = lambda action: action
    env._capture_observation = lambda **kwargs: _frame(
        owner, [0.2, 0.0], env._program_transition + 1
    )
    env.get_reward = lambda **kwargs: env.task_program_progress.reward(
        "lift", mode="first_achievement", scope="occurrence"
    )
    env._extend_reward = lambda rewards, **kwargs: rewards
    env._write_trajectory_step = lambda *args, **kwargs: None
    _, reward, terminated, truncated, info = env.step(torch.zeros(2, 1))
    assert reward.tolist() == [1.0, 0.0]
    assert terminated.tolist() == [True, False]
    assert not truncated.any()
    assert env.termination_manager.snapshot()["steps"].tolist() == [1, 1]
    assert len(calls) == 1 and provider.calls == 0
    sealed = info["terminal_outcomes"][0]
    assert sealed["execution"]["source"] == "rl"
    assert sealed["physical"]["stage_progress"]["achieved"] == [True]
    assert sealed["identity"]["seed"] == 7
    assert sealed["identity"]["config_hash"]
    owner._begin_episode([0])
    env._reset_physical_objective([0])
    assert not env.task_program_progress.snapshot()["achieved"][0].any()
    assert sealed["physical"]["stage_progress"]["achieved"] == [True]
    assert env.termination_manager.snapshot()["steps"].tolist() == [0, 1]


def test_replacing_episode_owner_with_matching_ids_does_not_rebind_evidence():
    env, owner, _, _, _ = _setup()
    p = env.task_program_progress
    p.advance(_frame(owner, [0.0, 0.0], 1), step=0, episode_owner=owner)
    foreign = EpisodeContext(2)
    foreign._begin_episode([0, 1])
    assert foreign.episode_ids == owner.episode_ids
    p.advance(_frame(foreign, [0.2, 0.2], 2), step=1, episode_owner=foreign)
    assert not p.snapshot()["achieved"].any()
    assert not p.condition()[1].any()


def test_multicall_segment_requires_each_measured_call_in_order():
    from embodichain.lab.task_program import CompiledTaskProgram

    env, owner, _, _, graph = _setup()
    original = env.task_program_progress._program_for_test
    stage = next(original.iter_segments())
    call = stage.calls[0]
    stage = replace(
        stage, calls=(call, replace(call, call_index=1, segment_call_index=1))
    )
    program = CompiledTaskProgram._create(
        program_id=original.program_id,
        instruction=original.instruction,
        integration=original.integration,
        segments=(stage,),
    )
    progress = TaskProgramProgress(
        program,
        {"lifted": graph},
        env_ids=(0, 1),
        episode_ids=owner.episode_ids,
        skill_conditions={"call:0": graph, "call:1": graph},
    )
    progress.advance(_frame(owner, [0.2, 0.2], 1), step=0, episode_owner=owner)
    assert progress.snapshot()["achieved"].tolist() == [[True, False], [True, False]]
    progress.advance(_frame(owner, [0.2, 0.2], 2), step=1, episode_owner=owner)
    assert progress.snapshot()["achieved"].tolist() == [[True, True], [True, True]]
    assert progress.indices(stage.segment_id) == (0, 1)
    assert progress.indices("call:1") == (1,)
    with pytest.raises(ValueError, match="every call_index"):
        TaskProgramProgress(
            program,
            {"lifted": graph},
            env_ids=(0, 1),
            episode_ids=owner.episode_ids,
            skill_conditions={"call:0": graph},
        )


def test_progress_rejects_unmatched_skill_conditions_in_direct_python_calls():
    env, owner, _, _, graph = _setup()
    program = env.task_program_progress._program_for_test
    registry = builtin_predicates()

    def unsatisfied(arguments, params):
        height = arguments["height"]
        return PredicateMeasurement(height >= 0.5, torch.isfinite(height), {})

    registry.register(
        "extra_height", PredicateDefinition(1, ("height",), dict, unsatisfied)
    )
    extra = registry.compile(
        EvaluationSpec(
            {"extra": PredicateSpec("extra_height", {"height": "height"}, {})},
            ("extra",),
        )
    )
    progress = TaskProgramProgress(
        program,
        {"lifted": graph},
        env_ids=(0, 1),
        episode_ids=owner.episode_ids,
        skill_conditions={("call", 0): extra},
    )
    progress.advance(_frame(owner, [0.2, 0.2], 1), step=0, episode_owner=owner)
    assert not progress.snapshot()["achieved"].any()
    for selector in (("call", 99), "missing_stage"):
        with pytest.raises(ValueError) as error:
            TaskProgramProgress(
                program,
                {"lifted": graph},
                env_ids=(0, 1),
                episode_ids=owner.episode_ids,
                skill_conditions={selector: extra},
            )
        assert repr(selector) in str(error.value)


def test_progress_identity_binds_additional_skill_acceptance_graph():
    env, owner, _, _, graph = _setup()
    original = env.task_program_progress
    changed = TaskProgramProgress(
        original._program_for_test,
        {"lifted": graph},
        env_ids=(0, 1),
        episode_ids=owner.episode_ids,
        skill_conditions={"lift": graph},
    )
    assert original.compiled.identity != changed.compiled.identity


def test_program_rewards_have_exact_milestone_and_terminal_values_across_partial_reset():
    from embodichain.lab.gym.envs.managers.reward_composition import (
        compose_reward_terms,
        _add_common_rewards,
    )
    from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext

    env, owner, _, _, _ = _setup(repeat=2)
    env.step_dt = 0.04
    env.episode_context = owner
    progress = env.task_program_progress
    specification = {"discount": 0.99, "skills": []}
    task = RLTaskCfg(
        success={
            "all_stages": ConditionCfg(
                func=program_stage_condition, params={"state": "achieved"}
            )
        },
        reward_composition=specification,
    )
    terms = compose_reward_terms(specification, program=True)
    _add_common_rewards(
        SimpleNamespace(rl_task=task, rewards=SimpleNamespace()),
        terms,
        tuple(progress._program_for_test.iter_segments()),
    )
    manager = env.reward_manager = RewardManager(terms, env)
    termination = TerminationManager(task, env)
    manager.initialize_observation(_frame(owner, [0.0, 0.0], 0), [0, 1])
    for step, heights, expected in (
        (0, [0.2, 0.0], [0.995, 0.0]),
        (1, [0.2, 0.2], [10.0, 0.995]),
        (2, [0.0, 0.2], [0.0, 10.0]),
    ):
        if step == 2:
            owner._begin_episode([0])
            progress.reset([0], [owner.episode_ids[0]])
            manager.reset([0])
            termination.reset([0])
            manager.initialize_observation(_frame(owner, [0.0, 0.2], 3), [0])
        frame = _frame(owner, heights, step + 4)
        progress.advance(frame, step=step, episode_owner=owner)
        termination.compute()
        info = {"task_outcome": termination.snapshot()}
        manager.transition_context = RewardTransitionContext(
            frame, termination.terminated
        )
        value, _ = manager.compute({}, torch.zeros(2, 1), info)
        torch.testing.assert_close(value, torch.tensor(expected), atol=1e-6, rtol=0)
        again, _ = manager.compute({}, torch.zeros(2, 1), info)
        torch.testing.assert_close(again, value)


@pytest.mark.parametrize(
    "composition,expected", [(False, (0.0, 7.0)), (True, (0.995, 7.49))]
)
@pytest.mark.parametrize("collision", ["call", "occurrence"])
def test_authored_stage_names_precede_call_aliases_with_and_without_composition(
    composition, expected, collision
):
    from embodichain.lab.gym.envs.managers.reward_composition import (
        compose_reward_terms,
        _add_common_rewards,
    )
    from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext

    selector = "call:0"
    if collision == "occurrence":
        fixture, _, _, _, _ = _setup(stage_names=("first", "second"))
        selector = next(
            fixture.task_program_progress._program_for_test.iter_segments()
        ).segment_id
    env, owner, _, _, _ = _setup(stage_names=("first", selector))
    env.step_dt = 0.04
    progress = env.task_program_progress
    manual = RewardCfg(
        func=program_stage_reward,
        weight=7.0,
        params={"stage": selector, "mode": "first_achievement", "scope": "occurrence"},
    )
    terms = {"manual": manual}
    if composition:
        declaration = {"discount": 0.99, "skills": []}
        generated = compose_reward_terms(declaration, program=True)
        _add_common_rewards(
            SimpleNamespace(
                rl_task=SimpleNamespace(reward_composition=declaration),
                rewards=SimpleNamespace(manual=manual),
            ),
            generated,
            tuple(progress._program_for_test.iter_segments()),
        )
        terms.update(generated)
    manager = env.reward_manager = RewardManager(terms, env)
    if composition:
        manager.initialize_observation(_frame(owner, [0.0, 0.0], 0), [0, 1])
    for step, reward in enumerate(expected):
        frame = _frame(owner, [0.2, 0.0], step + 1)
        progress.advance(frame, step=step, episode_owner=owner)
        if composition:
            manager.transition_context = RewardTransitionContext(
                frame, torch.zeros(2, dtype=torch.bool)
            )
        value, _ = manager.compute(
            {},
            torch.zeros(2, 1),
            {
                "task_outcome": {
                    "accepted": torch.zeros(2, dtype=torch.bool),
                    "failure": torch.zeros(2, dtype=torch.bool),
                },
            },
        )
        torch.testing.assert_close(
            value, torch.tensor([reward, 0.0]), atol=1e-6, rtol=0
        )
    assert progress.indices(selector) == (1,)

    class StageName(str):
        pass

    assert progress.indices(StageName(selector)) == (1,)


def test_structured_call_index_shaping_is_not_shadowed_by_an_authored_name():
    from embodichain.lab.gym.envs.managers.reward_composition import (
        compose_reward_terms,
    )
    from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext

    env, owner, _, _, _ = _setup(stage_names=("first", "call:0"))
    env.episode_context = owner
    manager = env.reward_manager = RewardManager(
        compose_reward_terms(
            {
                "discount": 0.99,
                "skills": [
                    {
                        "skill_id": "move_joints",
                        "call_index": 0,
                        "metrics": {"joints": {"field": "score"}},
                    }
                ],
            },
            program=True,
        ),
        env,
    )

    def measured(scores, revision):
        frame = _frame(owner, [0.0, 0.0], revision)
        value = torch.tensor(scores)
        sample = ObservationSample(
            ObservationDescriptor(
                "score", "scalar", "arena", "scalar", (), (), value.dtype
            ),
            value,
            torch.ones(2, dtype=torch.bool),
            (None, None),
        )
        return replace(frame, fields={**frame.fields, "score": sample})

    manager.initialize_observation(measured([0.2, 0.4], 0), [0, 1])
    frame = measured([0.8, 0.6], 1)
    env.task_program_progress.advance(frame, step=0, episode_owner=owner)
    manager.transition_context = RewardTransitionContext(
        frame, torch.zeros(2, dtype=torch.bool)
    )
    reward, _ = manager.compute({}, torch.zeros(2, 1), {})
    # Two progress units; only the first call's score contributes to phi.
    torch.testing.assert_close(reward, torch.tensor([0.296, 0.097]))


@pytest.mark.parametrize("selector", ["call:0", "occurrence", "lift"])
def test_explicit_milestone_alias_suppresses_automatic_reward_for_same_event(selector):
    from embodichain.lab.gym.envs.managers.reward_composition import (
        compose_reward_terms,
        _add_common_rewards,
    )
    from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext

    env, owner, _, _, _ = _setup(repeat=2)
    env.step_dt = 0.04
    progress = env.task_program_progress
    stages = tuple(progress._program_for_test.iter_segments())
    selected = stages[0].segment_id if selector == "occurrence" else selector
    manual = RewardCfg(
        func=program_stage_reward,
        weight=7.0,
        params={
            "stage": selected,
            "mode": "first_achievement",
            "scope": "occurrence",
        },
    )
    specification = {"discount": 0.99, "skills": []}
    generated = compose_reward_terms(specification, program=True)
    _add_common_rewards(
        SimpleNamespace(
            rl_task=SimpleNamespace(reward_composition=specification),
            rewards=SimpleNamespace(manual=manual),
        ),
        generated,
        stages,
    )
    env.reward_manager = RewardManager({**generated, "manual": manual}, env)
    env.reward_manager.initialize_observation(_frame(owner, [0.0, 0.0], 0), [0, 1])
    frame = _frame(owner, [0.2, 0.0], 1)
    progress.advance(frame, step=0, episode_owner=owner)
    env.reward_manager.transition_context = RewardTransitionContext(
        frame, torch.zeros(2, dtype=torch.bool)
    )
    info = {
        "task_outcome": {
            "accepted": torch.zeros(2, dtype=torch.bool),
            "failure": torch.zeros(2, dtype=torch.bool),
        }
    }
    value, _ = env.reward_manager.compute({}, torch.zeros(2, 1), info)
    # First stage: generic shaping .99*.5, explicitly selected event 7, no extra .5.
    torch.testing.assert_close(value, torch.tensor([7.495, 0.0]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("selector,first_reward", [("lift", 7.495), ("call:1", 0.995)])
def test_episode_scoped_bonus_only_suppresses_events_it_can_pay(selector, first_reward):
    from embodichain.lab.gym.envs.managers.reward_composition import (
        compose_reward_terms,
        _add_common_rewards,
    )
    from embodichain.lab.gym.envs.managers.reward_context import RewardTransitionContext

    env, owner, _, _, _ = _setup(repeat=2)
    env.step_dt = 0.04
    progress = env.task_program_progress
    manual = RewardCfg(
        func=program_stage_reward,
        weight=7.0,
        params={"stage": selector, "mode": "first_achievement", "scope": "episode"},
    )
    specification = {"discount": 0.99, "skills": []}
    generated = compose_reward_terms(specification, program=True)
    _add_common_rewards(
        SimpleNamespace(
            rl_task=SimpleNamespace(reward_composition=specification),
            rewards=SimpleNamespace(manual=manual),
        ),
        generated,
        tuple(progress._program_for_test.iter_segments()),
    )
    manager = env.reward_manager = RewardManager({**generated, "manual": manual}, env)
    manager.initialize_observation(_frame(owner, [0.0, 0.0], 0), [0, 1])
    for step, expected in enumerate((first_reward, 0.99)):
        frame = _frame(owner, [0.2, 0.0], step + 1)
        progress.advance(frame, step=step, episode_owner=owner)
        manager.transition_context = RewardTransitionContext(
            frame, torch.zeros(2, dtype=torch.bool)
        )
        value, _ = manager.compute(
            {},
            torch.zeros(2, 1),
            {
                "task_outcome": {
                    "accepted": torch.zeros(2, dtype=torch.bool),
                    "failure": torch.zeros(2, dtype=torch.bool),
                }
            },
        )
        torch.testing.assert_close(
            value, torch.tensor([expected, 0.0]), atol=1e-6, rtol=0
        )
