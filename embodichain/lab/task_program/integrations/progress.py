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

"""Measured Task Program progress for policy-owned execution.

This adapter composes the existing evaluator and milestone rules. It owns no
controller, expert bridge, observation acquisition or second history store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import json
from typing import Any, TYPE_CHECKING

import torch

from embodichain.lab.evaluation import (
    BooleanRule,
    CompiledEvaluation,
    EvaluationStatus as Status,
    MilestoneRule,
    RuleSession,
)
from embodichain.lab.evaluation._graph import _BoundPredicate
from embodichain.lab.evaluation.observations import (
    _graph_frame_inputs,
    _rule_tensor_report,
)
from embodichain.lab.task_program.compiler.program import (
    CompiledConditionValidator,
    CompiledTaskProgram,
    TaskProgramCompileError,
)
from ._program_identity import task_program_fingerprint

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_types import ObservationFrame

__all__ = ["TaskProgramProgress"]


def _progress_graph(program, conditions, final_rules, process_rules):
    nodes, names, fields = [], {}, set()

    def include(spec):
        if spec.stateful:
            raise ValueError(
                "Policy segment criteria must be instantaneous; temporal criteria need an explicit policy occurrence window."
            )
        local = {}
        for name, rule in spec._nodes:
            key = id(rule)
            if key not in names:
                target = f"condition/{len(names)}"
                names[key] = target
                if isinstance(rule, BooleanRule):
                    rule = replace(rule, refs=tuple(local[ref] for ref in rule.refs))
                elif not isinstance(rule, _BoundPredicate):
                    raise ValueError("Unsupported policy evaluation node.")
                nodes.append((target, rule))
            local[name] = names[key]
        fields.update(spec.required_fields)
        if any(
            field.endswith("/segment_start_frame") for field in spec.required_fields
        ):
            raise ValueError(
                "Policy criteria cannot use expert segment-start observations; bind a measured scene reference."
            )
        return local[spec.roots[0]]

    stages = tuple(program.iter_segments())
    roots, current, achievements = [], [], []
    for i, segment in enumerate(stages):
        path = segment.source_path

        def fail(message):
            raise TaskProgramCompileError("unsupported_policy_segment", path, message)

        if (
            segment.implicit
            or segment.parallel_block is not None
            or segment.post_policies
        ):
            fail(
                "Policy progress supports explicit sequential segments without expert post-policies."
            )
        if not segment.instruction or segment.contract is None:
            fail(
                "Policy segments require an instruction and measured acceptance contract."
            )
        refs = []
        for entry in segment.contract.prerequisites:
            if entry.kind != "condition":
                fail(
                    "Policy prerequisites must reference measured conditions, not expert symbolic state."
                )
            refs.append(include(conditions[entry.ref]))
        if not segment.validators or any(
            type(v) is not CompiledConditionValidator for v in segment.validators
        ):
            fail("Policy segment validators must use named shared conditions.")
        for entry in segment.contract.outcomes:
            if entry.required and entry.kind != "validator":
                fail(
                    "Required policy outcomes must reference measured condition validators."
                )
        for validator in segment.validators:
            refs.append(include(conditions[validator.cfg.ref]))
        goal, reached = f"stage/{i}/current", f"stage/{i}/achieved"
        nodes.append((goal, BooleanRule("all", tuple(dict.fromkeys(refs)))))
        nodes.append(
            (reached, MilestoneRule((goal,), () if i == 0 else (achievements[-1],)))
        )
        current.append(goal)
        achievements.append(reached)
        roots.extend((goal, reached))
    if not stages:
        raise ValueError("Policy progress requires at least one segment.")
    gates = {}
    for label, specs in (("final", final_rules), ("process", process_rules)):
        refs = tuple(include(spec) for spec in specs.values())
        if refs:
            key = f"gate/{label}"
            nodes.append((key, BooleanRule("all", tuple(dict.fromkeys(refs)))))
            gates[label] = key
            roots.append(key)
    identity = sha256(
        json.dumps(
            [
                task_program_fingerprint(program),
                [(name, spec.identity) for name, spec in conditions.items()],
                [(name, spec.identity) for name, spec in final_rules.items()],
                [(name, spec.identity) for name, spec in process_rules.items()],
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    graph = CompiledEvaluation(
        tuple(roots), tuple(sorted(fields)), identity, tuple(nodes)
    )
    graph._preset_joint_names()
    return stages, graph, tuple(current), tuple(achievements), gates


class TaskProgramProgress:
    """Cache measured stage truth and ordered achievements once per transition.

    The supported program subset is explicit sequential segments (including
    finite expanded repeats), with instantaneous named condition validators and
    condition-only prerequisites. Expert effects, symbolic state and action
    post-policies cannot certify policy progress. Historical achievements use
    the evaluator's MilestoneRule; current truth can independently revoke.

    Args:
        program: Provider-free compiled Task Program declarations.
        conditions: Trusted deployment rules, with native joint layouts bound.
        env_ids: Original environment IDs in result-row order.
        episode_ids: Expected episode identity for each row.
        device: Evaluation device.
        final_rules: Protected deployment final conditions.
        process_rules: Protected deployment process conditions.
    """

    def __init__(
        self,
        program: CompiledTaskProgram,
        conditions: Mapping[str, CompiledEvaluation],
        *,
        env_ids: Sequence[int],
        episode_ids: Sequence[int],
        device: str | torch.device = "cpu",
        final_rules: Mapping[str, CompiledEvaluation] | None = None,
        process_rules: Mapping[str, CompiledEvaluation] | None = None,
    ) -> None:
        (
            self._stages,
            self.compiled,
            self._current_refs,
            self._achievement_refs,
            self._gates,
        ) = _progress_graph(program, conditions, final_rules or {}, process_rules or {})
        self.program_identity = task_program_fingerprint(program)
        self._session = RuleSession(
            self.compiled,
            env_ids=env_ids,
            episode_ids=episode_ids,
            scope=f"policy/{self.program_identity}",
            device=device,
        )
        self._full_batch = tuple(env_ids) == tuple(range(len(env_ids)))
        shape = (len(env_ids), len(self._stages))
        self._current = torch.full(
            shape, Status.NOT_RUN, device=device, dtype=torch.int8
        )
        self._achieved = torch.zeros(shape, device=device, dtype=torch.bool)
        self._first = torch.zeros_like(self._achieved)
        self._first_episode = torch.zeros_like(self._achieved)
        self._rising = torch.zeros_like(self._achieved)
        self._eligible = torch.zeros_like(self._achieved)
        self._valid = torch.zeros_like(self._achieved)
        self._gate_results = {}
        self._snapshot = self._session.snapshot()
        seen = set()
        first_occurrences = []
        for stage in self._stages:
            first_occurrences.append(stage.name not in seen)
            seen.add(stage.name)
        self._first_occurrences = torch.tensor(
            first_occurrences, dtype=torch.bool, device=device
        )
        self._stage_dependencies = tuple(
            self._dependencies(ref) for ref in self._current_refs
        )

    @property
    def stages(self) -> tuple[str, ...]:
        """Return authored stage names, including repeated occurrences."""
        return tuple(stage.name for stage in self._stages)

    def indices(self, stage: str | None) -> tuple[int, ...]:
        """Resolve a stage selection.

        Args:
            stage: Authored name or occurrence ID; None selects all stages.

        Returns:
            Matching occurrence indices in program order.
        """
        if stage is None:
            return tuple(range(len(self._stages)))
        selected = tuple(
            i for i, s in enumerate(self._stages) if stage in (s.name, s.segment_id)
        )
        if not selected:
            raise ValueError(f"Unknown policy stage {stage!r}.")
        return selected

    def advance(
        self, frame: ObservationFrame, *, step: int, episode_owner: EpisodeContext
    ) -> None:
        """Consume a transition; duplicate slot reads do not advance.

        Args:
            frame: Canonical measured observations with source identity.
            step: Strictly increasing global control slot.
            episode_owner: Authoritative context whose rows qualify the frame.
        """
        if step == self._session._last_step and self._session._advanced:
            return
        before = self._achieved.clone()
        prior_current = (self._current == Status.SATISFIED) & self._eligible
        rows = self._session.env_ids
        current_episodes = episode_owner.episode_ids
        foreign = (
            frame.episode_owner is not episode_owner
            or (
                self._session._observation_owner is not None
                and self._session._observation_owner is not episode_owner
            )
            or len(frame.episode_ids) != len(current_episodes)
            or max(rows) >= len(current_episodes)
        )
        if foreign:
            self._session.advance(
                step,
                {},
                revisions=torch.full_like(self._session._revisions, -1),
                episode_ids=self._session.episode_ids,
            )
        else:
            if self._session._observation_rows is None:
                self._session._observation_rows = torch.tensor(
                    rows, device=self._current.device, dtype=torch.long
                )
            indices = self._session._observation_rows
            values, valid = _graph_frame_inputs(self.compiled, frame, indices)
            # Session qualification rejects foreign episodes row by row; owner
            # mismatches also invalidate only their affected rows.
            selected_owner = (
                current_episodes
                if self._full_batch and len(current_episodes) == len(rows)
                else tuple(current_episodes[row] for row in rows)
            )
            selected_source = (
                frame.episode_ids
                if self._full_batch and len(frame.episode_ids) == len(rows)
                else tuple(frame.episode_ids[row] for row in rows)
            )
            if selected_owner == self._session.episode_ids:
                revisions = torch.full_like(indices, frame.revision)
            else:
                revisions = torch.tensor(
                    [
                        frame.revision if actual == expected else -1
                        for actual, expected in zip(
                            selected_owner, self._session.episode_ids, strict=True
                        )
                    ],
                    device=self._current.device,
                    dtype=torch.long,
                )
            self._session.advance(
                step,
                values,
                valid=valid,
                revisions=revisions,
                episode_ids=selected_source,
            )
            self._session._observation_owner = episode_owner
        self._snapshot = self._session.snapshot()
        results = self._snapshot.rules
        self._current.copy_(
            torch.stack([results[ref].status for ref in self._current_refs], dim=1)
        )
        self._achieved.copy_(
            torch.stack(
                [
                    results[ref].status == Status.SATISFIED
                    for ref in self._achievement_refs
                ],
                dim=1,
            )
        )
        self._eligible[:, 0] = True
        self._eligible[:, 1:] = before[:, :-1]
        self._valid.copy_(
            (self._current == Status.SATISFIED) | (self._current == Status.FAILED)
        )
        self._first.copy_(self._achieved & ~before)
        self._rising.copy_(
            (self._current == Status.SATISFIED) & self._eligible & ~prior_current
        )
        self._first_episode.copy_(self._first & self._first_occurrences)
        self._gate_results = {name: results[ref] for name, ref in self._gates.items()}

    def reward(
        self,
        stage: str,
        *,
        mode: str,
        scope: str | None = None,
        metric: str | None = None,
    ) -> torch.Tensor:
        """Read a configured reward source without consuming trigger state.

        ``first_achievement`` requires explicit ``occurrence`` or ``episode``
        scope. ``rising`` allows revocation and re-achievement; ``satisfied``
        is a level reward. ``metric`` uses an exact measured metric key.
        Weight, sign and shaping scale belong to RewardManager configuration.

        Args:
            stage: Authored name or exact expanded occurrence ID.
            mode: satisfied, rising, first_achievement or metric.
            scope: Explicit occurrence/episode first-achievement scope.
            metric: Exact scalar metric key, required in metric mode.

        Returns:
            One owned reward source per row; unavailable metrics contribute zero.
        """
        ids = self.indices(stage)
        if mode == "first_achievement":
            if scope not in ("occurrence", "episode") or metric is not None:
                raise ValueError(
                    "First achievement requires explicit occurrence/episode scope."
                )
            value = self._first if scope == "occurrence" else self._first_episode
        else:
            if scope is not None:
                raise ValueError("Bonus scope applies only to first_achievement.")
            if mode == "satisfied":
                value = (self._current == Status.SATISFIED) & self._eligible
            elif mode == "rising":
                value = self._rising
            elif mode == "metric":
                if not metric:
                    raise ValueError("Metric reward requires an exact metric key.")
                values = []
                for i in ids:
                    # Predicate metrics live on their existing graph nodes.
                    found = [
                        r.metrics[metric]
                        for name in self._stage_dependencies[i]
                        if metric in (r := self._snapshot.rules[name]).metrics
                    ]
                    if not found:
                        values.append(
                            torch.zeros(
                                self._current.shape[0], device=self._current.device
                            )
                        )
                        continue
                    if len(found) != 1:
                        raise ValueError(
                            f"Metric {metric!r} is missing or ambiguous for stage {stage!r}."
                        )
                    if found[0].ndim != 1:
                        raise ValueError(
                            "Program metric rewards require a scalar per environment."
                        )
                    values.append(
                        torch.where(
                            self._valid[:, i] & self._eligible[:, i],
                            torch.nan_to_num(found[0]),
                            0.0,
                        )
                    )
                return torch.stack(values, 1).sum(1)
            else:
                raise ValueError(f"Unsupported policy reward mode {mode!r}.")
        if mode != "metric" and metric is not None:
            raise ValueError("Only metric rewards accept metric.")
        return value[:, ids].float().sum(1)

    def _dependencies(self, root):
        nodes = dict(self.compiled._nodes)
        selected = set()

        def visit(name):
            if name in selected:
                return
            selected.add(name)
            rule = nodes[name]
            for ref in getattr(rule, "refs", ()):
                visit(ref)

        visit(root)
        return selected

    def condition(
        self, stage: str | None = None, *, state: str = "achieved"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read cached satisfaction and evidence validity.

        Args:
            stage: Authored name or occurrence ID; None selects the whole program.
            state: Historical achieved or revocable current truth.

        Returns:
            Owned boolean satisfaction and validity vectors in result-row order.
        """
        ids = self.indices(stage)
        if state == "achieved":
            result = self._achieved[:, ids].all(1)
        elif state == "current":
            result = (
                (self._current[:, ids] == Status.SATISFIED) & self._eligible[:, ids]
            ).all(1)
        else:
            raise ValueError("Stage condition state must be achieved or current.")
        if stage is None:
            active = self._eligible & ~self._achieved
            active[:, -1] |= self._achieved.all(1)
            valid = (self._valid | ~active).all(1) & self._eligible[:, 0]
        else:
            valid = self._valid[:, ids].all(1)
        return result.clone(), valid.clone()

    def gates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return protected final truth, process failure and evidence validity.

        Returns:
            Owned boolean final-success, process-failure and validity vectors.
        """
        shape = self._current.shape[:1]
        success = torch.ones(shape, device=self._current.device, dtype=torch.bool)
        failed = torch.zeros_like(success)
        valid = torch.ones_like(success)
        for name, result in self._gate_results.items():
            known = (result.status == Status.SATISFIED) | (
                result.status == Status.FAILED
            )
            valid &= known
            if name == "final":
                success &= result.status == Status.SATISFIED
            else:
                failed |= result.status != Status.SATISFIED
        return success, failed, valid

    def reset(self, env_ids: Sequence[int], episode_ids: Sequence[int]) -> None:
        """Clear only selected rows, including achievement and bonus identity.

        Args:
            env_ids: Original environment IDs being reset.
            episode_ids: Their new expected episode identities in matching order.
        """
        self._session.reset(env_ids, episode_ids)
        rows = [self._session.env_ids.index(row) for row in env_ids]
        self._current[rows] = Status.NOT_RUN
        for value in (
            self._achieved,
            self._first,
            self._first_episode,
            self._rising,
            self._eligible,
            self._valid,
        ):
            value[rows] = False
        self._snapshot = self._session.snapshot()
        self._gate_results = {
            name: self._snapshot.rules[ref] for name, ref in self._gates.items()
        }

    def snapshot(self) -> dict[str, Any]:
        """Read progress without advancing or exposing mutable session state.

        Returns:
            Owned progress tensors, rule reports and expected occurrence identities.
        """
        return {
            "program_identity": self.program_identity,
            "evaluation_identity": self.compiled.identity,
            "env_ids": self._session.env_ids,
            "episode_ids": self._session.episode_ids,
            "step": self._snapshot.step,
            "acquisition_revisions": self._snapshot.revisions.clone(),
            "stage_names": self.stages,
            "occurrence_ids": tuple(s.segment_id for s in self._stages),
            "current": self._current.clone(),
            "achieved": self._achieved.clone(),
            "first_achievement": self._first.clone(),
            "first_in_episode": self._first_episode.clone(),
            "rules": {
                name: _rule_tensor_report(result)
                for name, result in self._snapshot.rules.items()
            },
        }


def _validate_policy_deployment(cfg, registration):
    """Validate supported declaration/control bindings without constructing a robot."""
    import math
    from embodichain.lab.gym.envs.managers.conditions import program_stage_condition
    from embodichain.lab.gym.envs.managers.rewards import program_stage_reward

    _validate_policy_extensions(cfg)
    owners = cfg.task_program_action_owners
    resources = {
        resource.resource_id
        for resource in registration.robot_profile_binding.resources
    }
    if (
        not isinstance(owners, Mapping)
        or set(owners) != resources
        or any(owner != "policy" for owner in owners.values())
    ):
        raise ValueError(
            "task_program.action_owners must assign every declared robot resource exactly to policy; expert/policy co-ownership is unsupported."
        )
    if not cfg.actions:
        raise ValueError("Policy Task Program execution requires env.actions.")
    program = registration.catalog.preflight(cfg.task_program, strict_contracts=True)
    stages, _, _, _, _ = _progress_graph(
        program,
        registration.evaluation_conditions,
        {
            ref: registration.evaluation_conditions[ref]
            for ref in registration.evaluation_final_refs
        },
        {
            ref: registration.evaluation_conditions[ref]
            for ref in registration.evaluation_process_refs
        },
    )
    selections = {s.name for s in stages} | {s.segment_id for s in stages}
    success_terms = cfg.rl_task.success.values()
    if not any(term.func is program_stage_condition for term in success_terms):
        raise ValueError(
            "Policy progress requires an explicit program_stage_condition in rl_task.success."
        )
    for group in (cfg.rl_task.success, cfg.rl_task.failure, cfg.rl_task.truncation):
        for term in group.values():
            if term.func is program_stage_condition:
                if term.params.get("state") not in ("achieved", "current"):
                    raise ValueError(
                        "Program condition state must be achieved or current."
                    )
                if (
                    term.params.get("stage") is not None
                    and term.params["stage"] not in selections
                ):
                    raise ValueError("Unknown policy stage in termination binding.")
    rewards = cfg.rewards
    terms = (
        rewards.values()
        if isinstance(rewards, Mapping)
        else (
            (getattr(rewards, name) for name in vars(rewards))
            if rewards is not None
            else ()
        )
    )
    for term in terms:
        if term is None or getattr(term, "func", None) is not program_stage_reward:
            continue
        params = term.params
        scale = params.get("scale", 1.0)
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
        ):
            raise ValueError("Program reward scale must be finite and numeric.")
        if params.get("stage") not in selections:
            raise ValueError("Unknown policy stage in reward binding.")
        mode = params.get("mode")
        if mode not in ("satisfied", "first_achievement", "rising", "metric"):
            raise ValueError("Unknown program stage reward mode.")
        if mode == "first_achievement":
            if params.get("scope") not in ("occurrence", "episode"):
                raise ValueError(
                    "First-achievement rewards require explicit occurrence/episode scope."
                )
        elif params.get("scope") is not None:
            raise ValueError("Bonus scope applies only to first_achievement.")
        metric = params.get("metric")
        if (mode == "metric" and (not isinstance(metric, str) or not metric)) or (
            mode != "metric" and metric is not None
        ):
            raise ValueError(
                "Exactly metric mode requires an explicit scalar metric key."
            )
    return program


def _validate_policy_extensions(cfg):
    reserved = {
        "sim",
        "robot",
        "episode_context",
        "task_program_progress",
        "termination_manager",
        "action_manager",
        "reward_manager",
        "observation_manager",
    }
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    for name in getattr(cfg, "extensions", {}) or {}:
        if (
            name.startswith("_")
            or name in reserved
            or hasattr(cfg, name)
            or hasattr(EmbodiedEnv, name)
        ):
            raise ValueError(
                f"Policy extensions cannot override environment-owned field {name!r}."
            )


def _validate_policy_action_coverage(registration, robot, manager):
    if manager is None:
        raise ValueError("Policy Task Program execution requires an ActionManager.")
    controlled = {
        joint
        for descriptor in manager.descriptors
        for joint in descriptor.term.joint_names
    }
    required = set()
    for resource in registration.robot_profile_binding.resources:
        for endpoint in getattr(resource, "endpoints", ()):
            required.update(
                robot.joint_names[i]
                for i in robot.get_joint_ids(
                    name=endpoint.control_part, remove_mimic=True
                )
            )
    if required - controlled:
        raise ValueError(
            f"Policy actions do not own declared resource joints: {sorted(required - controlled)!r}."
        )
