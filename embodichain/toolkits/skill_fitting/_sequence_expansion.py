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

"""Fixed-scene native-skill variants with original demonstration gaps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import yaml

from embodichain.data_pipeline.skill_fitting import file_digest
from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile

if TYPE_CHECKING:
    from embodichain.lab.scripts.run_task import CollectionPlan
    from embodichain.lab.sim.atomic_skills.engine import (
        AtomicSkillEngine,
        PlanTransform,
    )
    from embodichain.lab.sim.atomic_skills.invocation import ResolvedSkillRequest
    from embodichain.lab.sim.atomic_skills.plans import SkillPlan
    from embodichain.lab.sim.atomic_skills.state import PlanningContext
    from ._robot import RobotProfile

__all__: list[str] = []


def validate_sequence_expansion(
    profile: CombinedExpansionProfile, manifest: Path, num_envs: int
) -> dict[str, Any]:
    """Reject unsupported owners/factors before creating a simulation."""
    if not isinstance(profile, CombinedExpansionProfile):
        raise ValueError(
            "Native sequence expansion requires a CombinedExpansionProfile."
        )
    data = yaml.safe_load(manifest.read_text())
    if not isinstance(data, dict) or data.get("schema") != "native-skill-sequence/v1":
        raise ValueError("Expansion requires a native-skill-sequence/v1 manifest.")
    if (
        profile.source.kind != "skill_sequence"
        or profile.source.source_id != manifest.stem
    ):
        raise ValueError(
            "Expansion source must select kind=skill_sequence and source_id matching the manifest stem."
        )
    if profile.source.source_revision != data["schema"]:
        raise ValueError("Expansion source_revision must match the sequence schema.")
    if profile.source.call_count != len(data["timeline"]):
        raise ValueError(
            "Expansion source.call_count must count every timeline span, including DemoSkill."
        )
    if num_envs != 1 or profile.execution.max_inflight != 1:
        raise ValueError(
            "Native sequence expansion currently requires B=1 and max_inflight=1."
        )
    if (
        profile.scene_randomization.enabled
        or profile.scene_randomization.reference_family_count != 1
        or profile.affordance.enabled
        or profile.affordance.branches_per_family != 1
        or profile.trajectory.timing.enabled
        or profile.visual.enabled
        or profile.observation.profiles
        or profile.observation.capture_rgb
        or profile.observation.capture_video
    ):
        raise ValueError(
            "Native sequence expansion currently supports fixed-scene spatial trajectory variants only."
        )
    if profile.persistence.sink != "local_artifact":
        raise ValueError(
            "Native sequence expansion requires persistence.sink=local_artifact (causal HDF5 traces)."
        )
    eligible = [
        i
        for i in profile.source.trajectory_call_indices
        if data["timeline"][i]["kind"] == "atomic_skill"
    ]
    if profile.trajectory.variants_per_family > 1 and (
        not eligible or not profile.trajectory.spatial.enabled
    ):
        raise ValueError(
            "Non-nominal variants require spatial expansion on at least one atomic span."
        )
    return data


def validate_sequence_collection(
    args: Any,
    gym_config: Mapping[str, Any],
    profile: CombinedExpansionProfile,
    indices: tuple[int, ...],
    collection: Mapping[str, Any],
) -> CollectionPlan:
    """Reject unsupported recording/recipe requests before creating a simulation."""
    from embodichain.lab.scripts.run_task import (
        _validate_expansion_collection_plan,
        resolve_collection_plan,
    )

    plan = resolve_collection_plan(
        args, gym_config, expansion_collection=collection, legacy_recipe_indices=indices
    )
    _validate_expansion_collection_plan(profile, plan, num_envs=1)
    if getattr(args, "expansion_dataset_dir", None) or getattr(
        args, "expansion_save_video", False
    ):
        raise ValueError(
            "Native sequence expansion writes local HDF5 traces; dataset/video export is not connected."
        )
    return plan


def _transform_factory(
    profile: CombinedExpansionProfile,
    source_digest: str,
    recipe_index: int,
    attempt: int,
    robot_profile: RobotProfile,
    sampling_seed: int = 0,
) -> Callable[[int, AtomicSkillEngine], PlanTransform | None]:
    """Keep candidate work bounded to nominal + one proposal per invocation."""
    from embodichain.lab.sim.atomic_skills._plan_expansion import (
        _PlanExpansionSpec,
        _expand_plan,
    )
    from embodichain.lab.sim.motion.expansion import (
        SceneCase,
        SourceContext,
        TrajectoryExpansionCfg,
    )

    def factory(index: int, engine: AtomicSkillEngine) -> PlanTransform | None:
        if index not in profile.source.trajectory_call_indices:
            return None
        order = [robot_profile.joint_names.index(n) for n in engine.robot.joint_names]
        velocity = torch.tensor(robot_profile.velocity_limits)[order]
        acceleration = torch.tensor(robot_profile.acceleration_limits)[order]
        spatial = profile.trajectory.spatial
        # Each episode ordinal owns an independent reproducible proposal stream.
        # No list of all previously generated variants is retained in memory.
        requested = int(recipe_index > 0)
        operator = spatial.operators[
            (max(1, recipe_index) - 1) % len(spatial.operators)
        ]
        seed_text = json.dumps(
            [
                source_digest,
                profile.to_dict(),
                sampling_seed,
                recipe_index,
                index,
                attempt,
            ],
            sort_keys=True,
        )
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:15], 16)
        cfg = TrajectoryExpansionCfg.from_mapping(
            dict(
                augmentation=dict(
                    seed=seed,
                    max_variants_per_reference=2,
                    factors=dict(
                        spatial=dict(
                            enabled=bool(requested),
                            method=[operator],
                            joint_offset_scale=spatial.joint_offset_scale,
                            via_count=spatial.via_count,
                        )
                    ),
                    coverage=dict(
                        joint_dedup_normalized_tol=(
                            spatial.dedup_tolerance
                            if spatial.dedup_tolerance is not None
                            else 0.002
                        )
                    ),
                ),
                affordance=dict(enabled=False),
                scheduling=dict(candidate_budget=2),
                execution=dict(max_inflight=1),
                observation=dict(enabled=False, profiles=[]),
            )
        )

        def transform(
            request: ResolvedSkillRequest, context: PlanningContext, plan: SkillPlan
        ) -> SkillPlan:
            source = SourceContext(
                profile.source.source_id,
                source_digest,
                f"span:{index}",
                SceneCase(
                    "skill_sequence",
                    "fixed_scene",
                    source_digest,
                    plan.skill_id,
                    engine.robot.uid,
                ),
                context.require_control_dt(),
            )
            spec = _PlanExpansionSpec(
                source,
                cfg,
                requested,
                1 + requested,
                dict(recipe_index=recipe_index, span_index=index, proposal_seed=seed),
                protect_derivative_stencils=True,
                velocity_limits=velocity,
                acceleration_limits=acceleration,
            )
            rebuilt, records = _expand_plan(
                engine,
                request,
                context,
                plan,
                [spec],
                phase_permissions=profile.source.phase_permissions or None,
                phase_kinds=profile.source.phase_kinds or None,
            )
            if requested and records[0].metadata.get("trajectory_selected") != 1:
                raise ValueError(
                    "Requested native trajectory variant could not be generated; nominal fallback is disabled."
                )
            return rebuilt

        return transform

    return factory


def run_sequence_expansion(env: Any, args: Any, gym_config: dict[str, Any]) -> None:
    """Execute and persist selected complete episodes, preserving raw DemoSkill."""
    from embodichain.lab.scripts.run_task import (
        _resolve_expansion_request,
    )
    from ._execution import execute_sequence
    from ._robot import RobotProfile
    from ._bundle import _contained

    resolved = _resolve_expansion_request(args, gym_config)
    if resolved is None:
        raise ValueError("Native sequence expansion requires an Expansion Profile.")
    profile, indices, profile_path, collection = resolved
    manifest = Path(args.skill_sequence).resolve(strict=True)
    data = validate_sequence_expansion(profile, manifest, env.unwrapped.num_envs)
    robot_profile = RobotProfile.load(
        _contained(manifest.parent, data["robot_profile"])
    )
    plan = validate_sequence_collection(args, gym_config, profile, indices, collection)
    selected = plan.selection.take(0, plan.target_episodes)
    sampling_seed = getattr(args, "seed", None) or 0
    value = getattr(args, "expansion_output_dir", None)
    output = Path(value or profile.persistence.output_dir).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() if value else profile_path.parent) / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    digest = file_digest(manifest)
    record = dict(
        schema="native-skill-sequence-expansion/v1",
        source_manifest_sha256=digest,
        profile_sha256=file_digest(profile_path),
        profile=profile.to_dict(),
        demo_policy="original_commands_no_augmentation",
        command_completion_target=plan.target_episodes,
        command_completed_episodes=0,
        physical_success="unknown",
        sampling_seed=sampling_seed,
        episodes=[],
    )

    def publish():
        temporary = output / ".expansion_manifest.json.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(output / "expansion_manifest.json")

    publish()
    for ordinal in selected:
        episode = dict(recipe_index=ordinal, completed=False, attempts=[])
        record["episodes"].append(episode)
        for attempt in range(plan.max_attempts):
            if file_digest(manifest) != digest:
                raise ValueError("Sequence manifest changed during expansion.")
            directory = output / f"recipe-{ordinal:06d}" / f"attempt-{attempt:02d}"
            attempt_record = dict(
                attempt=attempt, directory=directory.relative_to(output).as_posix()
            )
            episode["attempts"].append(attempt_record)
            try:
                result = execute_sequence(
                    env,
                    manifest,
                    output_dir=directory,
                    plan_transform_factory=_transform_factory(
                        profile, digest, ordinal, attempt, robot_profile, sampling_seed
                    ),
                    record_trajectory=True,
                )
                attempt_record.update(
                    completed=True,
                    command_count=result["command_count"],
                    spans=result["spans"],
                )
                episode["completed"] = True
                record["command_completed_episodes"] += 1
            except ValueError as error:
                attempt_record.update(completed=False, error=str(error))
            finally:
                publish()
            if episode["completed"]:
                break
    if record["command_completed_episodes"] != plan.target_episodes:
        raise RuntimeError(
            f"Only {record['command_completed_episodes']}/{plan.target_episodes} native expansion episodes completed; see {output / 'expansion_manifest.json'}"
        )
    print(
        f"Native sequence expansion completed: {output / 'expansion_manifest.json'}",
        flush=True,
    )


def write_sequence_expansion_profile(
    directory: Path, timeline: list[dict[str, Any]]
) -> None:
    """Provide a small editable native-only profile beside each exported task."""
    atomic_indices = [
        i for i, span in enumerate(timeline) if span["kind"] == "atomic_skill"
    ]
    variants = 3 if atomic_indices else 1
    payload = dict(
        schema_version=1,
        source=dict(
            kind="skill_sequence",
            source_id="skills",
            source_revision="native-skill-sequence/v1",
            unit_scope="episode",
            template_id="native_atomic_call",
            call_count=len(timeline),
            affordance_call_indices=[],
            trajectory_call_indices=atomic_indices,
        ),
        scene_randomization=dict(
            enabled=False, reference_family_count=1, profile="none", change_scope="none"
        ),
        affordance=dict(enabled=False, branches_per_family=1, max_proposals=1),
        trajectory=dict(
            variants_per_family=variants,
            spatial=dict(
                enabled=bool(atomic_indices),
                operators=["joint_residual", "via_points"],
                joint_offset_scale=0.001,
                via_count=1,
                dedup_tolerance=0.0001,
            ),
            timing=dict(enabled=False),
        ),
        visual=dict(enabled=False, profiles=[]),
        observation=dict(profiles=[], capture_rgb=False, capture_video=False),
        scheduling=dict(candidate_budget=variants, reference_family_budget=1),
        execution=dict(max_inflight=1),
        persistence=dict(sink="local_artifact", output_dir="./expanded"),
        collection=dict(target_episodes=variants, max_attempts=2),
    )
    (directory / "expansion.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
