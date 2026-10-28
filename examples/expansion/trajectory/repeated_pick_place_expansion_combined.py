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

"""Preflight and persist the deterministic combined expansion schedule.

This command validates the profile and writes the schedule consumed by the
future physical host runner.  It intentionally reports a configured schedule,
not measured simulator success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from embodichain.lab.sim.motion.expansion import (
    CandidateRecipe,
    CombinedExpansionProfile,
    CombinedEpisodeCoordinator,
    CubeInitialPoseProvider,
    enumerate_candidate_recipes,
    load_expansion_profile,
    PhysicalSlotPool,
    VisualProfileRegistry,
    round_robin_recipes,
    schedule_digest,
)
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

_DEFAULT_TASK_CONFIG = resolve_config_path(
    "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml",
    base_dir=Path(__file__).parent,
)
_DEFAULT_OUTPUT_DIR = Path("output/repeated_pick_place_expansion_combined")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, default=_DEFAULT_TASK_CONFIG)
    parser.add_argument("--expansion-profile", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory. Default: {_DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Batch width; omission uses the profile's execution.max_inflight.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    return parser


def _preflight_augmentation(profile: CombinedExpansionProfile) -> None:
    """Reject a profile whose amplitude cannot produce a distinct variant.

    The host runner builds its augmentation settings from this profile, and the
    contract refuses an enabled spatial factor whose offset scale does not
    exceed the deduplication tolerance. Preflighting the very same settings
    reports that configuration error before a simulator is ever constructed.
    """
    from embodichain.lab.sim.motion.expansion import TrajectoryExpansionCfg
    from embodichain.lab.task_program.integrations.expansion import (
        _combined_augmentation_payload,
    )

    cfg = TrajectoryExpansionCfg.from_mapping(
        {
            "augmentation": _combined_augmentation_payload(profile, 0),
            "affordance": {"enabled": False},
            "scheduling": {"candidate_budget": profile.trajectory.variants_per_family},
            "execution": {"max_inflight": 1},
            "observation": {"enabled": False, "profiles": []},
        }
    )
    spatial = cfg.augmentation.factors.spatial
    print(
        f"trajectory augmentation: operators={list(spatial.method)} "
        f"scale={spatial.joint_offset_scale} via_count={spatial.via_count} "
        f"dedup_tolerance={cfg.augmentation.coverage.joint_dedup_normalized_tol}"
    )


def _record(recipe: CandidateRecipe) -> dict[str, object]:
    return {
        "recipe_index": recipe.recipe_index,
        "candidate_id": recipe.candidate_id,
        "reference_family_id": recipe.reference_family_id,
        "physical_geometry_family_id": recipe.physical_geometry_family_id,
        "cycle_schedule": [
            {
                "cycle_index": cycle.cycle_index,
                "affordance_requested": cycle.affordance_requested,
                "trajectory_requested": cycle.trajectory_requested,
            }
            for cycle in recipe.cycle_schedule
        ],
        "visual_profile_id": recipe.visual_profile_id,
        "visual_seed": recipe.visual_seed,
        "source_unit_id": recipe.source_unit_id,
        "source_revision": recipe.source_revision,
        "deterministic_seed_digest": recipe.deterministic_seed_digest,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.expansion_profile is None:
        from embodichain.lab.scripts.run_task import (
            _resolve_expansion_request,
        )

        if not args.task_config.is_file():
            raise FileNotFoundError(f"task config does not exist: {args.task_config}")
        task_path = resolve_config_path(args.task_config)
        # The task-bound resolver reads its own launcher namespace, so expose the
        # task path under the attribute name it expects. It returns the profile,
        # the selected logical recipes, the profile anchor, and the collection
        # section.
        args.gym_config = str(task_path)
        request = _resolve_expansion_request(args, load_config(task_path))
        if request is None:
            raise ValueError(
                "task config must bind a expansion policy when "
                "--expansion-profile is omitted"
            )
        profile, _, profile_anchor, _ = request
        profile_source = str(profile_anchor)
    else:
        if not args.expansion_profile.is_file():
            raise FileNotFoundError(
                f"expansion profile does not exist: {args.expansion_profile}"
            )
        profile = load_expansion_profile(args.expansion_profile)
        profile_source = str(args.expansion_profile)
    if not isinstance(profile, CombinedExpansionProfile):
        raise ValueError(
            "the combined command requires schema_version: 1 and trajectory"
        )
    _preflight_augmentation(profile)
    num_envs = (
        profile.execution.max_inflight if args.num_envs is None else args.num_envs
    )
    profile.validate_for_num_envs(num_envs)
    visual_registry = (
        VisualProfileRegistry.from_yaml(
            resolve_config_path(profile.visual.profile_file),
        )
        if profile.visual.enabled
        else None
    )
    if visual_registry is not None:
        if visual_registry.profile_ids != profile.visual.profiles:
            raise ValueError("visual registry IDs do not match the combined profile")
        profile.validate_registries(
            visual_profile_ids=visual_registry.profile_ids,
            observation_profile_ids=profile.observation.profiles,
        )
    families = CubeInitialPoseProvider.from_yaml(
        resolve_config_path(profile.scene_randomization.profile_file)
    ).enumerate(profile.scene_randomization.reference_family_count)
    recipes = enumerate_candidate_recipes(profile, families=families)
    digest = schedule_digest(recipes)
    scheduled_recipes = round_robin_recipes(
        recipes,
        family_count=len(families),
    )
    coordinator = CombinedEpisodeCoordinator(
        recipes,
        slot_pool=PhysicalSlotPool(num_envs),
        family_count=len(families),
        compatibility_key="strict:configured-preflight",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "schedule.jsonl"
    manifest_path = output_dir / "manifest.json"
    schedule_tmp = schedule_path.with_suffix(".jsonl.tmp")
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    schedule_tmp.write_text(
        "".join(
            json.dumps(_record(recipe), sort_keys=True) + "\n"
            for recipe in scheduled_recipes
        ),
        encoding="utf-8",
    )
    schedule_tmp.replace(schedule_path)
    manifest = {
        "status": "configured_preflight",
        "measured_success": False,
        "profile": profile_source,
        "schema_version": profile.schema_version,
        "reference_family_count": len(families),
        "candidate_count": len(scheduled_recipes),
        "max_inflight": num_envs,
        "visual_profiles": list(visual_registry.profile_ids),
        "observation_profiles": list(profile.observation.profiles),
        "schedule_digest": digest,
        "accepted": 0,
        "rejected": 0,
        "write_failed": 0,
        "lerobot_commit": False,
        "scheduler": dict(coordinator.snapshot()),
    }
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_tmp.replace(manifest_path)
    print(
        f"resolved m={len(families)} a={profile.affordance.branches_per_family} "
        f"t={profile.trajectory.variants_per_family} n={num_envs}"
    )
    print(f"schedule digest: {digest}")
    print(f"schedule: {schedule_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
