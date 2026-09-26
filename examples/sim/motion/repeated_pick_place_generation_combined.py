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

"""Preflight and persist the deterministic combined generation schedule.

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
    CombinedGenerationProfile,
    CombinedEpisodeCoordinator,
    CubeInitialPoseProvider,
    enumerate_candidate_recipes,
    load_generation_profile,
    PhysicalSlotPool,
    VisualProfileRegistry,
    round_robin_recipes,
    schedule_digest,
)

_DEFAULT_PROFILE = (
    _REPOSITORY_ROOT
    / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.combined.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument("--generation-profile", type=Path, default=_DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    return parser


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
    profile = load_generation_profile(args.generation_profile)
    if not isinstance(profile, CombinedGenerationProfile):
        raise ValueError(
            "the combined command requires schema_version: 1 and trajectory"
        )
    profile.validate_for_num_envs(args.num_envs)
    if args.task_config is not None and not args.task_config.is_file():
        raise FileNotFoundError(f"task config does not exist: {args.task_config}")
    visual_registry = VisualProfileRegistry.from_yaml(
        args.generation_profile.parent / profile.visual.profile_file,
    )
    if visual_registry.profile_ids != profile.visual.profiles:
        raise ValueError("visual registry IDs do not match the combined profile")
    profile.validate_registries(
        visual_profile_ids=visual_registry.profile_ids,
        observation_profile_ids=profile.observation.profiles,
    )
    families = CubeInitialPoseProvider().enumerate(
        profile.scene_randomization.reference_family_count
    )
    recipes = enumerate_candidate_recipes(profile, families=families)
    digest = schedule_digest(recipes)
    scheduled_recipes = round_robin_recipes(
        recipes,
        family_count=len(families),
    )
    coordinator = CombinedEpisodeCoordinator(
        recipes,
        slot_pool=PhysicalSlotPool(args.num_envs),
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
        "profile": str(args.generation_profile),
        "schema_version": profile.schema_version,
        "reference_family_count": len(families),
        "candidate_count": len(scheduled_recipes),
        "max_inflight": profile.execution.max_inflight,
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
        f"t={profile.trajectory.variants_per_family} n={args.num_envs}"
    )
    print(f"schedule digest: {digest}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
