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

"""Run configured Repeated Pick/Place with Generation Profile candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import gymnasium

from embodichain.cli.sim import add_seed_arg_to_parser, add_sim_args_to_parser
from embodichain.lab.gym.envs.demo import DemoEpisodeResult, execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim.motion.expansion import (
    TrajectoryGenerationJobCfg,
    load_generation_profile,
)
from embodichain.lab.task_program.integrations import TaskProgramGenerationRecord

_TASK_ROOT = (
    _REPOSITORY_ROOT
    / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place"
)
_DEFAULT_TASK_CONFIG = _TASK_ROOT / "task.franka.yaml"
_DEFAULT_GENERATION_PROFILE = _TASK_ROOT / "generation.demo.yaml"
_VIDEO_LOOK_AT = (
    (-1.25, -1.15, 0.95),
    (-0.25, -0.02, 0.25),
    (0.0, 0.0, 1.0),
)

__all__ = ["main"]


def _write_generation_json(
    profile: TrajectoryGenerationJobCfg,
    records: tuple[TaskProgramGenerationRecord, ...],
    result: DemoEpisodeResult,
    path: Path,
) -> None:
    """Write one JSON-safe profile, candidate, and Task Program result record."""
    payload = {
        "profile": profile.to_dict(),
        "generation": [record.to_metadata() for record in records],
        "episode": result.to_metadata(),
    }
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_generation_plot(
    records: tuple[TaskProgramGenerationRecord, ...],
    path: Path,
) -> None:
    """Plot every logical candidate and its phase boundaries by planning call."""
    if not records:
        raise RuntimeError("The generated episode produced no candidate records.")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        len(records),
        1,
        figsize=(11, max(4, 3 * len(records))),
        squeeze=False,
    )
    for call_index, record in enumerate(records):
        axis = axes[call_index, 0]
        for candidate, template in zip(
            record.candidates,
            record.templates,
            strict=True,
        ):
            times = template.dt.cumsum(0).detach().cpu().numpy()
            positions = template.positions.detach().cpu().numpy()
            selected = candidate.identity.candidate_id == record.selected_candidate_id
            alpha = 1.0 if selected else 0.35
            width = 2.0 if selected else 1.0
            label = candidate.trajectory_variant["spatial_operator"]
            for joint_index, joint_name in enumerate(template.joint_names):
                axis.plot(
                    times,
                    positions[:, joint_index],
                    alpha=alpha,
                    linewidth=width,
                    label=(f"{label}:{joint_name}" if joint_index == 0 else None),
                )
        selected_template = record.templates[record.candidate_index]
        selected_times = selected_template.dt.cumsum(0).detach().cpu().numpy()
        for phase in selected_template.phases:
            axis.axvline(
                selected_times[phase.start_index],
                color="black",
                alpha=0.2,
                linestyle="--",
            )
        axis.set_title(
            f"Call {record.workflow_call_index} · selected {record.candidate_index}"
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel("joint position")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _validate_candidate_indices(
    values: Iterable[int],
    *,
    profile: TrajectoryGenerationJobCfg,
) -> tuple[int, ...]:
    indices = tuple(values)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be nonempty and unique")
    limit = profile.augmentation.max_variants_per_reference
    if any(type(index) is not int or not 0 <= index < limit for index in indices):
        raise ValueError(f"candidate indices must be within [0, {limit})")
    return indices


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run configured Repeated Pick/Place trajectory candidates."
    )
    add_sim_args_to_parser(parser)
    add_seed_arg_to_parser(parser, default=None, scope="task environment")
    parser.add_argument(
        "--task-config",
        dest="gym_config",
        type=Path,
        default=_DEFAULT_TASK_CONFIG,
        help="Configured Task Program deployment.",
    )
    parser.add_argument(
        "--generation-profile",
        type=Path,
        default=_DEFAULT_GENERATION_PROFILE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-indices",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--save-video", action="store_true")
    parser.set_defaults(
        action_config=None,
        preview=False,
        filter_visual_rand=False,
        filter_dataset_saving=False,
        max_episodes=None,
        record_trajectory=False,
        trajectory_save_dir=None,
        profile=False,
        profile_output=None,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run selected candidates through the configured Task Program environment."""
    args = _create_parser().parse_args(argv)
    profile = load_generation_profile(args.generation_profile)
    candidate_indices = _validate_candidate_indices(
        args.candidate_indices,
        profile=profile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discover_task_packages()
    execute_init_hooks()
    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    try:
        for candidate_index in candidate_indices:
            env.reset(seed=args.seed, options={"save_data": False})
            recording_started = False
            try:
                if args.save_video:
                    recording_started = env.unwrapped.sim.start_window_record(
                        save_path=str(
                            args.output_dir / f"candidate_{candidate_index}.mp4"
                        ),
                        look_at=_VIDEO_LOOK_AT,
                        use_sim_time=True,
                    )
                    if not recording_started:
                        raise RuntimeError("Failed to start candidate video recording.")
                result = execute_demo_episode(
                    env,
                    episode_index=candidate_index,
                    generation_profile=profile,
                    generation_candidate_index=candidate_index,
                )
            finally:
                if recording_started:
                    if env.unwrapped.sim.is_window_recording():
                        env.unwrapped.sim.stop_window_record()
                    env.unwrapped.sim.wait_window_record_saves()
            records = env.unwrapped.task_program_generation_records
            _write_generation_json(
                profile,
                records,
                result,
                args.output_dir / f"candidate_{candidate_index}.json",
            )
            _save_generation_plot(
                records,
                args.output_dir / f"candidate_{candidate_index}.png",
            )
            if not result.completed:
                raise RuntimeError(
                    f"Candidate {candidate_index} stopped: {result.terminal_reason}"
                )
            env.reset(options={"save_data": False})
    finally:
        env.close()


if __name__ == "__main__":
    main()
