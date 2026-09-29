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

import argparse
from dataclasses import dataclass
import json
import os
import select
import sys
import time

from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
import tqdm

from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    DemoExecutionCfg,
    execute_demo_episode,
)
from embodichain.lab.gym.envs.wrapper import ReplayWrapper
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
    load_trajectory,
    _load_expansion_declaration,
)
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.utils.logger import (
    decorate_str_color,
    log_warning,
    log_info,
    log_error,
)

if TYPE_CHECKING:
    from embodichain.lab.visualization import VisualizationRuntime

_REPLAY_CONTROL_POLL_INTERVAL = 0.05
_ACTIVITY_BAR_WIDTH = 10


@dataclass(frozen=True)
class CollectionSelection:
    """Select logical recipes without encoding batch boundaries."""

    mode: str = "sequential"
    start_recipe_index: int = 0
    recipe_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"sequential", "explicit"}:
            raise ValueError("collection.selection.mode must be sequential or explicit")
        if type(self.start_recipe_index) is not int or self.start_recipe_index < 0:
            raise ValueError("start_recipe_index must be a non-negative integer")
        indices = tuple(self.recipe_indices)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("recipe_indices must contain non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("recipe_indices must be unique")
        if self.mode == "explicit" and not indices:
            raise ValueError("explicit selection requires recipe_indices")
        if self.mode == "sequential" and indices:
            raise ValueError("sequential selection cannot declare recipe_indices")
        object.__setattr__(self, "recipe_indices", indices)

    def take(self, cursor: int, count: int) -> tuple[int, ...]:
        """Return the next logical recipe IDs for one batch."""
        if type(cursor) is not int or cursor < 0:
            raise ValueError("selection cursor must be a non-negative integer")
        if type(count) is not int or count < 0:
            raise ValueError("selection count must be a non-negative integer")
        if self.mode == "explicit":
            selected = self.recipe_indices[cursor : cursor + count]
            if len(selected) != count:
                raise ValueError(
                    "explicit recipe selection is shorter than target_episodes"
                )
            return selected
        return tuple(
            self.start_recipe_index + cursor + offset for offset in range(count)
        )


@dataclass(frozen=True)
class CollectionPlan:
    """Unified target, retry, and logical recipe selection policy."""

    target_episodes: int
    max_attempts: int = 3
    selection: CollectionSelection = CollectionSelection()

    def __post_init__(self) -> None:
        if type(self.target_episodes) is not int or self.target_episodes < 0:
            raise ValueError("collection.target_episodes must be non-negative")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("collection.max_attempts must be at least 1")
        if not isinstance(self.selection, CollectionSelection):
            raise TypeError("collection.selection must be a CollectionSelection")
        if (
            self.selection.mode == "explicit"
            and len(self.selection.recipe_indices) != self.target_episodes
        ):
            raise ValueError("explicit recipe count must equal target_episodes")

    def batch_count(self, num_envs: int) -> int:
        """Return the number of batches needed at a given parallel width."""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        return (self.target_episodes + num_envs - 1) // num_envs


def _mapping_or_empty(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def resolve_collection_plan(
    args: Any,
    gym_config: Mapping[str, Any],
    *,
    expansion_collection: Mapping[str, Any] | None = None,
    legacy_recipe_indices: Sequence[int] = (),
) -> CollectionPlan:
    """Resolve the common collection policy for ordinary and Expansion runs."""
    task_collection = _mapping_or_empty(gym_config.get("collection"), "collection")
    expansion = _mapping_or_empty(expansion_collection, "expansion.collection")
    cli_target = getattr(args, "max_episodes", None)
    target_values = [
        value
        for value in (
            task_collection.get("target_episodes"),
            expansion.get("target_episodes"),
        )
        if value is not None
    ]
    legacy_target = gym_config.get("max_episodes")
    if cli_target is not None:
        target = cli_target
        if any(value != target for value in target_values):
            raise ValueError(
                "CLI --max-episodes conflicts with collection.target_episodes"
            )
    elif target_values:
        if len(set(target_values)) != 1:
            raise ValueError("collection.target_episodes is declared more than once")
        target = target_values[0]
    else:
        target = 1 if legacy_target is None else legacy_target
    attempts = expansion.get(
        "max_attempts",
        task_collection.get("max_attempts", gym_config.get("demo_max_attempts", 3)),
    )
    selection_data = _mapping_or_empty(
        expansion.get("selection", task_collection.get("selection", {})),
        "collection.selection",
    )
    legacy_indices = tuple(legacy_recipe_indices)
    declared_indices = selection_data.get("recipe_indices")
    if declared_indices is not None and legacy_indices:
        if tuple(declared_indices) != legacy_indices:
            raise ValueError(
                "collection.selection.recipe_indices conflicts with legacy candidate_indices"
            )
    if selection_data or legacy_indices:
        selection = CollectionSelection(
            mode=selection_data.get(
                "mode", "explicit" if legacy_indices else "sequential"
            ),
            start_recipe_index=selection_data.get("start_recipe_index", 0),
            recipe_indices=tuple(selection_data.get("recipe_indices", legacy_indices)),
        )
    else:
        selection = CollectionSelection()
    return CollectionPlan(int(target), int(attempts), selection)


def _indeterminate_progress(actions: Iterable[Any], description: str) -> Iterator[Any]:
    """Render an activity bar for an action stream with no exact length."""
    mode = decorate_str_color("[dynamic]", "yellow")

    def label(position: int, *, done: bool = False) -> str:
        if done:
            cells = "━" * _ACTIVITY_BAR_WIDTH
        else:
            cells_list = ["·"] * _ACTIVITY_BAR_WIDTH
            cells_list[position] = "╺"
            cells = "".join(cells_list)
        bar = decorate_str_color(f"│{cells}│", "cyan")
        completion = " ✓" if done else ""
        return f"Steps  {mode}  {description} {bar}{completion}"

    progress = tqdm.tqdm(
        total=None,
        desc=label(0),
        unit=" step",
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format="{desc} {n_fmt} steps [{elapsed}, {rate_fmt}]",
    )
    try:
        for action in actions:
            yield action
            progress.update()
            cycle_position = progress.n % (2 * _ACTIVITY_BAR_WIDTH - 2)
            position = min(cycle_position, 2 * _ACTIVITY_BAR_WIDTH - 2 - cycle_position)
            progress.set_description_str(label(position), refresh=False)
        progress.set_description_str(label(0, done=True), refresh=False)
    finally:
        progress.close()


def _progress_wrapper(actions: Iterable[Any], description: str) -> Iterable[Any]:
    """Wrap a segment action iterable in a visible terminal progress bar."""
    total = len(actions) if isinstance(actions, Sized) else None
    if total is None:
        return _indeterminate_progress(actions, description)
    mode = decorate_str_color("[fixed]", "blue")
    return tqdm.tqdm(
        actions,
        total=total,
        desc=description,
        unit=" step",
        file=sys.stdout,
        dynamic_ncols=True,
        colour="cyan",
        bar_format=(
            f"Steps  {mode}    {{desc}} {{percentage:3.0f}}%│{{bar}}│ "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )


def _env_target(env: Any) -> Any:
    """Return the underlying environment used for lifecycle introspection."""
    return getattr(env, "unwrapped", env)


def _normalize_save_env_ids(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[int, ...]:
    """Validate environment rows selected for one persisted episode batch."""
    num_envs = int(getattr(_env_target(env), "num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be at least 1, got {num_envs}.")

    if env_ids is None:
        normalized = tuple(range(num_envs))
    elif isinstance(env_ids, torch.Tensor):
        normalized = tuple(
            int(env_id) for env_id in env_ids.detach().cpu().reshape(-1).tolist()
        )
    else:
        normalized = tuple(int(env_id) for env_id in env_ids)

    if not normalized:
        raise ValueError("save_env_ids must select at least one environment.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"save_env_ids contains duplicates: {normalized}.")
    invalid = [env_id for env_id in normalized if not 0 <= env_id < num_envs]
    if invalid:
        raise ValueError(
            f"save_env_ids {invalid} are outside the valid range [0, {num_envs})."
        )
    return normalized


def _reset_episode_rows(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    save_data: bool,
) -> None:
    """Reset selected rows, committing or discarding their pending recordings."""
    selected = _normalize_save_env_ids(env, env_ids)
    target = _env_target(env)
    num_envs = int(getattr(target, "num_envs", 1))
    all_env_ids = tuple(range(num_envs))

    if selected == all_env_ids:
        if save_data:
            env.reset()
        else:
            env.reset(options={"save_data": False})
        return

    reset_ids = torch.tensor(
        selected,
        dtype=torch.int32,
        device=getattr(target, "device", None),
    )
    options: dict[str, Any] = {"reset_ids": reset_ids}
    if not save_data:
        options["save_data"] = False
    env.reset(options=options)


def _abort_pending_episode(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Discard buffered data before retrying or closing an environment."""
    stats = getattr(_env_target(env), "_collection_stats", None)
    if isinstance(stats, dict):
        stats["discard_reset_count"] = stats.get("discard_reset_count", 0) + 1
    _reset_episode_rows(env, env_ids, save_data=False)


def _commit_pending_episode(
    env: Any,
    save_env_ids: Sequence[int] | torch.Tensor | None,
) -> None:
    """Commit selected dataset rows through one reset of the full vector batch."""
    stats = getattr(_env_target(env), "_collection_stats", None)
    if isinstance(stats, dict):
        stats["commit_reset_count"] = stats.get("commit_reset_count", 0) + 1
    selected = _normalize_save_env_ids(env, save_env_ids)
    target = _env_target(env)
    all_env_ids = tuple(range(int(getattr(target, "num_envs", 1))))
    if selected == all_env_ids:
        _reset_episode_rows(env, selected, save_data=True)
        return

    commit_env_ids = torch.tensor(
        selected,
        dtype=torch.int32,
        device=getattr(target, "device", None),
    )
    env.reset(options={"save_data": False, "commit_env_ids": commit_env_ids})


def _save_failed_episodes_enabled(env: Any) -> bool:
    """Return whether the configured dataset manager keeps failed episodes."""
    dataset_manager = getattr(_env_target(env), "dataset_manager", None)
    return bool(
        dataset_manager is not None
        and getattr(dataset_manager, "save_failed_episodes", False)
    )


def _selected_rows_have_frames(
    result: DemoEpisodeResult,
    save_env_ids: Sequence[int],
) -> bool:
    """Return whether every selected row contains a persistable frame."""
    if result.lengths:
        return all(result.lengths[env_id] > 0 for env_id in save_env_ids)
    return result.length > 0


def _persistable_fragment_env_ids(
    env: Any,
    result: DemoEpisodeResult,
    save_env_ids: Sequence[int],
    *,
    include_failed: bool,
) -> tuple[int, ...]:
    """Select rows containing at least one eligible non-empty segment span."""
    persistable: list[int] = []
    metadata_getter = getattr(_env_target(env), "get_demo_episode_metadata", None)
    for env_id in save_env_ids:
        if callable(metadata_getter):
            metadata = metadata_getter(env_id)
            if isinstance(metadata, dict):
                eligible = any(
                    int(segment.get("end_step", 0)) > int(segment.get("start_step", 0))
                    and (bool(segment.get("success", False)) or include_failed)
                    for segment in metadata.get("segments", [])
                    if isinstance(segment, dict)
                )
                if eligible:
                    persistable.append(env_id)
                continue
        for segment in result.segments:
            if segment.active and not segment.active[env_id]:
                continue
            start = (
                segment.start_steps[env_id]
                if segment.start_steps
                else segment.start_step
            )
            end = segment.end_steps[env_id] if segment.end_steps else segment.end_step
            accepted = (
                segment.successes[env_id] if segment.successes else segment.success
            )
            if end > start and (accepted or include_failed):
                persistable.append(env_id)
                break
    return tuple(persistable)


def generate_and_execute_action_list(
    env: gymnasium.Env,
    idx: int,
    debug_mode: bool,
    *,
    episode_idx: int = 0,
    **kwargs: Any,
) -> bool:
    """Execute one legacy planner result through the common episode executor.

    This compatibility helper now represents one complete task episode. New
    multi-object tasks should implement ``create_demo_segments`` instead of
    calling this function repeatedly.

    Args:
        env: Environment used to generate and execute the actions.
        idx: Index of the legacy action list within the current episode.
        debug_mode: Whether debug mode is enabled.
        episode_idx: Index of the current episode.
        **kwargs: Additional arguments forwarded to action expansion.

    Returns:
        Whether a complete, successful episode was executed.
    """
    result = execute_demo_episode(
        env,
        episode_index=episode_idx,
        progress=_progress_wrapper,
        action_sentence=idx,
        **kwargs,
    )
    if not result.completed or not result.all_success:
        log_warning(
            f"Demo episode {episode_idx} is invalid ({result.terminal_reason}); "
            "it will not be saved."
        )
        return False
    return True


def generate_function(
    env: Any,
    num_traj: int | None = None,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    save_env_ids: Sequence[int] | torch.Tensor | None = None,
    execution_cfg: DemoExecutionCfg | None = None,
    **kwargs: Any,
) -> bool:
    """Generate, execute, and commit one demonstration collection batch.

    A task owns its segment count through ``create_demo_segments``. The legacy
    ``num_traj`` parameter is accepted only as ``None`` or ``1`` so callers do
    not accidentally repeat a one-grasp planner inside the same episode. When
    a dataset functor enables ``save_failed_episodes``, a failed result with at
    least one frame in every selected row is committed instead of retried.
    Continuous mode has one reset commit boundary; fragment mode delegates
    independent idempotent fragment commits to the dataset recorder.

    Args:
        env: The environment instance.
        num_traj: Deprecated compatibility value. Must be ``None`` or ``1``.
        time_id (int, optional): Identifier for the current time step or episode.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        save_env_ids: Environment rows to persist from this vector batch. Other
            rows are explicitly discarded after the selected rows commit.
        execution_cfg: Continuous or independent segment-fragment persistence
            settings. Checkpoint resume is not performed in either mode.
        **kwargs: Additional keyword arguments for data expansion.

    Returns:
        True if continuous episodes, or at least one eligible fragment row,
        were committed. With ``save_failed_episodes`` enabled, committed
        continuous episodes may be unsuccessful.
    """
    if num_traj not in (None, 1):
        raise ValueError(
            "num_traj no longer controls sub-trajectories. Implement "
            "create_demo_segments() in the task to define multiple segments."
        )

    max_attempts = int(kwargs.pop("max_attempts", 3))
    reset_before = bool(kwargs.pop("reset_before", True))
    expansion_record_sink = kwargs.pop("_expansion_record_sink", None)
    if expansion_record_sink is not None and not callable(expansion_record_sink):
        raise TypeError("_expansion_record_sink must be callable or None")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}.")
    normalized_save_env_ids = _normalize_save_env_ids(env, save_env_ids)
    save_failed_episodes = _save_failed_episodes_enabled(env)
    if execution_cfg is None:
        execution_cfg = DemoExecutionCfg()
    elif not isinstance(execution_cfg, DemoExecutionCfg):
        raise TypeError("execution_cfg must be a DemoExecutionCfg or None.")

    if reset_before:
        _abort_pending_episode(env)

    for attempt in range(1, max_attempts + 1):
        stats = getattr(_env_target(env), "_collection_stats", None)
        if isinstance(stats, dict):
            stats["attempts"] = stats.get("attempts", 0) + 1
        commit_succeeded = False
        try:
            result: DemoEpisodeResult = execute_demo_episode(
                env,
                episode_index=time_id,
                execution_cfg=execution_cfg,
                attempt_id=attempt - 1,
                progress=_progress_wrapper,
                **kwargs,
            )
            successful = result.completed and result.all_success
            fragment_env_ids = _persistable_fragment_env_ids(
                env,
                result,
                normalized_save_env_ids,
                include_failed=execution_cfg.save_failed_fragments,
            )
            persistable_failure = (
                not successful
                and save_failed_episodes
                and _selected_rows_have_frames(result, normalized_save_env_ids)
            )
            if execution_cfg.mode == "segment_fragments" and fragment_env_ids:
                if expansion_record_sink is not None:
                    expansion_record_sink(
                        tuple(
                            getattr(
                                _env_target(env), "task_program_expansion_records", ()
                            )
                        )
                    )
                _commit_pending_episode(env, fragment_env_ids)
                commit_succeeded = True
                if not successful:
                    log_warning(
                        f"Program run {time_id} stopped ({result.terminal_reason}); "
                        f"saved eligible segments from env rows {fragment_env_ids}."
                    )
                return True
            if execution_cfg.mode == "continuous" and (
                successful or persistable_failure
            ):
                if expansion_record_sink is not None:
                    expansion_record_sink(
                        tuple(
                            getattr(
                                _env_target(env), "task_program_expansion_records", ()
                            )
                        )
                    )
                # reset() is the commit boundary: dataset functors consume the
                # whole episode once, then buffers and scene state are reset.
                _commit_pending_episode(env, normalized_save_env_ids)
                commit_succeeded = True
                if persistable_failure:
                    log_warning(
                        f"Episode {time_id} failed ({result.terminal_reason}) but "
                        "was saved because save_failed_episodes is enabled."
                    )
                return True
        finally:
            # ``finally`` also covers KeyboardInterrupt, SystemExit, and
            # GeneratorExit. A failed commit is aborted as well, so close()
            # can never implicitly persist the pending partial episode.
            if not commit_succeeded:
                _abort_pending_episode(env)

        log_warning(
            f"Episode {time_id} attempt {attempt}/{max_attempts} failed: "
            f"{result.terminal_reason}. Discarding {result.length} frames."
        )
        if debug_mode:
            log_warning(
                "Failed demo trace: "
                + json.dumps(
                    result.to_metadata(),
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

    return False


def _configure_expansion_reset_event(
    env: Any,
    params: Mapping[str, Any],
) -> None:
    """Install one expansion batch payload in the reset lifecycle."""
    from embodichain.lab.gym.envs.managers.cfg import EventCfg

    target = _env_target(env)
    event_manager = getattr(target, "event_manager", None)
    if event_manager is None:
        raise RuntimeError("expansion requires an event manager")
    active_functors = event_manager.active_functors
    if "expansion_profile_reset" not in active_functors.get("reset", ()):
        raise RuntimeError(
            "expansion profile requires the expansion_profile_reset reset event"
        )
    current_cfg = event_manager.get_functor_cfg("expansion_profile_reset")
    event_manager.set_functor_cfg(
        "expansion_profile_reset",
        EventCfg(func=current_cfg.func, mode="reset", params=dict(params)),
    )


def _resolve_expansion_resource(anchor: Path, value: str) -> Path:
    """Resolve a task-local or packaged randomization resource path."""
    from embodichain.utils.config_paths import resolve_config_path

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts[:2] == ("embodichain_tasks", "configs"):
        return resolve_config_path(path)
    return (anchor.parent / path).resolve()


def _load_expansion_collection(path: Path) -> Mapping[str, Any]:
    """Load the optional collection section from an expansion declaration."""
    from embodichain.utils.utility import load_config

    data = load_config(path)
    return data.get("collection", {}) if isinstance(data, Mapping) else {}


def _resolve_expansion_request(
    args: Any,
    gym_config: Mapping[str, Any],
) -> tuple[Any, tuple[int, ...], Path] | None:
    """Resolve a task-bound or CLI-selected Expansion Profile."""
    from embodichain.utils.config_paths import resolve_config_path
    from embodichain.utils.utility import load_config
    from embodichain.lab.sim.motion.expansion import (
        CombinedExpansionProfile,
        load_expansion_profile,
    )

    task_path = resolve_config_path(args.gym_config)
    binding = gym_config.get("expansion")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ValueError("expansion must be a mapping")
        binding, expansion_config_path = _load_expansion_declaration(
            binding,
            base_dir=task_path.parent,
        )
        unknown = set(binding) - {
            "profile",
            "policy",
            "overrides",
            "runtime",
            "collection",
            "selection",
            "candidate_indices",
        }
        if unknown:
            raise ValueError(
                f"expansion contains unsupported fields: {sorted(unknown)}"
            )
    cli_profile = getattr(args, "expansion_profile", None)
    binding_profile = None if binding is None else binding.get("profile")
    binding_policy = None if binding is None else binding.get("policy")
    if binding_profile is not None and binding_policy is not None:
        raise ValueError("expansion may select profile or policy, not both")
    profile_value = cli_profile if cli_profile is not None else binding_profile
    use_binding_policy = binding_policy is not None and cli_profile is None
    if profile_value is None and not use_binding_policy:
        if getattr(args, "expansion_recipe_indices", None) is not None:
            raise ValueError("expansion candidate indices require a profile")
        return None
    if use_binding_policy:
        if not isinstance(binding_policy, Mapping):
            raise ValueError("expansion.policy must be a mapping")
        policy_path_value = binding_policy.get("component")
        if (
            not isinstance(policy_path_value, str)
            or not policy_path_value.strip()
            or policy_path_value != policy_path_value.strip()
        ):
            raise ValueError("expansion.policy.component must be a nonempty path")
        policy_path = Path(policy_path_value).expanduser()
        if not policy_path.is_absolute():
            policy_base_dir = (
                expansion_config_path.parent
                if expansion_config_path is not None
                else task_path.parent
            )
            policy_path = policy_base_dir / policy_path
        policy_path = policy_path.resolve()
        policy_data = load_config(policy_path)
        overrides = binding.get("overrides", {}) if binding is not None else {}
        if not isinstance(overrides, Mapping):
            raise ValueError("expansion.overrides must be a mapping")

        def merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
            merged = dict(base)
            for key, value in patch.items():
                if isinstance(value, Mapping):
                    current = merged.get(key, {})
                    if not isinstance(current, Mapping):
                        raise ValueError(
                            f"expansion.overrides.{key} cannot replace a scalar"
                        )
                    merged[key] = merge(current, value)
                else:
                    merged[key] = value
            return merged

        if not isinstance(policy_data, Mapping):
            raise ValueError("expansion policy component must be a mapping")
        profile_data = dict(policy_data)
        profile_data.pop("collection", None)
        profile = CombinedExpansionProfile.from_mapping(merge(profile_data, overrides))
        profile_path = expansion_config_path or task_path
    else:
        if (
            not isinstance(profile_value, str)
            or not profile_value.strip()
            or profile_value != profile_value.strip()
        ):
            raise ValueError("expansion.profile must be a nonempty path")
        profile_path = Path(profile_value).expanduser()
        if not profile_path.is_absolute():
            profile_path = (
                Path.cwd() / profile_path
                if cli_profile is not None
                else (
                    expansion_config_path.parent
                    if expansion_config_path is not None
                    else task_path.parent
                )
                / profile_path
            )
        profile_path = profile_path.resolve()
        profile = load_expansion_profile(profile_path)
    candidate_values = getattr(args, "expansion_recipe_indices", None)
    if candidate_values is None:
        candidate_values = getattr(args, "expansion_candidate_indices", None)
    if candidate_values is None and binding is not None:
        candidate_values = binding.get("candidate_indices")
        if candidate_values is None:
            selection = binding.get("selection")
            if isinstance(selection, Mapping):
                candidate_values = selection.get("recipe_indices")
    if candidate_values is None:
        candidate_indices: tuple[int, ...] = ()
    else:
        if not isinstance(candidate_values, (list, tuple)):
            raise ValueError("expansion.selection.recipe_indices must be a sequence")
        candidate_indices = tuple(candidate_values)
        if not candidate_indices or len(set(candidate_indices)) != len(
            candidate_indices
        ):
            raise ValueError("expansion candidate indices must be nonempty and unique")
        if any(type(index) is not int or index < 0 for index in candidate_indices):
            raise ValueError(
                "expansion candidate indices must be non-negative integers"
            )
    return profile, candidate_indices, profile_path


def _validate_expansion_collection_plan(
    profile: Any,
    plan: CollectionPlan,
    *,
    num_envs: int,
) -> None:
    """Validate collection capacity and logical recipe selection before launch."""
    from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile

    if not isinstance(profile, CombinedExpansionProfile):
        raise ValueError(
            "run-task expansion requires a CombinedExpansionProfile; "
            "direct source profiles need a source-specific provider host"
        )
    if num_envs < 1:
        raise ValueError("expansion requires at least one environment")
    profile.validate_for_num_envs(num_envs)
    total = (
        profile.scene_randomization.reference_family_count
        * profile.affordance.branches_per_family
        * profile.trajectory.variants_per_family
    )
    if plan.selection.mode == "sequential":
        if plan.selection.start_recipe_index + plan.target_episodes > total:
            raise ValueError("collection selection exceeds the expansion recipe budget")
    elif any(index >= total for index in plan.selection.recipe_indices):
        raise ValueError("collection selection contains an out-of-range recipe")


def _run_expansion(
    env: Any,
    args: Any,
    gym_config: Mapping[str, Any],
) -> None:
    """Run expansion candidates through the shared episode collection plan."""
    from embodichain.lab.sim.motion.expansion import (
        CombinedExpansionProfile,
        CubeInitialPoseProvider,
        VisualProfileRegistry,
        enumerate_candidate_recipes,
    )

    request = _resolve_expansion_request(args, gym_config)
    if request is None:
        raise ValueError("run_expansion requires an expansion profile")
    profile, configured_indices, profile_path = request
    if not isinstance(profile, CombinedExpansionProfile):
        raise ValueError("run-task expansion requires a CombinedExpansionProfile")
    target = _env_target(env)
    num_envs = int(getattr(target, "num_envs", 1))
    profile.validate_for_num_envs(num_envs)
    plan = resolve_collection_plan(
        args,
        gym_config,
        expansion_collection=_load_expansion_collection(profile_path),
        legacy_recipe_indices=configured_indices,
    )
    _validate_expansion_collection_plan(profile, plan, num_envs=num_envs)
    if getattr(args, "disable_sensor", False) and profile.visual.enabled:
        raise ValueError(
            "--disable-sensor cannot be used with a combined profile that "
            "contains visual expansion"
        )

    visual_registry = (
        VisualProfileRegistry.from_yaml(
            _resolve_expansion_resource(profile_path, profile.visual.profile_file)
        )
        if profile.visual.enabled
        else None
    )
    if profile.scene_randomization.enabled:
        family_path = _resolve_expansion_resource(
            profile_path, profile.scene_randomization.profile_file
        )
        families = CubeInitialPoseProvider.from_yaml(family_path).enumerate(
            profile.scene_randomization.reference_family_count
        )
    else:
        families = CubeInitialPoseProvider().enumerate(
            profile.scene_randomization.reference_family_count
        )
    recipes = enumerate_candidate_recipes(profile, families=families)
    if plan.selection.mode == "sequential":
        if plan.selection.start_recipe_index + plan.target_episodes > len(recipes):
            raise ValueError("collection selection exceeds the expansion recipe budget")
    elif any(index >= len(recipes) for index in plan.selection.recipe_indices):
        raise ValueError("collection selection contains an out-of-range recipe")

    output_dir = getattr(args, "expansion_output_dir", None)
    if output_dir is None:
        output_dir = profile.persistence.output_dir
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "target_episodes": plan.target_episodes,
        "planned_episodes": plan.target_episodes,
        "committed_episodes": 0,
        "rejected_episodes": 0,
        "attempts": 0,
        "batch_count": 0,
        "prepare_reset_count": 0,
        "commit_reset_count": 0,
        "discard_reset_count": 0,
    }
    setattr(target, "_collection_stats", stats)
    manifest: dict[str, Any] = {
        "profile": str(profile_path),
        "target_episodes": plan.target_episodes,
        "planned_episodes": plan.target_episodes,
        "selection": {
            "mode": plan.selection.mode,
            "start_recipe_index": plan.selection.start_recipe_index,
            "recipe_indices": list(plan.selection.recipe_indices),
        },
        "num_envs": num_envs,
        "batches": [],
    }
    committed = 0
    cursor = 0
    while committed < plan.target_episodes:
        batch_size = min(num_envs, plan.target_episodes - committed)
        selected = plan.selection.take(cursor, batch_size)
        if any(index >= len(recipes) for index in selected):
            raise ValueError("collection selection contains an out-of-range recipe")
        batch_recipes = tuple(recipes[index] for index in selected)
        reset_payload: dict[str, Any] = {}
        family_by_id = {family.reference_family_id: family for family in families}
        if profile.scene_randomization.enabled:
            reset_payload["cube_pose"] = torch.tensor(
                [
                    [
                        *family_by_id[recipe.reference_family_id].cube_position,
                        *family_by_id[recipe.reference_family_id].cube_quaternion_xyzw,
                    ]
                    for recipe in batch_recipes
                ],
                dtype=torch.float32,
                device=target.device,
            )
        if profile.visual.enabled:
            reset_payload.update(
                {
                    "visual_registry": visual_registry,
                    "visual_assignments": {
                        row: recipe.visual_profile_id
                        for row, recipe in enumerate(batch_recipes)
                    },
                    "visual_seed": (args.seed if args.seed is not None else 7) + cursor,
                }
            )
        if reset_payload:
            _configure_expansion_reset_event(env, reset_payload)
        stats["prepare_reset_count"] += 1
        env.reset(seed=args.seed, options={"save_data": False})
        batch_records: list[Any] = []
        generated = generate_function(
            env,
            time_id=cursor,
            save_path=str(output_path),
            save_video=getattr(args, "expansion_save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            save_env_ids=tuple(range(batch_size)),
            max_attempts=plan.max_attempts,
            reset_before=False,
            expansion_profile=profile,
            expansion_candidate_index=selected[0] if selected else 0,
            expansion_recipe_indices=selected,
            _expansion_record_sink=batch_records.extend,
        )
        stats["batch_count"] += 1
        batch_record = {
            "recipe_indices": list(selected),
            "status": "accepted" if generated else "rejected",
            "records": [record.to_metadata() for record in batch_records],
        }
        manifest["batches"].append(batch_record)
        if not generated:
            stats["rejected_episodes"] += batch_size
            manifest.update(stats)
            (output_path / "expansion_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            raise RuntimeError(
                f"Failed to collect expansion batch at recipe cursor {cursor} "
                f"after {plan.max_attempts} attempts"
            )
        committed += batch_size
        cursor += batch_size
        stats["committed_episodes"] = committed
    manifest.update(stats)
    (output_path / "expansion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def replay(env, trajectory_path: str, mode: str = "kinematic") -> None:
    """Replay a recorded trajectory.

    The caller retains ownership of ``env``. Wrapper-specific replay state is
    restored before returning, but the environment is not closed here.

    Args:
        env: The environment built from the same config that recorded the
            trajectory (wrapped via :class:`ReplayWrapper`).
        trajectory_path: Path to the ``.pt`` trajectory file.
        mode: ``"kinematic"`` (exact, physics off), ``"dynamic"`` (feed recorded
            actions, physics on), or ``"control"`` (interactive kinematic scrubber).
    """
    data = load_trajectory(trajectory_path)
    meta = data["meta"]
    lengths = meta["lengths"]
    log_info(
        f"Replaying trajectory: num_envs={meta['num_envs']}, lengths={lengths}, "
        f"num_steps={meta['num_steps']}, mode={mode}",
        color="green",
    )
    replay_env = ReplayWrapper(env.unwrapped, trajectory_path, mode=mode)
    try:
        if mode == "control":
            replay_control(replay_env)
        else:
            replay_auto(replay_env, mode)
    finally:
        # ReplayWrapper.close() also closes its wrapped environment. Restore
        # its local state here and leave the single close() to cli().
        try:
            replay_env.env.sim.enable_physics(True)
        finally:
            replay_env.env._replay_no_auto_reset = False


def replay_auto(replay_env: ReplayWrapper, mode: str) -> None:
    """Auto-replay the full trajectory with a progress bar."""
    num_steps = int(replay_env._lengths.min().item())
    replay_env.reset()
    max_err = 0.0
    rec_states = replay_env._trajectory["states"]
    for i in tqdm.tqdm(range(num_steps), desc=f"Replaying ({mode})", unit="step"):
        obs, reward, term, trunc, info = replay_env.step(None)
        if mode == "kinematic":
            st = min(i, num_steps - 1)
            err = (
                (replay_env.env.robot.get_qpos() - rec_states["robot"]["qpos"][:, st])
                .abs()
                .max()
                .item()
            )
            max_err = max(max_err, err)
        if bool(trunc.all()):
            break
    if mode == "kinematic":
        log_info(
            f"Replay complete ({num_steps} steps). Max state error vs recorded: {max_err:.6e}",
            color="green",
        )
    else:
        log_info(f"Replay complete ({num_steps} steps).", color="green")


class _ReplayControlInput:
    """Read replay-control commands immediately when stdin is a terminal."""

    def __init__(self):
        self._fd = None
        self._term_attrs = None
        self.single_key = False

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        if os.name == "nt":
            self.single_key = True
            return self

        import termios
        import tty

        self._term_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.single_key = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._term_attrs is not None and self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_attrs)

    def read_key(self, timeout: float | None = None) -> str | None:
        """Read one key, or return ``None`` when the timeout expires."""
        if os.name == "nt" and self.single_key:
            import msvcrt

            if timeout is None:
                return msvcrt.getwch().lower()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch().lower()
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            return None

        if timeout is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None

        if self.single_key:
            value = sys.stdin.read(1)
        else:
            value = sys.stdin.readline()
        if value == "":
            raise EOFError
        return value.lower() if self.single_key else value.strip().lower()


def _read_replay_control_command(
    control_input: _ReplayControlInput, initial: str | None = None
) -> str:
    """Read a command, collecting multi-digit jump targets until Enter."""
    command = control_input.read_key() if initial is None else initial
    if command is None:
        return ""
    if not control_input.single_key or not command.isdigit():
        return command

    digits = command
    print(digits, end="", flush=True)
    while True:
        key = control_input.read_key()
        if key in ("\r", "\n"):
            print()
            return digits
        if key in ("\b", "\x7f"):
            if digits:
                digits = digits[:-1]
                print("\b \b", end="", flush=True)
            continue
        if key.isdigit():
            digits += key
            print(key, end="", flush=True)


def _run_replay_control_loop(
    replay_env: ReplayWrapper,
    control_input: _ReplayControlInput,
    *,
    visualization_runtime: VisualizationRuntime | None = None,
) -> None:
    """Run the interactive replay loop using the provided input source."""
    num_steps = int(replay_env._lengths.min().item())
    max_step = int(getattr(replay_env, "control_max_step", num_steps - 1))
    step = 0
    dt = (
        float(replay_env.env.sim_cfg.physics_dt)
        * replay_env.env.cfg.sim_steps_per_control
    )
    pending_command = None
    auto_playing = False
    prompt_visible = False
    terminal_active = True

    def publish_state(*, visible: bool = True) -> None:
        if visualization_runtime is not None:
            visualization_runtime.publish_replay_control(
                step=step,
                max_step=max_step,
                visible=visible,
            )

    def seek(target: int) -> None:
        nonlocal step
        step = max(0, min(int(target), max_step))
        replay_env.go_to_step(step)
        publish_state()

    def read_key(timeout: float | None) -> str | None:
        nonlocal terminal_active
        if not terminal_active:
            time.sleep(timeout or _REPLAY_CONTROL_POLL_INTERVAL)
            return None
        try:
            return control_input.read_key(timeout=timeout)
        except EOFError:
            if visualization_runtime is None:
                raise
            terminal_active = False
            return None

    seek(0)
    print(f"Trajectory has {num_steps} transitions (state indices 0..{max_step}).")
    try:
        while True:
            if visualization_runtime is not None:
                browser_step = visualization_runtime.drain_replay_control_command()
                if browser_step is not None:
                    if auto_playing:
                        print(f"\nPaused at step {step}.")
                    auto_playing = False
                    pending_command = None
                    seek(browser_step)
                    prompt_visible = False
                    continue

            if auto_playing:
                if step >= max_step:
                    auto_playing = False
                    prompt_visible = False
                    print(f"\nAuto replay finished at step {step}.")
                    continue

                seek(step + 1)
                print(
                    f"\r[auto step {step}/{max_step}]  press any key to pause",
                    end="",
                    flush=True,
                )
                try:
                    key = read_key(dt)
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if key is None:
                    continue

                auto_playing = False
                prompt_visible = False
                print(f"\nPaused at step {step}.")
                if key not in ("a", " ", "\r", "\n"):
                    pending_command = key
                continue

            if not prompt_visible:
                print(
                    f"[step {step}/{max_step}]  n=next  p=prev  <N>=jump  "
                    "a=auto  r=reset  q=quit"
                )
                if control_input.single_key and terminal_active:
                    print("> ", end="", flush=True)
                prompt_visible = True
            try:
                if pending_command is not None:
                    initial = pending_command
                elif visualization_runtime is not None:
                    initial = read_key(_REPLAY_CONTROL_POLL_INTERVAL)
                    if initial is None:
                        continue
                else:
                    initial = None
                command = _read_replay_control_command(
                    control_input,
                    initial=initial,
                )
            except (EOFError, KeyboardInterrupt):
                break
            finally:
                pending_command = None
            if control_input.single_key and not command.isdigit():
                print()
            prompt_visible = False
            if command in ("q", "quit"):
                break
            if command in ("n", ""):
                seek(step + 1)
            elif command in ("p", "b"):
                seek(step - 1)
            elif command == "r":
                seek(0)
            elif command == "a":
                auto_playing = True
            elif command.isdigit():
                seek(int(command))
            elif command == " ":
                continue
            else:
                print(f"Unknown command: {command!r}")
    finally:
        publish_state(visible=False)


def _replay_visualization_runtime(
    replay_env: ReplayWrapper,
) -> VisualizationRuntime | None:
    """Return the running Viser runtime used by a replay environment."""
    runtime = getattr(replay_env.env.sim, "visualization_runtime", None)
    if runtime is None or not runtime.is_running:
        return None
    return runtime


def replay_control(replay_env: ReplayWrapper) -> None:
    """Run an interactive, single-key kinematic trajectory scrubber."""
    visualization_runtime = _replay_visualization_runtime(replay_env)
    if replay_env.env.sim_cfg.headless and visualization_runtime is None:
        log_warning(
            "control mode with --headless: no window to view the scrub. "
            "Re-run without --headless or enable --viser to see the replay."
        )
    replay_env.reset()
    with _ReplayControlInput() as control_input:
        _run_replay_control_loop(
            replay_env,
            control_input,
            visualization_runtime=visualization_runtime,
        )


def main(args: Any, env: Any, gym_config: dict[str, Any]) -> None:
    """Run the selected workflow without taking ownership of ``env``."""
    if getattr(args, "replay", False):
        log_info("Replay mode.", color="green")
        replay(
            env,
            args.replay_trajectory,
            getattr(args, "replay_mode", "kinematic"),
        )
        return

    if getattr(args, "preview", False):
        log_info(
            "Preview mode enabled. Launching environment preview...", color="green"
        )
        preview(env)
        return

    if (
        getattr(args, "expansion_profile", None) is not None
        or "expansion" in gym_config
    ):
        log_info("Expansion mode enabled.", color="green")
        _run_expansion(env, args, gym_config)
        return

    plan = resolve_collection_plan(args, gym_config)
    _abort_pending_episode(env)
    num_envs = int(getattr(_env_target(env), "num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be at least 1, got {num_envs}.")
    stats = {
        "target_episodes": plan.target_episodes,
        "planned_episodes": plan.target_episodes,
        "committed_episodes": 0,
        "rejected_episodes": 0,
        "attempts": 0,
        "batch_count": 0,
        "prepare_reset_count": 0,
        "commit_reset_count": 0,
        "discard_reset_count": 0,
    }
    setattr(_env_target(env), "_collection_stats", stats)

    environment_label = "environment" if num_envs == 1 else "environments"
    tqdm.tqdm.write(
        "\n".join(
            (
                "╭─ EmbodiChain · Run Task",
                f"│ Task        {gym_config.get('id', 'unknown')}",
                f"│ Episodes    {plan.target_episodes}",
                f"│ Parallel    {num_envs} {environment_label}",
                f"│ Attempts    {plan.max_attempts} per episode",
                "╰─",
            )
        ),
        file=sys.stdout,
    )

    committed = 0
    cursor = 0
    with tqdm.tqdm(
        total=plan.target_episodes,
        desc="Collecting episodes",
        unit="episode",
        file=sys.stdout,
        dynamic_ncols=True,
        colour="green",
        bar_format=(
            "{desc:<22} {percentage:3.0f}%│{bar}│ {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    ) as episode_progress:
        while committed < plan.target_episodes:
            batch_size = min(num_envs, plan.target_episodes - committed)
            stats["prepare_reset_count"] += 1
            env.reset()
            generated = generate_function(
                env,
                time_id=cursor,
                save_path=getattr(args, "save_path", ""),
                save_video=getattr(args, "save_video", False),
                debug_mode=getattr(args, "debug_mode", False),
                save_env_ids=tuple(range(batch_size)),
                regenerate=getattr(args, "regenerate", False),
                max_attempts=plan.max_attempts,
                reset_before=False,
            )
            stats["batch_count"] += 1
            if not generated:
                stats["rejected_episodes"] += batch_size
                raise RuntimeError(
                    f"Failed to collect batch starting at {cursor} after "
                    f"{plan.max_attempts} attempts."
                )
            committed += batch_size
            cursor += batch_size
            stats["committed_episodes"] = committed
            episode_progress.update(batch_size)

    episode_label = "episode" if committed == 1 else "episodes"
    batch_label = "batch" if stats["batch_count"] == 1 else "batches"
    tqdm.tqdm.write(
        f"✓ Collection complete · {committed} {episode_label} saved in "
        f"{stats['batch_count']} {batch_label}",
        file=sys.stdout,
    )

    # Log the trajectory save location before cli() tears down the sim and, by
    # default, os._exit()s the process.
    if getattr(args, "record_trajectory", False):
        save_dir = args.trajectory_save_dir
        if save_dir is None:
            import os

            from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

            save_dir = os.path.join(
                EMBODICHAIN_DEFAULT_DATA_ROOT,
                "trajectories",
                env.unwrapped._traj_run_id,
            )
        log_info(
            f"Trajectories recorded to: {save_dir} "
            "(replay with --replay --replay_trajectory <path>)",
            color="green",
        )


def preview(env: gymnasium.Env) -> None:
    """
    Run the following code to create a demonstration and perform env steps.

    ```
    # Demo version of environment rollout
    for i in range(10):
        qpos = env.robot.get_qpos()

        obs, reward, terminated, truncated, info = env.step(qpos)

    # reset the environment
    env.reset()
    ```

    Run the following code to preview the sensor observations.

    ```
    env.preview_sensor_data("camera")
    ```
    """
    _, _ = env.reset()

    end = False
    while end is False:
        print("Press `p` to enter embed mode to interact with the environment.")
        print("Press `q` to quit the simulation.")
        txt = input()
        if txt == "p":
            try:
                from IPython import embed
            except ImportError:
                log_error(
                    "IPython is not installed. Preview mode requires IPython to be "
                    "available. Please install it with `pip install ipython` and try again."
                )
                continue

            embed()
        elif txt == "q":
            end = True

    return


def _create_parser() -> argparse.ArgumentParser:
    """Create the ``run-env`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain run-env",
        description="Run an environment for data expansion or interactive preview.",
    )

    add_env_launcher_args_to_parser(parser, require_gym_config=True)
    parser.set_defaults(viser_image_fps=None)

    parser.add_argument(
        "--task-program",
        type=str,
        default=None,
        help="Path to a declarative Task Program (.json, .yaml, or .yml).",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Log the structured trace for each failed demo attempt.",
    )
    parser.add_argument(
        "--expansion-profile",
        "--expansion_profile",
        type=str,
        default=None,
        help=(
            "Expansion Profile path; overrides the profile selected by the "
            "task expansion declaration."
        ),
    )
    parser.add_argument(
        "--expansion-recipe-indices",
        "--expansion_recipe_indices",
        "--expansion-candidate-indices",
        "--expansion_candidate_indices",
        dest="expansion_recipe_indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit logical expansion recipe indices (legacy candidate name is accepted).",
    )
    parser.add_argument(
        "--output-dir",
        "--expansion-output-dir",
        "--expansion_output_dir",
        dest="expansion_output_dir",
        type=str,
        default=None,
        help="Directory for expansion provenance and manifests.",
    )
    parser.add_argument(
        "--dataset-dir",
        "--expansion-dataset-dir",
        "--expansion_dataset_dir",
        dest="expansion_dataset_dir",
        type=str,
        default=None,
        help="Override dataset manager save_path values.",
    )
    parser.add_argument(
        "--expansion-save-video",
        "--expansion_save_video",
        action="store_true",
        help="Save expansion episode videos when the environment supports it.",
    )

    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay a recorded trajectory (--replay_trajectory required).",
    )
    parser.add_argument(
        "--replay_trajectory",
        type=str,
        default=None,
        help="Path to the .pt trajectory file to replay.",
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        choices=["kinematic", "dynamic", "control"],
        default="kinematic",
        help="Replay mode: kinematic (exact, default), dynamic (physics), "
        "control (interactive scrubber).",
    )
    return parser


def _abort_and_close_env(env: Any, *, exit_process: bool | None = None) -> None:
    """Abort pending data, then close the environment exactly once.

    Args:
        env: Gym environment or wrapper.
        exit_process: Optional process-exit policy for ``EmbodiedEnv.close``.
            An abort failure always forces ``False`` so the error can propagate.
    """
    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        _abort_pending_episode(env)
    except BaseException as error:
        abort_error = error
    try:
        if exit_process is None and abort_error is None:
            env.close()
        else:
            target = getattr(env, "unwrapped", env)
            target.close(
                exit_process=False if abort_error is not None else exit_process
            )
    except BaseException as error:
        close_error = error

    # Recorder finalization is a durability barrier. Never turn a failed flush
    # into a warning that lets an apparently successful data-expansion run
    # continue.
    if close_error is not None:
        if abort_error is not None:
            close_error.add_note(
                "Pending episode abort also failed: "
                f"{type(abort_error).__name__}: {abort_error}"
            )
        raise close_error
    if abort_error is not None:
        raise abort_error


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for environment runner.

    Parses CLI arguments, builds the environment config, and launches
    the data expansion, preview, or replay workflow.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    args = _create_parser().parse_args(argv)

    if getattr(args, "replay", False):
        if not args.replay_trajectory:
            log_error("--replay requires --replay_trajectory <path>.")
            return
        if getattr(args, "preview", False):
            log_error("--replay and --preview are mutually exclusive.")
            return

    # Step 1: Discover all task packages via entry_points
    discover_task_packages()

    # Step 2: Execute init hooks (register managers, asset resolvers, etc.)
    execute_init_hooks()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    expansion_request = None
    if (
        getattr(args, "expansion_profile", None) is not None
        or "expansion" in gym_config
    ):
        expansion_request = _resolve_expansion_request(args, gym_config)
    if expansion_request is not None:
        from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile

        profile = expansion_request[0]
        program = getattr(env_cfg, "task_program", None)
        if program is None:
            raise ValueError("expansion requires a configured Task Program")
        if (
            profile.source.kind != "task_program"
            or profile.source.source_id != program.program_id
        ):
            raise ValueError(
                "expansion profile source must match the configured Task Program"
            )
        expansion_plan = resolve_collection_plan(
            args,
            gym_config,
            expansion_collection=_load_expansion_collection(expansion_request[2]),
            legacy_recipe_indices=expansion_request[1],
        )
        if isinstance(profile, CombinedExpansionProfile):
            _validate_expansion_collection_plan(
                profile,
                expansion_plan,
                num_envs=env_cfg.num_envs,
            )
            if getattr(args, "disable_sensor", False) and profile.visual.enabled:
                raise ValueError(
                    "--disable-sensor cannot be used with a combined profile "
                    "that contains visual expansion"
                )
            if profile.scene_randomization.enabled or profile.visual.enabled:
                configured_events = getattr(env_cfg, "events", None)
                if getattr(configured_events, "expansion_profile_reset", None) is None:
                    raise ValueError(
                        "combined expansion with scene or visual variation "
                        "requires the expansion_profile_reset event"
                    )
        else:
            raise ValueError("expansion requires a CombinedExpansionProfile")

    if args.replay and args.replay_mode == "control":
        log_info("Dataset saving disabled for control replay mode.", color="green")

    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    # Ensure the sim is torn down via env.close() (-> SimulationManager.destroy())
    # before the interpreter shuts down. Without this, C++ resources (dexsim/warp/
    # CUDA) are finalized during Python shutdown in an unpredictable order, which
    # segfaults on exit (exit code 139). ``destroy()`` queues a deferred cleanup
    # task and, by default (EMBODICHAIN_SIM_EXIT_PROCESS=1), calls ``os._exit(0)``
    # to skip the unsafe teardown entirely. When that env var is disabled (e.g.
    # dev/test), ``flush_cleanup_queue`` drains the queue and runs the deferred
    # destruction + GC + scene-barrier so we still exit cleanly.
    body_error: BaseException | None = None
    try:
        main(args, env, gym_config)
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            # close() may auto-save trajectory state and finalizes asynchronous
            # dataset writers. Resolve the pending transaction as an abort
            # first, including when main() exits via an interrupt or SystemExit.
            _abort_and_close_env(
                env,
                # Successful CLI runs keep the existing fast-exit default.
                # While unwinding, cleanup must return so the original error
                # (including Ctrl-C) remains observable to the caller/shell.
                exit_process=False if body_error is not None else None,
            )
        except BaseException as cleanup_error:
            if body_error is None:
                raise
            if isinstance(body_error, SystemExit) and body_error.code in (None, 0):
                # A nominal zero exit must not hide a failed recorder barrier.
                raise cleanup_error
            body_error.add_note(
                "Environment cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        finally:
            try:
                from embodichain.lab.sim.sim_manager import SimulationManager

                SimulationManager.flush_cleanup_queue()
            except Exception as error:
                log_warning(f"Failed to flush simulation cleanup queue: {error}")


if __name__ == "__main__":
    cli()


__all__ = [
    "cli",
    "generate_and_execute_action_list",
    "generate_function",
    "main",
    "preview",
]
