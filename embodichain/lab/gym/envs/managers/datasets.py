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

"""Dataset functors for collecting and saving episode data."""

from __future__ import annotations

import copy
import json
import math
import threading

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import gymnasium as gym
import torch
import tqdm

from tensordict import TensorDict

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.data.enum import LeRobotKey
from embodichain.data_pipeline.depth_video import (
    DepthSidecarManager,
    DepthVideoCfg,
    detect_depth_encoder,
)
from embodichain.lab.sim.sensors import Camera, ContactSensor
from embodichain.lab.gym.envs.demo import DEMO_ANNOTATION_KEYS, DEMO_SCHEMA_VERSION
from .manager_base import Functor
from .cfg import DatasetFunctorCfg

__all__ = ["LeRobotRecorder"]

CAMERA_IMAGE_FRAMES = {
    "color": "",
    "color_right": "_right",
}
# Depth and mask share the ``observation.<modality>.<sensor>[_right]`` key
# layout (see ``_camera_feature_key``), but are stored differently: depth can
# be offloaded to compressed sidecar videos (issue #424, Path A), while mask is
# always kept as an exact numeric LeRobot feature.
CAMERA_DEPTH_FRAMES = {
    "depth",
    "depth_right",
}
CAMERA_MASK_FRAMES = {
    "mask",
    "mask_right",
}
CAMERA_AUXILIARY_FRAMES = CAMERA_DEPTH_FRAMES | CAMERA_MASK_FRAMES

DEMO_FRAME_FEATURES = {
    "episode_step": "annotation.episode_step",
    "segment_id": "annotation.segment_id",
    "segment_step": "annotation.segment_step",
    "segment_start": "annotation.segment_start",
    "segment_end": "annotation.segment_end",
    "segment_accepted": "annotation.segment_accepted",
    "segment_attempt_id": "annotation.segment_attempt_id",
    "continuity_id": "annotation.continuity_id",
    "terminated": "annotation.terminated",
    "truncated": "annotation.truncated",
}
LEROBOT_SUBTASK_INDEX_KEY = "subtask_index"
LEROBOT_SUBTASKS_PATH = Path("meta/subtasks.parquet")

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    import pandas as pd

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
    __all__ = ["LeRobotRecorder"]
except ImportError:
    LEROBOT_AVAILABLE = False
    __all__ = []


class LeRobotRecorder(Functor):
    """Functor for recording episodes in LeRobot format.

    This functor handles:

    - Recording observation-action pairs during episodes
    - Converting data to LeRobot format
    - Saving episodes when they complete
    """

    def __init__(self, cfg: DatasetFunctorCfg, env: EmbodiedEnv):
        """Initialize the LeRobot dataset recorder.

        Args:
            cfg: Functor configuration containing params:
                - save_path: Root directory for saving datasets
                - robot_meta: Robot metadata for dataset
                - instruction: Optional task instruction
                - extra: Optional extra metadata
                - use_videos: Whether to save videos
                - image_writer_threads: Number of threads for image writing
                - image_writer_processes: Number of processes for image writing
            env: The environment instance
        """
        if not LEROBOT_AVAILABLE:
            logger.log_error(
                "LeRobot is not installed. Please install it with: pip install lerobot"
            )

        super().__init__(cfg, env)

        # Extract parameters from cfg.params
        params = cfg.params

        # Required parameters
        self.lerobot_data_root = params.get(
            "save_path", EMBODICHAIN_DEFAULT_DATASET_ROOT
        )
        self.robot_meta = dict(params.get("robot_meta", {}) or {})
        self.dataset_fps = self._derive_dataset_fps()

        # Optional parameters
        self.instruction = params.get("instruction", None)
        self.extra = params.get("extra", {})

        # Experimental parameters for extra episode info saving.
        self.use_videos = params.get("use_videos", False)

        # Async image writing (lerobot official AsyncImageWriter).
        # When > 0, per-frame PNG writes are offloaded to a thread/process pool
        # so add_frame() no longer blocks on PIL.Image.save(). This is the
        # single biggest lever for saving throughput with camera sensors.
        # Threads share the process (cheap, GIL-released by PIL C path);
        # processes add isolation at a higher spawn cost.
        self.image_writer_threads = int(params.get("image_writer_threads", 0))
        self.image_writer_processes = int(params.get("image_writer_processes", 0))

        # Compressed depth sidecar videos (issue #424, Path A). When enabled and
        # an HEVC encoder is available, camera depth is written as gray12le/HEVC
        # MP4s alongside the LeRobot dataset instead of (or, with
        # ``keep_numeric_fallback``, in addition to) numeric Parquet features.
        self.depth_video_cfg = self._parse_depth_video_cfg(params)
        self._depth_video_enabled = self._resolve_depth_video_enabled(
            self.depth_video_cfg
        )
        # Per-sensor depth specs collected in _build_features: {sensor_key: shape}.
        self._depth_sensor_specs: Dict[str, tuple] = {}
        # Sidecar manager, created in _initialize_dataset once the root is known.
        self._depth_manager: Optional[DepthSidecarManager] = None

        # LeRobot dataset instance
        self.dataset: Optional[LeRobotDataset] = None
        self.dataset_full_path: Optional[Path] = None

        # Tracking
        self.total_time: float = 0.0
        self.curr_episode: int = 0
        self._metadata_lock = threading.Lock()
        self._subtask_to_index: dict[str, int] = {}
        self._fragment_commit_lock = threading.RLock()
        self._committed_fragment_ids: dict[str, int] = {}
        self._partial_fragment_commits: dict[str, tuple[int, str]] = {}
        self._finalize_lock = threading.Lock()
        self._finalized = False
        self._finalize_result: Optional[str] = None
        self._finalize_error: Optional[str] = None

        # Initialize dataset
        self._initialize_dataset()

    def _derive_dataset_fps(self) -> int:
        """Derive the integer LeRobot sampling rate from the environment step."""
        step_dt = float(self._env.step_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(
                f"env.step_dt must be a finite positive number, got {step_dt!r}."
            )

        control_frequency = 1.0 / step_dt
        dataset_fps = round(control_frequency)
        if not math.isclose(
            control_frequency, float(dataset_fps), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"The environment control frequency ({control_frequency:g} Hz) is "
                "not an integer, but LeRobot requires an integer dataset FPS. "
                "Choose compatible physics_dt and sim_steps_per_control values or "
                "resample observations before recording."
            )
        return dataset_fps

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return (
            str(self.dataset_full_path) if self.dataset_full_path else "Not initialized"
        )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        save_path: Optional[str] = None,
        robot_meta: Optional[Dict] = None,
        instruction: Optional[str] = None,
        extra: Optional[Dict] = None,
        use_videos: bool = False,
        **kwargs,
    ) -> None:
        """Main entry point for the recorder functor.

        This method is called by DatasetManager.apply(mode="save") to save completed episodes.
        It reads data from the environment's episode buffers.

        Args:
            env: The environment instance.
            env_ids: Environment IDs to save. If None, attempts to save all environments.
            save_path: Unused at call time (honored at construction).
            robot_meta: Unused at call time (honored at construction).
            instruction: Unused at call time (honored at construction).
            extra: Unused at call time (honored at construction).
            use_videos: Unused at call time (honored at construction).
            **kwargs: Construction-only params (e.g. ``image_writer_threads``,
                ``image_writer_processes``, ``save_path``) passed through by
                ``DatasetManager.apply`` via ``**functor_cfg.params``. They are
                read in :meth:`__init__` and ignored here.
        """
        with self._finalize_lock:
            if self._finalized:
                raise RuntimeError("LeRobotRecorder is already finalized")

            # If env_ids is None, check all environments for completed episodes
            if env_ids is None:
                env_ids = torch.arange(env.num_envs, device=env.device)
            elif isinstance(env_ids, (list, range)):
                env_ids = torch.tensor(list(env_ids), device=env.device)

            # Save episodes for specified environments
            if len(env_ids) > 0:
                self._save_episodes(env_ids)

    def _save_episodes(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        """Save completed episodes for specified environments.

        This reads each env's slice from the rollout buffer and delegates to
        :meth:`_persist_episode_payload`. The slice read happens in the caller
        thread so that subclasses (e.g. :class:`AsyncLeRobotRecorder`) can
        clone the slice and defer the actual conversion/disk-write to a
        background worker without racing the buffer reuse on reset.
        """
        for env_id in env_ids.cpu().tolist():
            step = self._episode_length(env_id)
            # The first env.reset() can request a dataset save before any
            # transition has been recorded.  That is an empty buffer, not a
            # committed episode whose persistence failed.
            if step <= 0:
                continue
            obs_list = self._env.rollout_buffer["obs"][env_id, :step]
            action_list = self._env.rollout_buffer["actions"][env_id, :step]
            annotations = {
                key: self._env.rollout_buffer[key][env_id, :step]
                for key in DEMO_ANNOTATION_KEYS
                if key in self._env.rollout_buffer.keys()
            }
            metadata_getter = getattr(self._env, "get_demo_episode_metadata", None)
            episode_metadata = (
                metadata_getter(env_id) if metadata_getter is not None else None
            )
            payloads = list(
                self._episode_payloads(
                    env_id,
                    obs_list,
                    action_list,
                    annotations,
                    episode_metadata,
                )
            )
            if (
                episode_metadata is not None
                and episode_metadata.get("output_mode") == "segment_fragments"
                and not payloads
            ):
                raise RuntimeError(
                    f"Committed fragment collection for env {env_id} had no "
                    "eligible segment spans."
                )
            resolved_fragment_ids: list[str] = []
            for payload in payloads:
                fragment_id = self._fragment_id_from_metadata(payload[-1])
                try:
                    saved = self._persist_episode_payload(*payload)
                except Exception as error:
                    if fragment_id is None:
                        raise
                    prior = (
                        f" Earlier fragments {resolved_fragment_ids!r} remain "
                        "committed and will be deduplicated on retry."
                        if resolved_fragment_ids
                        else ""
                    )
                    raise RuntimeError(
                        f"Failed to persist fragment {fragment_id!r} for env "
                        f"{env_id}.{prior}"
                    ) from error
                if not saved:
                    label = (
                        f"fragment {fragment_id!r}"
                        if fragment_id is not None
                        else f"episode for env {env_id}"
                    )
                    raise RuntimeError(f"Committed {label} was not persisted.")
                if fragment_id is not None:
                    resolved_fragment_ids.append(fragment_id)

    def _episode_payloads(
        self,
        env_id: int,
        obs_list: Any,
        action_list: Any,
        annotations: Mapping[str, Any],
        episode_metadata: Mapping[str, Any] | None,
    ) -> Iterable[tuple[int, Any, Any, Mapping[str, Any], Mapping[str, Any] | None]]:
        """Yield one continuous payload or independent natural-segment slices."""
        if (
            episode_metadata is None
            or episode_metadata.get("output_mode") != "segment_fragments"
        ):
            yield env_id, obs_list, action_list, annotations, episode_metadata
            return

        episode_length = min(len(obs_list), len(action_list))
        include_failed = bool(episode_metadata.get("save_failed_fragments", False))
        for segment in episode_metadata.get("segments", []):
            if not isinstance(segment, Mapping):
                raise TypeError("Segment sidecar metadata must be a mapping.")
            accepted = bool(segment.get("success", False))
            if not accepted and not include_failed:
                continue
            start = int(segment.get("start_step", 0))
            end = int(segment.get("end_step", 0))
            if start < 0 or end > episode_length or end <= start:
                raise ValueError(
                    "Fragment span must be a non-empty subset of the buffered "
                    f"episode; got [{start}, {end}) for length {episode_length}."
                )

            length = end - start
            fragment_annotations = {
                key: values[start:end].clone() for key, values in annotations.items()
            }
            reference = next(iter(fragment_annotations.values()), None)
            device = getattr(reference, "device", None)
            fragment_annotations["episode_step"] = torch.arange(
                length, dtype=torch.int64, device=device
            )
            fragment_annotations["segment_step"] = torch.arange(
                length, dtype=torch.int64, device=device
            )
            fragment_annotations["segment_start"] = torch.zeros(
                length, dtype=torch.bool, device=device
            )
            fragment_annotations["segment_start"][0] = True
            fragment_annotations["segment_end"] = torch.zeros(
                length, dtype=torch.bool, device=device
            )
            fragment_annotations["segment_end"][-1] = True
            fragment_annotations["segment_accepted"] = torch.full(
                (length,), accepted, dtype=torch.bool, device=device
            )
            fragment_annotations["segment_attempt_id"] = torch.full(
                (length,),
                int(segment.get("attempt_id", episode_metadata.get("attempt_id", 0))),
                dtype=torch.int64,
                device=device,
            )
            fragment_annotations["continuity_id"] = torch.full(
                (length,),
                int(
                    segment.get(
                        "continuity_id", episode_metadata.get("continuity_id", 0)
                    )
                ),
                dtype=torch.int64,
                device=device,
            )

            fragment_segment = copy.deepcopy(dict(segment))
            fragment_segment.update({"start_step": 0, "end_step": length})
            fragment_metadata = copy.deepcopy(dict(episode_metadata))
            program_run_id = str(fragment_metadata.get("program_run_id", "unknown"))
            segment_provenance = segment.get("metadata", {})
            if not isinstance(segment_provenance, Mapping):
                segment_provenance = {}
            source_program_id = segment_provenance.get("task_program_id")
            if source_program_id is None:
                # Schema-v2 Expert Program sidecars remain readable.
                source_program_id = segment_provenance.get("expert_program_id")
            segment_attempt_id = int(
                segment.get("attempt_id", episode_metadata.get("attempt_id", 0))
            )
            continuity_id = int(
                segment.get("continuity_id", episode_metadata.get("continuity_id", 0))
            )
            fragment_terminated = bool(
                torch.as_tensor(
                    fragment_annotations.get(
                        "terminated", torch.zeros(length, dtype=torch.bool)
                    )[-1]
                ).item()
            )
            fragment_truncated = bool(
                torch.as_tensor(
                    fragment_annotations.get(
                        "truncated", torch.zeros(length, dtype=torch.bool)
                    )[-1]
                ).item()
            )
            fragment_metadata.update(
                {
                    "fragment": True,
                    "fragment_origin": "natural_segment",
                    "fragment_id": (
                        f"{program_run_id}:{env_id}:"
                        f"{int(segment.get('segment_id', 0))}:"
                        f"{segment_attempt_id}:{continuity_id}"
                    ),
                    "source_program_id": source_program_id,
                    "program_segment_id": segment_provenance.get("program_segment_id"),
                    "source_episode_index": fragment_metadata.get("episode_index"),
                    "source_env_id": env_id,
                    "source_start_step": start,
                    "source_end_step": end,
                    "length": length,
                    "completed": accepted,
                    "success": accepted,
                    "terminated": fragment_terminated,
                    "truncated": fragment_truncated,
                    "terminal_reason": (
                        "segment_succeeded"
                        if accepted
                        else segment.get("failure_reason")
                        or segment.get("outcome_kind")
                        or "segment_failed"
                    ),
                    "segments": [fragment_segment],
                }
            )
            yield (
                env_id,
                obs_list[start:end].clone(),
                action_list[start:end].clone(),
                fragment_annotations,
                fragment_metadata,
            )

    @staticmethod
    def _fragment_id_from_metadata(
        episode_metadata: Mapping[str, Any] | None,
    ) -> str | None:
        """Return the stable idempotency key for one fragment payload."""
        if episode_metadata is None or not episode_metadata.get("fragment", False):
            return None
        fragment_id = episode_metadata.get("fragment_id")
        if (
            not isinstance(fragment_id, str)
            or not fragment_id
            or fragment_id != fragment_id.strip()
        ):
            raise ValueError(
                "Fragment metadata must contain a non-empty fragment_id without "
                "outer whitespace."
            )
        return fragment_id

    def _ensure_fragment_commit_tracking(self) -> None:
        """Initialize fragment commit state for lightweight test instances."""
        if not hasattr(self, "_fragment_commit_lock"):
            self._fragment_commit_lock = threading.RLock()
        if not hasattr(self, "_committed_fragment_ids"):
            self._committed_fragment_ids = {}
        if not hasattr(self, "_partial_fragment_commits"):
            self._partial_fragment_commits = {}

    def _persist_episode_payload(
        self,
        env_id: int,
        obs_list: Any,
        action_list: Any,
        annotations: Mapping[str, Any] | None = None,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Persist one payload, deduplicating completed fragment commits.

        Fragment ids are scoped to the current recorder/dataset. A successful
        fragment is an independent commit and is skipped if the same source
        collection is retried. A post-commit failure is sticky: the LeRobot
        episode already exists, so retrying raises instead of creating a
        duplicate with incomplete sidecar durability.
        """
        fragment_id = self._fragment_id_from_metadata(episode_metadata)
        if fragment_id is None:
            return self._save_single_episode(
                env_id,
                obs_list,
                action_list,
                annotations=annotations,
                episode_metadata=episode_metadata,
            )

        self._ensure_fragment_commit_tracking()
        with self._fragment_commit_lock:
            partial_commit = self._partial_fragment_commits.get(fragment_id)
            if partial_commit is not None:
                episode_index, message = partial_commit
                raise RuntimeError(
                    f"Fragment {fragment_id!r} already reached LeRobot episode "
                    f"{episode_index}, but post-commit finalization failed: "
                    f"{message}. Refusing to write a duplicate."
                )
            committed_episode = self._committed_fragment_ids.get(fragment_id)
            if committed_episode is not None:
                logger.log_info(
                    f"[LeRobotRecorder] Skipping duplicate fragment "
                    f"{fragment_id!r}; already saved as episode "
                    f"{committed_episode}."
                )
                return True

            episode_index = int(getattr(self, "curr_episode", 0))
            saved = self._save_single_episode(
                env_id,
                obs_list,
                action_list,
                annotations=annotations,
                episode_metadata=episode_metadata,
            )
            if saved:
                self._committed_fragment_ids[fragment_id] = episode_index
            return saved

    def _episode_length(self, env_id: int) -> int:
        """Return the valid buffered length for one environment."""
        rollout_steps = getattr(self._env, "rollout_steps", None)
        if rollout_steps is not None:
            return int(rollout_steps[env_id].item())
        if "valid" in self._env.rollout_buffer.keys():
            return int(self._env.rollout_buffer["valid"][env_id].sum().item())
        return int(self._env.current_rollout_step)

    def _save_single_episode(
        self,
        env_id: int,
        obs_list: Any,
        action_list: Any,
        annotations: Mapping[str, Any] | None = None,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Convert and persist one episode already sliced from the buffer.

        This operates purely on the provided ``obs_list`` / ``action_list``
        (which may be live buffer views or detached clones) and never touches
        ``self._env.rollout_buffer`` or ``self._env.current_rollout_step``,
        so it is safe to call from a background thread on cloned data.

        Args:
            env_id: Environment id (used for logging only).
            obs_list: Per-frame observations for the episode.
            action_list: Per-frame actions for the episode.
            annotations: Optional per-frame segment and terminal annotations.
            episode_metadata: Optional episode/segment sidecar metadata.

        Returns:
            True if the episode was saved successfully, False otherwise.
        """
        task = (
            self.instruction.get("lang", "unknown_task")
            if self.instruction
            else "unknown_task"
        )
        if episode_metadata is not None and episode_metadata.get("fragment"):
            segments = episode_metadata.get("segments", [])
            if segments and isinstance(segments[0], Mapping):
                task = self._normalize_subtask_description(
                    segments[0].get("instruction") or task
                )

        if len(obs_list) == 0:
            logger.log_warning(f"No episode data to save for env {env_id}")
            return False

        # Native expert buffers provide causal (s_t, a_t) pairs of equal length.
        # Keep the trim for legacy/external callers that still pass a trailing
        # next observation.
        if len(obs_list) > len(action_list):
            obs_list = obs_list[:-1]
        episode_length = min(len(obs_list), len(action_list))
        obs_list = obs_list[:episode_length]
        action_list = action_list[:episode_length]
        if annotations is not None:
            annotations = {
                key: values[:episode_length] for key, values in annotations.items()
            }

        # Update metadata
        extra_info = self.extra.copy() if self.extra else {}
        current_episode_time = len(obs_list) * float(self._env.step_dt)

        episode_extra_info = extra_info.copy()
        previous_total_time = self.total_time
        self.total_time += current_episode_time
        episode_extra_info["total_time"] = self.total_time

        depth_prefix = f"{LeRobotKey.OBS_PREFIX.value}depth."
        episode_index = self.curr_episode
        dataset_committed = False
        fragment_id = self._fragment_id_from_metadata(episode_metadata)
        episode_attempt_id = int((episode_metadata or {}).get("attempt_id", 0))
        episode_continuity_id = int((episode_metadata or {}).get("continuity_id", 0))
        try:
            frame_subtasks = [
                self._subtask_for_frame(task, episode_metadata, frame_index)
                for frame_index in range(episode_length)
            ]
            subtask_indices = self._register_subtasks(frame_subtasks)
            if self._depth_manager is not None:
                self._depth_manager.start_episode(
                    episode_index, list(self._depth_sensor_specs.keys())
                )
            for frame_index, (obs, action) in enumerate(
                tqdm.tqdm(
                    zip(obs_list, action_list),
                    total=len(obs_list),
                    desc=f"Converting env {env_id} episode to LeRobot format",
                )
            ):
                frame_segment = self._segment_for_frame(episode_metadata, frame_index)
                frame_annotations = {
                    "episode_step": frame_index,
                    "segment_id": 0,
                    "segment_step": frame_index,
                    "segment_start": frame_index == 0,
                    "segment_end": frame_index == len(obs_list) - 1,
                    "segment_accepted": (
                        bool(frame_segment.get("success", True))
                        if frame_segment is not None
                        else True
                    ),
                    "segment_attempt_id": (
                        int(frame_segment.get("attempt_id", episode_attempt_id))
                        if frame_segment is not None
                        else episode_attempt_id
                    ),
                    "continuity_id": (
                        int(frame_segment.get("continuity_id", episode_continuity_id))
                        if frame_segment is not None
                        else episode_continuity_id
                    ),
                    "terminated": False,
                    "truncated": False,
                }
                if annotations is not None:
                    frame_annotations.update(
                        {
                            key: values[frame_index]
                            for key, values in annotations.items()
                            if key in DEMO_FRAME_FEATURES
                        }
                    )
                if frame_index == len(obs_list) - 1:
                    # Legacy/manual collection has no end-segment callback;
                    # the last committed frame is still an episode boundary.
                    frame_annotations["segment_end"] = True
                frame_subtask = frame_subtasks[frame_index]
                frame = self._convert_frame_to_lerobot(
                    obs,
                    action,
                    task,
                    annotations=frame_annotations,
                    subtask_index=subtask_indices[frame_subtask],
                )
                # Offload depth to the sidecar writer and drop it from the frame
                # so LeRobot's RGB-only image/video path never sees it. With
                # ``keep_numeric_fallback`` the numeric feature is retained too.
                if self._depth_manager is not None:
                    for key in list(frame.keys()):
                        if key.startswith(depth_prefix):
                            sensor_key = key[len(depth_prefix) :]
                            self._depth_manager.add_frame(sensor_key, frame[key])
                            if not self.depth_video_cfg.keep_numeric_fallback:
                                del frame[key]
                self.dataset.add_frame(frame)

            self._normalize_scalar_episode_buffer()
            self.dataset.save_episode()
            # LeRobot has committed this index. Advance immediately so a later
            # depth/metadata failure cannot make the next queued episode reuse
            # and overwrite the same sidecar filename.
            dataset_committed = True
            self.curr_episode += 1
            if self._depth_manager is not None:
                self._depth_manager.end_episode(episode_index)

            sidecar_metadata = dict(episode_metadata or {})
            if not sidecar_metadata.get("segments"):
                sidecar_metadata["segments"] = [
                    {
                        "segment_id": 0,
                        "name": "legacy",
                        "start_step": 0,
                        "end_step": len(obs_list),
                        "success": True,
                        "target_uid": None,
                        "instruction": task,
                        "failure_reason": None,
                        "metadata": {},
                    }
                ]
            sidecar_metadata.update(episode_extra_info)
            sidecar_metadata.update(
                {
                    "schema_version": DEMO_SCHEMA_VERSION,
                    "lerobot_episode_index": episode_index,
                    "env_id": env_id,
                    "length": len(obs_list),
                    "instruction": task,
                }
            )
            self._write_episode_metadata(sidecar_metadata)

            logger.log_info(
                f"[LeRobotRecorder] Saved dataset to: {self.dataset_path}\n"
                f"  Episode {episode_index} (env {env_id}): {len(obs_list)} frames"
            )

            return True
        except Exception as error:
            if dataset_committed and fragment_id is not None:
                self._ensure_fragment_commit_tracking()
                with self._fragment_commit_lock:
                    self._partial_fragment_commits[fragment_id] = (
                        episode_index,
                        f"{type(error).__name__}: {error}",
                    )
            if not dataset_committed:
                self.total_time = previous_total_time
            if self._depth_manager is not None and not dataset_committed:
                try:
                    self._depth_manager.abort_episode()
                except Exception as abort_error:  # noqa: BLE001 - preserve primary
                    error.add_note(
                        "Depth sidecar abort also failed: "
                        f"{type(abort_error).__name__}: {abort_error}"
                    )
            raise

    def _normalize_scalar_episode_buffer(self) -> None:
        """Collapse single-value arrays before LeRobot serializes an episode.

        LeRobot 0.4.4 validates a scalar feature declared with shape ``(1,)``
        as a one-dimensional NumPy array when :meth:`add_frame` is called, but
        maps that same feature to a Hugging Face ``Value`` during
        :meth:`save_episode`. Without normalization, LeRobot stacks the frame
        arrays into shape ``(frames, 1)`` and ``datasets`` converts each
        ``array([value])`` to a Python scalar. NumPy 2.4 rejects that implicit
        conversion.

        This method runs after per-frame validation and replaces buffered
        one-element numeric arrays with dtype-preserving NumPy scalars. The
        subsequent LeRobot stack therefore has shape ``(frames,)``, matching
        the Hugging Face scalar schema.

        Raises:
            ValueError: If a feature declared with shape ``(1,)`` contains a
                buffered value with more than one element.
        """
        if self.dataset is None:
            return

        episode_buffer = getattr(self.dataset, "episode_buffer", None)
        features = getattr(self.dataset, "features", None)
        if not isinstance(episode_buffer, dict) or not isinstance(features, Mapping):
            return

        for feature_key, feature in features.items():
            if not isinstance(feature, Mapping):
                continue
            if tuple(feature.get("shape", ())) != (1,):
                continue

            values = episode_buffer.get(feature_key)
            if not isinstance(values, list) or not values:
                continue

            try:
                dtype = np.dtype(feature["dtype"])
            except (KeyError, TypeError, ValueError):
                continue

            normalized_values: list[np.generic] = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                array = np.asarray(value, dtype=dtype)
                if array.size != 1:
                    raise ValueError(
                        f"Scalar LeRobot feature {feature_key!r} expected one "
                        f"value, got shape {array.shape}."
                    )
                normalized_values.append(array.reshape(-1)[0])
            episode_buffer[feature_key] = normalized_values

    @staticmethod
    def _segment_for_frame(
        episode_metadata: Mapping[str, Any] | None,
        frame_index: int,
    ) -> Mapping[str, Any] | None:
        """Return the sidecar segment owning one frame, when available."""
        if episode_metadata is None:
            return None
        for segment in episode_metadata.get("segments", []):
            if not isinstance(segment, Mapping):
                continue
            if (
                int(segment.get("start_step", 0))
                <= frame_index
                < int(segment.get("end_step", 0))
            ):
                return segment
        return None

    @staticmethod
    def _subtask_for_frame(
        default_subtask: str,
        episode_metadata: Mapping[str, Any] | None,
        frame_index: int,
    ) -> str:
        """Resolve a segment-specific instruction for one LeRobot frame."""
        segment = LeRobotRecorder._segment_for_frame(episode_metadata, frame_index)
        if segment is not None:
            return LeRobotRecorder._normalize_subtask_description(
                segment.get("instruction") or default_subtask
            )
        return LeRobotRecorder._normalize_subtask_description(default_subtask)

    @staticmethod
    def _normalize_subtask_description(description: Any) -> str:
        """Return a non-empty description suitable for the subtask table."""
        return str(description).strip() or "unknown_task"

    def _register_subtasks(self, descriptions: Iterable[str]) -> dict[str, int]:
        """Register subtask descriptions and persist LeRobot's lookup table.

        LeRobot 0.4.4 can resolve a per-frame ``subtask_index`` through
        ``meta/subtasks.parquet``, but its recording API does not create that
        table. EmbodiChain therefore maintains the same description-to-index
        convention used by LeRobot's task table.

        Args:
            descriptions: Subtask descriptions referenced by an episode.

        Returns:
            The global dataset index for every referenced description.

        Raises:
            RuntimeError: If the dataset path is unavailable while new
                descriptions need to be persisted.
        """
        normalized = [
            self._normalize_subtask_description(description)
            for description in descriptions
        ]
        with self._metadata_lock:
            new_descriptions: list[str] = []
            for description in normalized:
                if description in self._subtask_to_index:
                    continue
                self._subtask_to_index[description] = len(self._subtask_to_index)
                new_descriptions.append(description)

            if new_descriptions:
                try:
                    self._write_subtasks_metadata()
                except Exception:
                    for description in reversed(new_descriptions):
                        self._subtask_to_index.pop(description)
                    raise

            return {
                description: self._subtask_to_index[description]
                for description in dict.fromkeys(normalized)
            }

    def _write_subtasks_metadata(self) -> None:
        """Atomically write the LeRobot 0.4.4 subtask lookup table."""
        if self.dataset_full_path is None or self.dataset is None:
            raise RuntimeError("LeRobotDataset is not initialized.")

        ordered_subtasks = sorted(
            self._subtask_to_index.items(), key=lambda item: item[1]
        )
        subtasks = pd.DataFrame(
            {
                LEROBOT_SUBTASK_INDEX_KEY: np.asarray(
                    [index for _, index in ordered_subtasks], dtype=np.int64
                )
            },
            index=pd.Index([description for description, _ in ordered_subtasks]),
        )
        subtasks.index.name = None

        metadata_path = self.dataset_full_path / LEROBOT_SUBTASKS_PATH
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        try:
            subtasks.to_parquet(temporary_path)
            temporary_path.replace(metadata_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        # Keep the live writer's metadata consistent with a freshly loaded
        # LeRobotDataset, which exposes this table through ``meta.subtasks``.
        self.dataset.meta.subtasks = subtasks

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert common tensor/array values for metadata serialization."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def _write_episode_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Append one episode record to EmbodiChain's LeRobot sidecar."""
        if self.dataset_full_path is None:
            return
        metadata_dir = self.dataset_full_path / "meta"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "embodichain_episodes.jsonl"
        with self._metadata_lock, metadata_path.open("a", encoding="utf-8") as stream:
            json.dump(dict(metadata), stream, default=self._json_default)
            stream.write("\n")

    def finalize(self) -> Optional[str]:
        """Finalize resources without implicitly committing a partial episode.

        Episodes are committed only when :meth:`__call__` is invoked by an
        explicit ``reset(save_data=True)``. Closing the environment therefore
        leaves any still-live rollout buffer uncommitted.

        Returns:
            The finalized dataset path, or ``None`` when no dataset exists.

        Raises:
            RuntimeError: If one or more dataset resources cannot be finalized.
        """
        with self._finalize_lock:
            if self._finalized:
                if self._finalize_error is not None:
                    raise RuntimeError(self._finalize_error)
                return self._finalize_result

            errors: list[str] = []
            if self.dataset is not None:
                # Flush + stop the async image writer (if enabled) so every
                # explicitly committed frame lands on disk before metadata is
                # finalized.
                if self.dataset.image_writer is not None:
                    try:
                        self.dataset.stop_image_writer()
                    except Exception as error:  # noqa: BLE001 - aggregate cleanup
                        errors.append(f"image writer: {error}")
                try:
                    self.dataset.finalize()
                except Exception as error:  # noqa: BLE001 - aggregate cleanup
                    errors.append(f"LeRobot dataset: {error}")

            # Depth videos are written per committed episode; this only flushes
            # their metadata and must still be attempted if LeRobot cleanup fails.
            if self._depth_manager is not None:
                try:
                    self._depth_manager.finalize()
                except Exception as error:  # noqa: BLE001 - aggregate cleanup
                    errors.append(f"depth sidecar: {error}")

            self._finalize_result = (
                self.dataset_path if self.dataset is not None else None
            )
            self._finalized = True

            if errors:
                self._finalize_error = (
                    "LeRobotRecorder failed to finalize "
                    f"{len(errors)} resource(s): {'; '.join(errors)}"
                )
                raise RuntimeError(self._finalize_error)

            if self.dataset is not None:
                logger.log_info(
                    f"[LeRobotRecorder] Dataset finalized successfully\n"
                    f"  Path: {self.dataset_path}\n"
                    f"  Total episodes: {self.curr_episode}\n"
                    f"  Total time: {self.total_time:.2f}s"
                )
            return self._finalize_result

    def close(self) -> Optional[str]:
        """Finalize the recorder; repeated calls are safe."""
        return self.finalize()

    def _parse_depth_video_cfg(self, params: Dict) -> DepthVideoCfg:
        """Parse the optional ``depth_video`` parameter into a config.

        Args:
            params: Functor parameter dict.

        Returns:
            A :class:`DepthVideoCfg`. ``enable`` defaults to ``False`` when no
            ``depth_video`` entry is present.
        """
        dv = params.get("depth_video", None)
        if dv is None:
            return DepthVideoCfg(enable=False)
        if isinstance(dv, DepthVideoCfg):
            return dv
        if isinstance(dv, dict):
            try:
                return DepthVideoCfg(**dv)
            except TypeError as e:
                logger.log_warning(
                    f"Invalid depth_video config: {e}; disabling depth video."
                )
                return DepthVideoCfg(enable=False)
        logger.log_warning(
            f"Ignoring depth_video config of unexpected type "
            f"{type(dv).__name__}; expected DepthVideoCfg or dict."
        )
        return DepthVideoCfg(enable=False)

    @staticmethod
    def _resolve_depth_video_enabled(cfg: DepthVideoCfg) -> bool:
        """Return whether compressed depth video can actually be written.

        Depth video is only enabled when the user opts in *and* an HEVC encoder
        is available; otherwise we silently fall back to numeric depth features
        (PR #422) so recording never fails on hosts without libx265.

        Args:
            cfg: Parsed depth video config.

        Returns:
            True if the sidecar writer should be active.
        """
        if not cfg.enable:
            return False
        if detect_depth_encoder(cfg.vcodec) is None:
            logger.log_warning(
                f"No HEVC encoder (libx265/hevc) available for depth video; "
                f"falling back to numeric depth features (PR #422)."
            )
            return False
        return True

    def _initialize_dataset(self) -> None:
        """Initialize the LeRobot dataset."""
        robot_type = self.robot_meta.get("robot_type", "robot")
        scene_type = self.extra.get("scene_type", "scene")
        task_description = self.extra.get("task_description", "task")

        robot_type = str(robot_type).lower().replace(" ", "_")
        task_description = str(task_description).lower().replace(" ", "_")

        # Use lerobot_data_root from __init__
        lerobot_data_root = Path(self.lerobot_data_root)

        # Generate dataset folder name with auto-incrementing suffix
        base_name = f"{robot_type}_{scene_type}_{task_description}"

        # Find the next available sequence number by checking existing folders
        existing_dirs = list(lerobot_data_root.glob(f"{base_name}_*"))
        if not existing_dirs:
            dataset_id = 0
        else:
            # Extract sequence numbers from existing directories
            max_id = -1
            for dir_path in existing_dirs:
                suffix = dir_path.name[len(base_name) + 1 :]  # +1 for underscore
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
            dataset_id = max_id + 1

        # Format dataset name with zero-padding (3 digits: 000, 001, 002, ...)
        dataset_name = f"{base_name}_{dataset_id:03d}"

        # LeRobot's root parameter is the COMPLETE dataset path (not parent directory)
        self.dataset_full_path = lerobot_data_root / dataset_name

        features = self._build_features()

        self.dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.dataset_fps,
            root=str(self.dataset_full_path),
            robot_type=robot_type,
            features=features,
            use_videos=self.use_videos,
            metadata_buffer_size=1,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
        )
        logger.log_info(f"Created LeRobot dataset at: {self.dataset_full_path}")

        # Set up the depth sidecar manager now that the dataset root and fps are
        # known. Sensors were registered into _depth_sensor_specs by
        # _build_features() above.
        if self._depth_video_enabled and self._depth_sensor_specs:
            self._depth_manager = DepthSidecarManager(
                dataset_root=self.dataset_full_path,
                fps=self.dataset_fps,
                cfg=self.depth_video_cfg,
            )
            for sensor_key, shape in self._depth_sensor_specs.items():
                self._depth_manager.register_sensor(sensor_key, shape)
            logger.log_info(
                f"[LeRobotRecorder] Depth sidecar video enabled for sensors: "
                f"{list(self._depth_sensor_specs.keys())} "
                f"(codec={self.depth_video_cfg.vcodec}, "
                f"lossless={self.depth_video_cfg.lossless})"
            )
        elif self.depth_video_cfg.enable and not self._depth_video_enabled:
            logger.log_info(
                "[LeRobotRecorder] depth_video requested but unavailable; "
                "depth will be stored as numeric features."
            )

    def _build_features(self) -> Dict:
        """Build LeRobot features dict."""
        features = {}

        state_dim = len(self._env.active_joint_ids)
        # Create joint names.
        joint_names = [
            self._env.robot.joint_names[i] for i in self._env.active_joint_ids
        ]

        features[LeRobotKey.OBS_STATE.value] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }
        features[LeRobotKey.OBS_QVEL.value] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }
        features[LeRobotKey.OBS_QF.value] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }

        # Use full qpos dimension for action (includes gripper)
        action_dim = state_dim
        features[LeRobotKey.ACTION.value] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": joint_names,
        }
        features[LEROBOT_SUBTASK_INDEX_KEY] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }

        for feature_key in DEMO_FRAME_FEATURES.values():
            features[feature_key] = {
                "dtype": "int64",
                "shape": (1,),
                "names": [feature_key.rsplit(".", 1)[-1]],
            }

        # Setup sensor observation features based env.observation.sensor
        if self._env.has_sensors:
            sensor_obs_space: dict = self._env.single_observation_space["sensor"]

            for sensor_name, value in sensor_obs_space.items():
                sensor = self._env.get_sensor(sensor_name)

                if isinstance(sensor, Camera):
                    for frame_name, space in value.items():
                        if frame_name in CAMERA_IMAGE_FRAMES:
                            feature_key = self._camera_feature_key(
                                sensor_name, frame_name
                            )
                            features[feature_key] = {
                                "dtype": "video" if self.use_videos else "image",
                                "shape": (sensor.cfg.height, sensor.cfg.width, 3),
                                "names": ["height", "width", "channel"],
                            }
                        elif frame_name in CAMERA_DEPTH_FRAMES:
                            feature_key = self._camera_feature_key(
                                sensor_name, frame_name
                            )
                            if self._depth_video_enabled:
                                # Record the sidecar sensor spec; only register a
                                # numeric feature when an exact raw fallback is
                                # requested.
                                _, _, side = frame_name.partition("_")
                                suffix = f"_{side}" if side else ""
                                self._depth_sensor_specs[f"{sensor_name}{suffix}"] = (
                                    tuple(space.shape)
                                )
                                if not self.depth_video_cfg.keep_numeric_fallback:
                                    continue
                            features[feature_key] = {
                                "dtype": str(space.dtype),
                                "shape": space.shape,
                                "names": (
                                    ["height", "width"]
                                    if len(space.shape) == 2
                                    else ["height", "width", "channel"]
                                ),
                            }
                        elif frame_name in CAMERA_MASK_FRAMES:
                            feature_key = self._camera_feature_key(
                                sensor_name, frame_name
                            )
                            features[feature_key] = {
                                "dtype": str(space.dtype),
                                "shape": space.shape,
                                "names": (
                                    ["height", "width"]
                                    if len(space.shape) == 2
                                    else ["height", "width", "channel"]
                                ),
                            }
                        else:
                            logger.log_warning(
                                f"Unsupported camera frame '{frame_name}' in sensor '{sensor_name}'"
                            )
                elif isinstance(sensor, ContactSensor):
                    for frame_name, space in value.items():
                        features[f"{sensor_name}.{frame_name}"] = {
                            "dtype": str(space.dtype),
                            "shape": space.shape,
                            "names": frame_name,
                        }

        # Add any extra features specified in observation space excluding 'robot' and 'sensor'
        for key, space in self._env.single_observation_space.items():
            if key in ["robot", "sensor"]:
                continue

            if isinstance(space, gym.spaces.Dict):
                # Handle nested Dict observation spaces (e.g., physics attributes)
                self._add_nested_features(features, key, space)
                continue

            features[key] = {
                "dtype": str(space.dtype),
                "shape": space.shape,
                "names": key,
            }

        self._modify_feature_names(features)
        return features

    @staticmethod
    def _camera_feature_key(sensor_name: str, frame_name: str) -> str:
        """Return the LeRobot feature key for a camera frame.

        Args:
            sensor_name: Camera sensor identifier.
            frame_name: Camera frame name from the observation space.

        Returns:
            A LeRobot-compatible feature key.

        Raises:
            ValueError: If the frame is not a supported image, depth, or mask frame.
        """
        if frame_name in CAMERA_IMAGE_FRAMES:
            suffix = CAMERA_IMAGE_FRAMES[frame_name]
            return f"{LeRobotKey.OBS_IMAGES.value}.{sensor_name}{suffix}"

        if frame_name in CAMERA_AUXILIARY_FRAMES:
            modality, _, side = frame_name.partition("_")
            suffix = f"_{side}" if side else ""
            return f"{LeRobotKey.OBS_PREFIX.value}{modality}.{sensor_name}{suffix}"

        raise ValueError(f"Unsupported camera frame: {frame_name}")

    def _add_nested_features(
        self, features: Dict, key: str, space: gym.spaces.Dict
    ) -> None:
        """Add features from nested Dict observation space.

        This recursively processes nested observation spaces and adds them to the features dict.
        For example, physics attributes stored as 'object_physics' with sub-keys
        (mass, friction, damping, inertia, body_scale) will be flattened to:
        - observation.object_physics.mass
        - observation.object_physics.friction
        - observation.object_physics.damping
        - observation.object_physics.inertia
        - observation.object_physics.body_scale

        Args:
            features: The features dict to update.
            key: The top-level key of the nested space.
            space: The nested Dict observation space.
        """
        for sub_key, sub_space in space.spaces.items():
            if isinstance(sub_space, gym.spaces.Dict):
                # Recursively handle deeper nesting
                self._add_nested_features(features, f"{key}.{sub_key}", sub_space)
            else:
                feature_name = f"{LeRobotKey.OBS_PREFIX.value}{key}.{sub_key}"
                # Handle empty shapes for scalar values (e.g., mass, friction, damping)
                # LeRobot requires non-empty shapes, so convert () to (1,)
                shape = sub_space.shape if sub_space.shape else (1,)
                features[feature_name] = {
                    "dtype": str(sub_space.dtype),
                    "shape": shape,
                    "names": sub_key,
                }

    def _modify_feature_names(self, features: dict[str, Any]) -> None:
        """Get feature names for an observation based on its functor config.

        Note:
            The `space` parameter is kept for API consistency but not used
            directly, as the feature names are derived from the functor config
            and entity properties.

        For observations generated by `get_object_uid`, returns meaningful names:
        - RigidObject: object UID names
        - Articulation/Robot: link names

        Args:
            key: The observation space key.
            space: The observation space.

        Returns:
            A list of feature names for the observation.
        """
        from embodichain.lab.gym.envs.managers.observations import get_object_uid
        from embodichain.lab.sim.objects import RigidObject, Articulation, Robot

        # Change the features shape if is ()
        for key, feature in features.items():
            if feature["shape"] == ():
                features[key]["shape"] = (1,)

        # Add extra observation in `add` mode based on functor config
        if "add" in self._env.observation_manager.active_functors:
            for functor_name in self._env.observation_manager.active_functors["add"]:
                functor_cfg = self._env.observation_manager.get_functor_cfg(
                    functor_name=functor_name
                )
                if functor_cfg.func == get_object_uid:
                    obs_key = functor_cfg.name
                    asset_uid = functor_cfg.params["entity_cfg"].uid
                    asset = self._env.sim.get_asset(asset_uid)
                    if isinstance(asset, RigidObject):
                        features[obs_key]["names"] = asset_uid
                    elif isinstance(asset, (Articulation, Robot)):
                        link_names = asset.link_names
                        features[obs_key]["names"] = link_names
                    else:
                        logger.log_warning(
                            f"Asset with UID '{asset_uid}' is not RigidObject, Articulation or Robot. Cannot assign feature names based on asset properties."
                        )

    def _convert_frame_to_lerobot(
        self,
        obs: TensorDict,
        action: TensorDict | torch.Tensor,
        task: str,
        annotations: Mapping[str, Any] | None = None,
        subtask_index: int = 0,
    ) -> Dict:
        """Convert a single frame to LeRobot format.

        Args:
            obs: Single environment observation (already extracted from batch)
            action: Single environment action (already extracted from batch)
            task: Episode-level task description.
            annotations: Optional segment and terminal fields for this frame.
            subtask_index: Dataset-global index of the active subtask description.

        Returns:
            Frame dict in LeRobot format with numpy arrays
        """
        frame = {
            "task": task,
            LEROBOT_SUBTASK_INDEX_KEY: torch.tensor([subtask_index], dtype=torch.int64),
        }

        if self._env.has_sensors:
            sensor_obs_space: dict = self._env.single_observation_space["sensor"]

            # Add images
            for sensor_name, value in sensor_obs_space.items():
                sensor = self._env.get_sensor(sensor_name)

                if isinstance(sensor, Camera):
                    for frame_name in value:
                        if (
                            frame_name not in CAMERA_IMAGE_FRAMES
                            and frame_name not in CAMERA_AUXILIARY_FRAMES
                        ):
                            continue

                        feature_key = self._camera_feature_key(sensor_name, frame_name)
                        frame_data = obs["sensor"][sensor_name][frame_name]
                        if frame_name in CAMERA_IMAGE_FRAMES:
                            frame_data = frame_data[:, :, :3]
                        frame[feature_key] = frame_data.cpu()
                elif isinstance(sensor, ContactSensor):
                    for frame_name in value.keys():
                        frame[f"{sensor_name}.{frame_name}"] = obs["sensor"][
                            sensor_name
                        ][
                            frame_name
                        ].cpu()  # Debug here to inspect contact sensor data
                else:
                    logger.log_warning(
                        f"Unsupported sensor type for '{sensor_name}' when converting to LeRobot format. Currently only support Camera and ContactSensor."
                    )

        # Add state (use LeRobot standard key "observation.state")
        frame[LeRobotKey.OBS_STATE.value] = obs["robot"]["qpos"].cpu()
        # Keep additional proprio data that may be useful even though not in official LeRobot format
        frame[LeRobotKey.OBS_QVEL.value] = obs["robot"]["qvel"].cpu()
        frame[LeRobotKey.OBS_QF.value] = obs["robot"]["qf"].cpu()

        # Add extra observation features if they exist
        for key in obs.keys():
            if key in ["robot", "sensor"]:
                continue

            value = obs[key]
            if isinstance(value, TensorDict):
                # Handle nested TensorDict (e.g., physics attributes)
                self._add_nested_obs_to_frame(frame, key, value)
            else:
                if value.shape == ():
                    value = value.unsqueeze(0)
                frame[key] = value.cpu()

        # Add action.
        if isinstance(action, torch.Tensor):
            action_data = action.cpu()
        elif isinstance(action, TensorDict):
            # Extract qpos from action dict
            action_tensor = action.get("qpos", None)
            if action_tensor is None:
                # Fallback to first tensor value
                for v in action.values():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        action_tensor = v
                        break

            if isinstance(action_tensor, torch.Tensor):
                action_data = action_tensor.cpu()

        frame[LeRobotKey.ACTION.value] = action_data

        if annotations is not None:
            for annotation_key, feature_key in DEMO_FRAME_FEATURES.items():
                if annotation_key not in annotations:
                    continue
                value = torch.as_tensor(annotations[annotation_key]).item()
                frame[feature_key] = torch.tensor([int(value)], dtype=torch.int64)

        return frame

    def _add_nested_obs_to_frame(
        self, frame: Dict, key: str, nested_obs: TensorDict
    ) -> None:
        """Add nested observation data to frame dict.

        This recursively processes nested TensorDict observations and adds them to the frame dict.
        For example, physics attributes stored as 'object_physics' with sub-keys
        (mass, friction, damping, inertia, body_scale) will be flattened to:
        - observation.object_physics.mass
        - observation.object_physics.friction
        - observation.object_physics.damping
        - observation.object_physics.inertia
        - observation.object_physics.body_scale

        Args:
            frame: The frame dict to update.
            key: The top-level key of nested observation.
            nested_obs: The nested TensorDict observation.
        """
        for sub_key, sub_value in nested_obs.items():
            if isinstance(sub_value, TensorDict):
                # Recursively handle deeper nesting
                self._add_nested_obs_to_frame(frame, f"{key}.{sub_key}", sub_value)
            else:
                value = sub_value.cpu()
                # Handle 0D tensors (scalars) - convert to 1D for LeRobot compatibility
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = value.unsqueeze(0)
                frame[f"{LeRobotKey.OBS_PREFIX.value}{key}.{sub_key}"] = value

    def _update_dataset_info(self, updates: dict) -> bool:
        """Update dataset metadata."""
        if self.dataset is None:
            logger.log_error("LeRobotDataset not initialized.")
            return False

        try:
            self.dataset.meta.info.update(updates)
            return True
        except Exception as e:
            logger.log_error(f"Failed to update dataset info: {e}")
            return False
