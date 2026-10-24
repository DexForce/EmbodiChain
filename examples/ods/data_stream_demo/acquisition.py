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

"""Real expert acquisition shared by the stream and file-backed comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

from examples.ods.simple_bc_demo.config import load_task_config

if TYPE_CHECKING:
    from tensordict import TensorDict

    from embodichain.data_pipeline.engine import OnlineDataEngine

__all__ = [
    "CAMERA_PROFILE_VERSION",
    "VisionMode",
    "camera_uids",
    "camera_configs",
    "camera_profile_version",
    "load_acquisition_config",
    "make_engine",
    "iter_generation_episodes",
    "poll_generation_episodes",
]

VisionMode = Literal["state", "single", "dual"]

# The original +Y wrist mount self-occluded. Version 2 qualified a fixed
# +X-side mount with SPP=1/OptiX, without temporal reconstruction or
# super-resolution. Version 3 keeps those settings and separates the inherited
# cube/goal material identities so their existing colors remain independent.
CAMERA_PROFILE_VERSION = "v3"


def camera_profile_version(profile: str = "benchmark") -> str:
    """Identify the benchmark or close-up PickPlace camera calibration."""
    if profile not in ("benchmark", "pick_place"):
        raise ValueError(f"Unknown camera profile: {profile!r}")
    return CAMERA_PROFILE_VERSION if profile == "benchmark" else "pick-place-v2"


def camera_uids(vision: VisionMode) -> tuple[str, ...]:
    """Return the frozen camera order, rejecting unknown profiles."""
    if vision == "state":
        return ()
    if vision == "single":
        return ("cam_high",)
    if vision == "dual":
        return ("cam_high", "cam_wrist")
    raise ValueError(f"Unknown vision profile: {vision!r}")


def _camera(
    uid: str, extrinsics: dict[str, Any], *, fov_degrees: float = 60.0
) -> dict[str, Any]:
    # Square pinhole images use the same horizontal and vertical FOV.
    focal = 128.0 / math.tan(math.radians(fov_degrees / 2))
    return {
        "sensor_type": "Camera",
        "uid": uid,
        "width": 256,
        "height": 256,
        "intrinsics": [focal, focal, 128.0, 128.0],
        "near": 0.005,
        "far": 3.0,
        "enable_color": True,
        "enable_depth": False,
        "enable_mask": False,
        "enable_normal": False,
        "enable_position": False,
        "extrinsics": extrinsics,
    }


def camera_configs(
    vision: VisionMode, *, camera_profile: str = "benchmark"
) -> list[dict[str, Any]]:
    """Return ordered camera calibration without importing simulation bindings."""
    camera_profile_version(camera_profile)
    cameras = camera_uids(vision)
    sensors = []
    if cameras:
        eye, target = [-0.50, -0.90, 0.88], [-0.50, 0.10, 0.20]
        if camera_profile == "pick_place":
            eye, target = [-0.60, -0.45, 0.50], [-0.42, 0.10, 0.08]
        sensors.append(
            _camera(
                "cam_high",
                {
                    "eye": eye,
                    "target": target,
                    "up": [0.0, 0.0, 1.0],
                },
            )
        )
    if vision == "dual":
        sensors.append(
            _camera(
                "cam_wrist",
                {
                    "parent": "FrankaPanda/fr3_hand_tcp",
                    "eye": (
                        [0.12, 0.0, -0.04]
                        if camera_profile == "pick_place"
                        else [0.10, 0.0, -0.10]
                    ),
                    "target": (
                        [0.0, 0.0, 0.05]
                        if camera_profile == "pick_place"
                        else [0.0, 0.0, 0.03]
                    ),
                    "up": [0.0, 1.0, 0.0],
                },
                fov_degrees=90.0 if camera_profile == "pick_place" else 60.0,
            )
        )
    return sensors


def load_acquisition_config(
    vision: VisionMode,
    device: str,
    producer_envs: int = 2,
    seed: int = 19001,
    *,
    camera_profile: str = "benchmark",
) -> dict[str, Any]:
    """Expand the existing physical task with the showcase's sensor profile.

    The expert, reset distribution, physical success checks and action decoder
    are the existing BC task's implementations. Only the independent camera
    suite, scene material identities and run-local production settings change.
    This function never imports a learner, policy model or trainer.

    Args:
        vision: State-only, third-person, or third-person and wrist acquisition.
        device: Explicit simulation tensor device, such as ``cuda:0``.
        producer_envs: Number of parallel physical environments.
        seed: Expert/reset random seed.
        camera_profile: Historical benchmark or close-up PickPlace calibration.

    Returns:
        Expanded Gym configuration; callers may serialize it as a run artifact.
    """
    cameras = camera_uids(vision)
    version = camera_profile_version(camera_profile)
    if type(producer_envs) is not int or producer_envs < 1:
        raise ValueError("producer_envs must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = load_task_config(device)
    # Native materials are registered by UID. Both inherited shapes otherwise
    # use "default_mat", causing the goal's green to replace the cube's orange.
    # Keep their original RGBA values and every physical property unchanged.
    for collection, object_uid in (
        ("rigid_object", "cube"),
        ("background", "goal_marker"),
    ):
        obj = next(obj for obj in config[collection] if obj["uid"] == object_uid)
        obj["shape"]["visual_material"]["uid"] = f"ods_stream_{object_uid}_material"
    config.update(
        num_envs=producer_envs,
        seed=seed,
        headless=True,
        enable_sensor=bool(cameras),
        camera_profile_version=version,
        renderer="fast-rt" if cameras else "no-render",
        render_cfg={
            "spp": 1,
            "denoising": {
                "window": "off",
                "offscreen": "optix" if cameras else "off",
            },
        },
    )
    # Replace the inherited preview camera; state-only keeps no hidden sensor.
    config["sensor"] = camera_configs(vision, camera_profile=camera_profile)
    return config


def make_engine(
    vision: VisionMode,
    device: str,
    *,
    producer_envs: int = 2,
    seed: int = 19001,
    buffer_size: int = 8,
    budget: int | None = None,
    max_episode_steps: int = 600,
    initialization_timeout: float = 1800.0,
    shutdown_timeout: float = 30.0,
    camera_profile: str = "benchmark",
) -> OnlineDataEngine:
    """Allocate, but do not start, the manual CPU-shared-memory producer.

    Args:
        vision: Frozen camera profile.
        device: Simulation device; shared data remains on CPU.
        producer_envs: Parallel expert environments, dividing ``buffer_size``.
        seed: Acquisition seed, independent of the consumer RNG.
        buffer_size: Number of rolling trajectory slots.
        budget: Optional per-episode chunk budget and equal hard ceiling.
        max_episode_steps: Physical episode and shared-row capacity.
        initialization_timeout: Initial accepted-data fill deadline in seconds.
        shutdown_timeout: Graceful producer shutdown bound in seconds.
        camera_profile: Camera calibration shared with policy evaluation.

    Returns:
        Unstarted engine with source provenance and bounded lifecycle logging.
    """
    from embodichain.data_pipeline.engine import OnlineDataEngine, OnlineDataEngineCfg

    config = load_acquisition_config(
        vision, device, producer_envs, seed, camera_profile=camera_profile
    )
    if type(buffer_size) is not int or buffer_size < 1 or buffer_size % producer_envs:
        raise ValueError("buffer_size must be positive and divisible by producer_envs")
    if type(max_episode_steps) is not int or max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    config["max_episode_steps"] = max_episode_steps
    return OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config=config,
            state_dim=9,
            action_dim=7,
            action_semantics="policy_request",
            action_contracts=("eef_pose.relative@1", "gripper.parallel@1"),
            buffer_size=buffer_size,
            max_episode_steps=max_episode_steps,
            buffer_device="cpu",
            refill_policy="manual",
            include_provenance=True,
            episode_sample_budget=budget,
            episode_sample_hard_limit=budget,
            lifecycle_event_capacity=2 * buffer_size,
            initialization_timeout=initialization_timeout,
            shutdown_timeout=shutdown_timeout,
        )
    )


def poll_generation_episodes(
    engine: OnlineDataEngine,
    generation: int,
    next_episode_ordinal: int,
    *,
    max_count: int,
) -> list[TensorDict]:
    """Copy an available ordered prefix without waiting for a generation.

    The owner advances the global ``next_episode_ordinal`` by the number
    returned, and copies all rows before requesting another generation.
    The producer publishes parallel episodes together, so a completed subbatch
    is readable while it generates subsequent subbatches. A busy publication
    lock or an unpublished next episode returns an empty list; this function
    never waits for a writer or consumes the engine's sampling quota.

    Args:
        engine: Started owner-side manual engine with provenance enabled.
        generation: Requested generation whose rows have not been replaced.
        next_episode_ordinal: Next global accepted episode ordinal. Generation
            ``g`` contains ordinals ``(g - 1) * buffer_size`` through
            ``g * buffer_size - 1``; its end ordinal is also accepted.
        max_count: Maximum owned episode copies to return, bounded by the
            caller's available retention slots. Zero performs only health and
            generation checks.

    Returns:
        Independent valid-prefix TensorDicts in consecutive slot order. Advance
        the ordinal only after taking ownership of these returned episodes.

    Raises:
        RuntimeError: The worker failed, a later generation was requested, or
            an expected publication has been lost.
        ValueError: Arguments or published validity/provenance are inconsistent.
    """
    import torch

    if type(generation) is not int or generation < 1:
        raise ValueError("generation must be a positive integer")
    generation_start = (generation - 1) * engine.buffer_size
    generation_end = generation * engine.buffer_size
    if (
        type(next_episode_ordinal) is not int
        or not generation_start <= next_episode_ordinal <= generation_end
    ):
        raise ValueError("next_episode_ordinal must belong to the requested generation")
    if type(max_count) is not int or max_count < 0:
        raise ValueError("max_count must be a nonnegative integer")
    engine._require_owner_process("poll_generation_episodes")
    status = engine.get_flow_status()
    if status["requested_generation"] < generation:
        raise ValueError("generation has not been requested")
    if status["requested_generation"] != generation:
        raise RuntimeError("Cannot acquire a generation after a newer refill request")
    if not max_count or next_episode_ordinal == generation_end:
        return []

    publication_lock = engine._lock_index.get_lock()
    if not publication_lock.acquire(False):
        return []
    episodes = []
    try:
        engine.check_health()
        write_start, write_end = engine._lock_index[:]
        next_slot = next_episode_ordinal - generation_start
        for slot in range(next_slot, min(engine.buffer_size, next_slot + max_count)):
            if write_start <= slot < write_end:
                break
            row = engine.shared_buffer[slot]
            published_generation = row.get("ods_generation", None)
            if published_generation is None:
                raise ValueError("Incremental acquisition requires source provenance")
            first_generation = int(published_generation[0])
            if first_generation != generation:
                if (
                    first_generation > generation
                    or status["completed_generation"] >= generation
                    or 0 <= slot < write_start
                ):
                    raise RuntimeError(
                        f"Expected generation {generation} slot {slot} was not retained"
                    )
                break
            valid = row["valid"].bool()
            length = int(valid.sum())
            if length < 1 or not torch.equal(
                valid, torch.arange(valid.numel(), device=valid.device) < length
            ):
                raise ValueError("Acquisition requires a nonempty valid prefix")
            prefix = row[:length]
            continuity = prefix.get("continuity_id", None)
            if continuity is not None and (
                (continuity < 0).any() or (continuity != continuity[0]).any()
            ):
                raise ValueError("Acquisition cannot cross continuity boundaries")
            for key, expected in (
                ("ods_run_id", engine.run_id),
                ("ods_generation", generation),
                ("ods_episode_id", generation_start + slot),
            ):
                if (prefix[key] != expected).any():
                    raise ValueError(f"Episode {key} differs from publication identity")
            if not torch.equal(
                prefix["ods_frame_index"],
                torch.arange(length, device=valid.device),
            ):
                raise ValueError("Episode frame ordering is inconsistent")
            # Payload and identity must be copied under the same lock that
            # publishes the next write window. The caller may retain the copy
            # across subsequent fills without pinning shared cache slots.
            episodes.append(prefix.clone())
        engine.check_health()
    finally:
        publication_lock.release()
    return episodes


def iter_generation_episodes(
    engine: OnlineDataEngine, generation: int
) -> Iterator[TensorDict]:
    """Copy completed episodes one at a time for bounded file-writer staging.

    The caller must finish this iterator before requesting the next refill.
    Each yielded TensorDict owns an independent, valid-prefix-only copy of one
    episode, including its native images and immutable provenance. Keeping only
    a bounded writer queue avoids duplicating the whole visual cache.

    Args:
        engine: Started owner-side manual-refill engine.
        generation: Completed generation that must still occupy every slot.

    Yields:
        An independent ``[valid_steps]`` episode in cache-slot order.

    Raises:
        RuntimeError: A refill is pending or data no longer matches generation.
        ValueError: Episode validity/provenance/continuity is inconsistent.
    """
    import torch

    if type(generation) is not int or generation < 1:
        raise ValueError("generation must be a positive integer")
    engine._require_owner_process("iter_generation_episodes")
    for slot in range(engine.buffer_size):
        status = engine.get_flow_status()
        if (
            status["requested_generation"] != generation
            or status["completed_generation"] != generation
        ):
            raise RuntimeError(
                "Snapshot requires a completed, non-refilling generation"
            )
        # The core has no full-episode snapshot API. Use the same publication
        # lock as sampling and the existing BC offline collector. Never return
        # a shared-buffer view, even when the producer is nominally idle.
        with engine._lock_index.get_lock():
            if tuple(engine._lock_index[:]) != (-1, -1):
                raise RuntimeError("Snapshot cannot copy an active write window")
            row = engine.shared_buffer[slot]
            valid = row["valid"].bool()
            length = int(valid.sum())
            if length < 1 or not torch.equal(
                valid, torch.arange(valid.numel(), device=valid.device) < length
            ):
                raise ValueError("Offline acquisition requires a nonempty valid prefix")
            episode = row[:length].clone()
        continuity = episode.get("continuity_id", None)
        if continuity is not None and (
            (continuity < 0).any() or (continuity != continuity[0]).any()
        ):
            raise ValueError("Offline acquisition cannot cross continuity boundaries")
        for key, expected in (
            ("ods_run_id", engine.run_id),
            ("ods_generation", generation),
        ):
            if (episode[key] != expected).any():
                raise ValueError(f"Episode {key} differs from snapshot identity")
        if (
            (episode["ods_episode_id"] < 0).any()
            or (episode["ods_episode_id"] != episode["ods_episode_id"][0]).any()
            or not torch.equal(
                episode["ods_frame_index"],
                torch.arange(length, device=episode.device),
            )
        ):
            raise ValueError(
                "Episode source identity or frame ordering is inconsistent"
            )
        yield episode
