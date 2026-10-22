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
    "load_acquisition_config",
    "make_engine",
    "iter_generation_episodes",
]

VisionMode = Literal["state", "single", "dual"]

# The original +Y wrist mount self-occluded. Version 2 qualified a fixed
# +X-side mount with SPP=1/OptiX, without temporal reconstruction or
# super-resolution. Version 3 keeps those settings and separates the inherited
# cube/goal material identities so their existing colors remain independent.
CAMERA_PROFILE_VERSION = "v3"


def camera_uids(vision: VisionMode) -> tuple[str, ...]:
    """Return the frozen camera order, rejecting unknown profiles."""
    if vision == "state":
        return ()
    if vision == "single":
        return ("cam_high",)
    if vision == "dual":
        return ("cam_high", "cam_wrist")
    raise ValueError(f"Unknown vision profile: {vision!r}")


def _camera(uid: str, extrinsics: dict[str, Any]) -> dict[str, Any]:
    # A square pinhole camera with horizontal/vertical FOV=60 degrees.
    focal = 128.0 / math.tan(math.radians(30.0))
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


def load_acquisition_config(
    vision: VisionMode,
    device: str,
    producer_envs: int = 2,
    seed: int = 19001,
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

    Returns:
        Expanded Gym configuration; callers may serialize it as a run artifact.
    """
    cameras = camera_uids(vision)
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
        camera_profile_version=CAMERA_PROFILE_VERSION,
        renderer="fast-rt" if cameras else "no-render",
        render_cfg={
            "spp": 1,
            "denoising": {
                "window": "off",
                "offscreen": "optix" if cameras else "off",
            },
        },
    )
    # Replace the inherited 640x480 preview camera; state-only must not leave
    # a hidden camera or acquire images solely for report/video generation.
    config["sensor"] = []
    if cameras:
        config["sensor"].append(
            _camera(
                "cam_high",
                {
                    "eye": [-0.50, -0.90, 0.88],
                    "target": [-0.50, 0.10, 0.20],
                    "up": [0.0, 0.0, 1.0],
                },
            )
        )
    if vision == "dual":
        config["sensor"].append(
            _camera(
                "cam_wrist",
                {
                    "parent": "FrankaPanda/fr3_hand_tcp",
                    "eye": [0.10, 0.0, -0.10],
                    "target": [0.0, 0.0, 0.03],
                    "up": [0.0, 1.0, 0.0],
                },
            )
        )
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

    Returns:
        Unstarted engine with source provenance and bounded lifecycle logging.
    """
    from embodichain.data_pipeline.engine import OnlineDataEngine, OnlineDataEngineCfg

    config = load_acquisition_config(vision, device, producer_envs, seed)
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
