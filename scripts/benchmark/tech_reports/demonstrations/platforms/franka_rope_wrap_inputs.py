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
"""Shared scene inputs, recorded controls, and measurements for Franka rope wrap."""

from __future__ import annotations

from hashlib import file_digest
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

_CONFIGS = Path(__file__).resolve().parents[5] / "embodichain/extensions/tasks/configs"
_TASK = _CONFIGS / "tasks/manipulation/cable/franka_rope_wrap"
TASK_CONFIG = _TASK / "task.franka.yaml"
JOINT_NAMES = (
    *(f"fr3_joint{i}" for i in range(1, 8)),
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)
_WRAP_SKILLS = {
    "simulation.cable_pickup": "pick_up",
    "simulation.cable_wrap_phase": "move_joints",
}


def _scene_paths() -> tuple[Path, Path, Path]:
    task = yaml.safe_load(TASK_CONFIG.read_text())
    return (
        _TASK / "config.json",
        _TASK / task["environment"]["component"],
        _TASK / task["embodiment"]["component"],
    )


def load_scene() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Read the scene referenced by the canonical Task Program entrypoint."""
    config_path, env_path, robot_path = _scene_paths()
    config = json.loads(config_path.read_text())
    env = yaml.safe_load(env_path.read_text())
    robot = yaml.safe_load(robot_path.read_text())["simulation"]
    return config, env, robot, _cable_usd(env_path, env)


def _cable_usd(env_path: Path, env: dict[str, Any]) -> Path:
    return env_path.parent / env["simulation"]["usd_scene"]["source"]["file_path"]


def file_info(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def input_metadata(recording: Path, robot_urdf: Path) -> dict[str, dict[str, str]]:
    config_path, env_path, robot_path = _scene_paths()
    deployment = yaml.safe_load(TASK_CONFIG.read_text())
    program = deployment["task_program"]
    return {
        name: file_info(path)
        for name, path in {
            "recording": recording,
            "robot_urdf": robot_urdf,
            "cable_usd": _cable_usd(env_path, yaml.safe_load(env_path.read_text())),
            "task_config": config_path,
            "environment": env_path,
            "embodiment": robot_path,
            "deployment": TASK_CONFIG,
            "program": _TASK / program["program"],
            "integration": _TASK / program["integration"],
            "execution_policy": _TASK / program["execution_policy"],
            "evaluation": _TASK / deployment["evaluation"]["component"],
        }.items()
    }


def load_commands(
    path: Path, *, control_hz: int, substeps: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the position and velocity targets of a completed Franka wrap recording."""
    import torch
    from tensordict import TensorDict
    from tensordict._reductions import _make_td

    # The recording also contains TensorDict state fields. Only read the action
    # tensor and primitive metadata after scoped weights-only deserialization.
    with torch.serialization.safe_globals([TensorDict, _make_td]):
        recording = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta = recording["meta"]
    steps = meta.get("num_steps")
    if (
        meta.get("artifact_purpose") != "recording"
        or meta.get("action_kind") != "expert"
        or meta.get("expert_trajectory_schema_version") != 1
        or meta.get("joint_command_mode") != "position_velocity"
        or meta.get("robot_uid") != "franka"
        or meta.get("robot_dof") != 9
        or meta.get("initial_state_step") != 0
        or meta.get("num_envs") != 1
        or type(steps) is not int
        or steps < 2 * control_hz
        or meta.get("lengths") != [steps]
        or meta.get("qpos_slice") != [0, 9]
        or meta.get("qvel_slice") != [9, 18]
    ):
        raise ValueError(
            "Expected one complete Franka expert position/velocity recording."
        )
    if (
        meta.get("control_frequency") != control_hz
        or meta.get("sim_steps_per_control") != substeps
        or not math.isclose(meta["step_dt"], 1 / control_hz, rel_tol=1e-8)
        or not math.isclose(
            meta["physics_dt"], 1 / (control_hz * substeps), rel_tol=1e-8
        )
    ):
        raise ValueError("Recording and scene control/physics clocks differ.")

    joint_names = tuple(meta["joint_names"])
    if (
        len(joint_names) != len(JOINT_NAMES)
        or set(joint_names) != set(JOINT_NAMES)
        or sorted(meta["active_joint_ids"]) != list(range(len(JOINT_NAMES)))
    ):
        raise ValueError(
            "Expected all nine Franka joint names in the recorded action layout."
        )
    order = [joint_names.index(name) for name in JOINT_NAMES]
    actions = recording["actions"]
    if (
        actions.shape != (1, steps, 18)
        or actions.dtype != torch.float32
        or not torch.isfinite(actions).all().item()
    ):
        raise ValueError("Expected finite float32 [1, steps, 18] recorded targets.")

    episodes = meta["demo_episodes"]
    if len(episodes) != 1:
        raise ValueError("Expected a single recorded task episode.")
    episode = episodes[0]
    if (
        episode.get("instruction_source") != "task_program"
        or episode.get("completed") is not True
        or episode.get("success") is not True
        or episode.get("length") != steps
    ):
        raise ValueError(
            "Expected a completed and accepted Task Program source episode."
        )

    phases = []
    start = 0
    for segment in episode["segments"]:
        details = segment["metadata"]
        calls = details["runtime"]["calls"]
        end = segment["end_step"]
        semantic_id = calls[0].get("semantic_id") if len(calls) == 1 else None
        skill_id = _WRAP_SKILLS.get(semantic_id)
        if (
            details.get("task_program_id") != "franka_rope_wrap"
            or skill_id is None
            or calls[0].get("skill_id") != skill_id
            or calls[0].get("command_count") != end - start
            or segment.get("start_step") != start
            or not start < end <= steps
        ):
            raise ValueError(
                "Recording does not contain continuous supported Franka wrap commands."
            )
        phases.append(
            {
                "name": segment["name"],
                "semantic_id": semantic_id,
                "skill_id": skill_id,
                "start_step": start,
                "end_step": end,
            }
        )
        start = end
    if start != steps or phases[-1]["name"] != "hold_closed":
        raise ValueError("Expected continuous wrap phases ending in a closed grasp.")

    targets = actions[0].numpy()
    provenance = {
        "episode_uuid": episode["episode_uuid"],
        "program_hash": meta["program_hash"],
        "seed": episode.get("seed"),
        "task_program_id": "franka_rope_wrap",
        "joint_names": list(JOINT_NAMES),
        "recorded_joint_names": list(joint_names),
        "phases": phases,
    }
    return targets[:, :9][:, order].copy(), targets[:, 9:][:, order].copy(), provenance


def geometry_summary(
    poses: np.ndarray,
    segment_lengths: np.ndarray,
    post_xy: list[float],
    control_hz: int,
) -> dict[str, Any]:
    """Measure winding and settling from newly simulated XYZW segment poses."""
    hold_frames = 2 * control_hz + 1
    if len(poses) < hold_frames:
        raise ValueError("Expected a complete two-second hold window.")
    segment_poses = poses[-1]
    tail = poses[-hold_frames:, -1]
    tail_axis = Rotation.from_quat(tail[:, 3:]).as_matrix()[:, :, 2]
    tail_positions = tail[:, :3] + tail_axis * segment_lengths[-1] / 2
    displacement = np.linalg.norm(tail_positions - tail_positions[0], axis=1)
    axis = Rotation.from_quat(segment_poses[:, 3:]).as_matrix()[:, :, 2]
    offsets = axis * segment_lengths[:, None] / 2
    starts, ends = segment_poses[:, :3] - offsets, segment_poses[:, :3] + offsets
    nodes = np.vstack((starts[:1], (ends[:-1] + starts[1:]) / 2, ends[-1:]))
    xy = nodes[:, :2] - post_xy
    angle = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    return {
        "winding_deg": float(np.rad2deg(angle[-1] - angle[0])),
        "free_end_m": nodes[-1].tolist(),
        "hold_max_free_end_displacement_m": float(displacement.max()),
        "hold_window_s": 2.0,
        "max_joint_gap_m": float(np.linalg.norm(ends[:-1] - starts[1:], axis=1).max()),
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
