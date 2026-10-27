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

"""Prepare an immutable numeric view of accepted PickPlace demonstrations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets.sources import TensorDataSource
from embodichain.learning.sl import SupervisedBatch, TensorBatchSource
from examples.ods.simple_bc_demo.observations import _geometry_statistics

from ._pick_place import pose_targets

__all__: list[str] = []

ACTION_CONTRACTS = ("eef_pose.relative@1", "gripper.parallel@1")
ACTION_SEMANTICS = "policy_request:eef_pose.relative@1+gripper.parallel@1"
OBSERVATION_SCHEMA = "pick_place.geometry@2"
STEP_DT = 1 / 30


def validate_environment(config: Mapping[str, Any]) -> None:
    """Validate the recorded task, clocks and actual action-controller contract."""
    if (
        config.get("id")
        != "examples.ods.simple_bc_demo.environment:SimpleBCPickPlace-Franka-v1"
        or config.get("physics") != "default"
    ):
        raise ValueError(
            "State BC requires the original default-physics Franka PickPlace task"
        )
    physics = config.get("physics_config")
    env = config.get("env")
    robot = config.get("robot")
    if (
        not isinstance(physics, Mapping)
        or not isinstance(env, Mapping)
        or not isinstance(robot, Mapping)
    ):
        raise ValueError(
            "State BC environment needs explicit physics and controller configuration"
        )
    extensions = env.get("extensions")
    if not isinstance(extensions, Mapping):
        raise ValueError("State BC requires an explicit pose-control configuration")
    interval = physics.get("physics_dt")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isclose(interval, 1 / 240, rel_tol=0, abs_tol=1e-10)
        or physics.get("enable_ccd") is not True
        or config.get("max_episode_steps") != 600
        or robot.get("class_type") != "FrankaPanda"
        or robot.get("robot_type") != "panda"
        or env.get("target_control_frequency") != 30
        or env.get("control_parts") != ["arm", "eef"]
        or extensions.get("pick_place_action_mode") != "pose"
    ):
        raise ValueError(
            "State BC requires CCD, 240 Hz physics and 30 Hz arm/eef pose control"
        )
    actions = env.get("actions")
    if not isinstance(actions, Mapping) or list(actions) != ["arm", "eef"]:
        raise ValueError("State BC requires explicit ordered action contracts")
    arm, eef = actions.get("arm", {}), actions.get("eef", {})
    if not isinstance(arm, Mapping) or not isinstance(eef, Mapping):
        raise ValueError("State BC requires arm and eef action descriptors")
    arm_params, eef_params = arm.get("params", {}), eef.get("params", {})
    if (
        not isinstance(arm_params, Mapping)
        or not isinstance(eef_params, Mapping)
        or (arm.get("contract"), eef.get("contract")) != ACTION_CONTRACTS
        or arm_params.get("part_name") != "arm"
        or arm_params.get("ik_method") != "dls"
        or arm_params.get("translation_scale") != 0.02
        or arm_params.get("rotation_scale") != 0.1
        or eef_params.get("part_name") != "eef"
        or eef_params.get("command_mode") != "continuous"
        or eef_params.get("lower_command") != {".*": 0.0}
        or eef_params.get("upper_command") != {".*": 0.04}
    ):
        raise ValueError("State BC controller frame, units or gripper contract differ")


def sha256(path: str | Path) -> str:
    """Hash files with bounded staging; RGB bytes are not decoded."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _validate_manifest(data: Mapping[str, Any]) -> None:
    if (
        data.get("complete") is not True
        or data.get("geometry_width") != 51
        or data.get("action_width") != 7
        or data.get("action_semantics") != "policy_request"
        or tuple(data.get("action_contracts", ())) != ACTION_CONTRACTS
        or data.get("step_dt") != STEP_DT
        or data.get("physics") != "default"
    ):
        raise ValueError(
            "State BC requires the declared PickPlace action/physics schema"
        )


def load_numeric(
    view_path: str | Path, *, verify_checksums: bool = True
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    """Load only aligned numeric states/actions, checking their declared identity."""
    path = Path(view_path).resolve()
    view = json.loads(path.read_text())
    _validate_manifest(view)
    if view.get("state_view_format") != "pick-place-state-view-v1":
        raise ValueError("State data view format differs")
    if view.get("cameras") != []:
        raise ValueError("A state data view must not declare cameras")
    parent = view.get("parent_manifest", {})
    if sha256(parent["path"]) != parent["sha256"]:
        raise ValueError("Parent manifest changed after state view creation")
    records = view.get("episodes", [])
    if not records or len(records) != view.get("requested_episodes"):
        raise ValueError("State data view must list every selected episode")
    observations, actions = [], []
    for record in records:
        episode_path = Path(record["path"])
        if verify_checksums and sha256(episode_path) != record["sha256"]:
            raise ValueError("State episode differs from the frozen manifest")
        with h5py.File(episode_path, "r") as episode:
            geometry = torch.from_numpy(episode["geometry"][:]).float()
            request = torch.from_numpy(episode["actions"][:]).float()
        length = record["length"]
        if (
            type(length) is not int
            or length < 1
            or geometry.shape != (length, 51)
            or request.shape != (length, 7)
            or not torch.isfinite(geometry).all()
            or not torch.isfinite(request).all()
            or bool((request.abs() > 1.000001).any())
        ):
            raise ValueError("State episode has invalid aligned observations/actions")
        observations.append(geometry)
        actions.append(request)
    return observations, actions, view


def prepare_state_view(
    manifest_path: str | Path, output: str | Path, *, episode_count: int = 32
) -> dict[str, Any]:
    """Freeze accepted episodes and state-only statistics in a new directory."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if type(episode_count) is not int or episode_count < 1:
        raise ValueError("episode_count must be positive")
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    records = manifest.get("episodes", [])
    if len(records) < episode_count:
        raise ValueError("The corpus contains fewer than the requested episodes")
    selected = records[:episode_count]
    if any(record.get("physical_success") is not True for record in selected):
        raise ValueError("Every selected demonstration must have physical success")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("State view output must be a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    view = {
        "version": 1,
        "state_view_format": "pick-place-state-view-v1",
        "complete": True,
        "parent_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "requested_episodes": episode_count,
        "episodes": [
            {**record, "path": str((manifest_path.parent / record["path"]).resolve())}
            for record in selected
        ],
        "cameras": [],
        "geometry_width": 51,
        "action_width": 7,
        "action_semantics": "policy_request",
        "action_contracts": list(ACTION_CONTRACTS),
        "step_dt": STEP_DT,
        "physics": "default",
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_timing": "pre_action",
        "state_only": True,
    }
    view_path = output / "manifest.json"
    _write_json(view_path, view)
    observations, actions, _ = load_numeric(view_path)
    geometry, requests = torch.cat(observations), torch.cat(actions)
    # Historical normalization uses the first 32 training episodes, not
    # padded frames, future observation features or development data.
    initial_geometry = torch.cat(observations[: min(32, len(observations))])
    initial_actions = torch.cat(actions[: min(32, len(actions))])
    mean, std = _geometry_statistics(initial_geometry)
    targets = pose_targets(initial_geometry, initial_actions)
    target_mean = targets.mean(0)
    target_std = targets.std(0, correction=0).clamp_min(0.05)
    target_mean[3:], target_std[3:] = 0, 1
    statistics = {
        "maniskill_action": {
            "observation_mean": torch.zeros(51),
            "observation_std": torch.ones(51),
            "target_mean": torch.zeros(7),
            "target_std": torch.ones(7),
        },
        "legacy_chunk": {
            "observation_mean": mean,
            "observation_std": std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
    }
    stats = output / "normalization.pt"
    with stats.open("xb") as stream:
        torch.save(statistics, stream)
    report = {
        "view": str(view_path),
        "view_sha256": sha256(view_path),
        "parent_manifest_sha256": view["parent_manifest"]["sha256"],
        "normalization": str(stats),
        "normalization_sha256": sha256(stats),
        "episodes": len(observations),
        "valid_frames": len(geometry),
        "observation_max_abs": float(geometry.abs().max()),
        "observation_count_above_action_model_clip": int((geometry.abs() > 1e6).sum()),
        "action_max_abs": float(requests.abs().max()),
        "normalization_episodes": min(32, len(observations)),
        "cameras_loaded": False,
    }
    _write_json(output / "prepared.json", report)
    return report


def state_schema() -> dict[str, Any]:
    """Describe feature order, controller semantics and pre-action timing."""
    names = (
        [f"qpos_{i}" for i in range(9)]
        + [f"qvel_{i}" for i in range(9)]
        + [f"tcp_position_{axis}" for axis in "xyz"]
        + [f"tcp_rotation6d_{i}" for i in range(6)]
        + [f"cube_position_{axis}" for axis in "xyz"]
        + [f"cube_rotation6d_{i}" for i in range(6)]
        + [f"goal_position_{axis}" for axis in "xyz"]
        + [f"cube_minus_tcp_{axis}" for axis in "xyz"]
        + [f"goal_minus_cube_{axis}" for axis in "xyz"]
        + [f"cube_velocity_{i}" for i in range(6)]
    )
    assert len(names) == 51
    return {
        "version": OBSERVATION_SCHEMA,
        "observation": {"names": names, "dtype": "float32", "timing": "pre_action"},
        "action": {
            "names": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
            "dtype": "float32",
            "semantics": "policy_request",
            "step_dt": STEP_DT,
            "terms": [{"term": {"contract": name}} for name in ACTION_CONTRACTS],
            "translation_scale_m": 0.02,
            "rotation_scale_rad": 0.1,
        },
    }


def resident_source(view_path: str | Path) -> TensorDataSource:
    """Build an immutable episode-major numeric source for causal windows."""
    observations, actions, view = load_numeric(view_path)
    rows, steps = len(observations), max(map(len, observations))
    geometry = torch.zeros(rows, steps, 51, dtype=torch.float32)
    request = torch.zeros(rows, steps, 7, dtype=torch.float32)
    valid = torch.zeros(rows, steps, dtype=torch.bool)
    for row, (obs, act) in enumerate(zip(observations, actions, strict=True)):
        geometry[row, : len(obs)].copy_(obs)
        request[row, : len(act)].copy_(act)
        valid[row, : len(obs)] = True
    data = TensorDict(
        {"obs": geometry, "actions": request, "valid": valid}, batch_size=[rows, steps]
    )
    return TensorDataSource(
        data,
        schema=state_schema(),
        source_id="pick-place-state-view:" + sha256(view_path),
        copy_data=False,
        metadata={
            "view_sha256": sha256(view_path),
            "parent_manifest_sha256": view["parent_manifest"]["sha256"],
        },
    )


def frame_source(view_path: str | Path) -> TensorBatchSource:
    """Draw each real recorded action frame uniformly with replacement."""
    observations, actions, view = load_numeric(view_path)
    geometry, request = torch.cat(observations), torch.cat(actions)
    batch = SupervisedBatch(
        {"history": geometry[:, None]},
        request[:, None],
        {"target_valid": torch.ones(len(request), 1, dtype=torch.bool)},
        {
            "view_sha256": sha256(view_path),
            "parent_manifest_sha256": view["parent_manifest"]["sha256"],
            "schema": state_schema(),
            "sampling": "uniform-valid-action-frame-with-replacement@1",
        },
    )
    return TensorBatchSource(batch, source_id="pick-place-frame:" + sha256(view_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_state_view(
                args.manifest, args.output_dir, episode_count=args.episodes
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
