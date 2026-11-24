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
"""Load portable scene inputs and actual Franka RJ45 expert commands."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from franka_rope_wrap_inputs import JOINT_NAMES, file_info, sha256

__all__ = [
    "JOINT_NAMES",
    "TASK_CONFIG",
    "TASK_ID",
    "load_asset_manifest",
    "load_commands",
    "load_measurements",
    "load_scene",
    "save_asset_manifest",
    "save_results",
    "scene_inputs",
]

_TASK_PACKAGE = Path(__file__).resolve().parents[5] / "embodichain/extensions/tasks"
_TASK = _TASK_PACKAGE / "configs/tasks/manipulation/cable/franka_rj45_cycle"
TASK_CONFIG = _TASK / "task.franka.yaml"
TASK_ID = "franka_rj45_cycle"


def load_scene() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read the scene and embodiment referenced by the standard deployment."""
    import yaml

    deployment = yaml.safe_load(TASK_CONFIG.read_text())
    return (
        json.loads((_TASK / "config.json").read_text()),
        yaml.safe_load((_TASK / deployment["environment"]["component"]).read_text()),
        yaml.safe_load((_TASK / deployment["embodiment"]["component"]).read_text())[
            "simulation"
        ],
    )


def scene_inputs(recording: Path, models: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Identify the recording, physical configuration and prepared asset files."""
    import yaml

    deployment = yaml.safe_load(TASK_CONFIG.read_text())
    paths = {
        "recording": recording,
        "deployment": TASK_CONFIG,
        "configuration": _TASK / "config.json",
        "environment": _TASK / deployment["environment"]["component"],
        "embodiment": _TASK / deployment["embodiment"]["component"],
        "evaluation": _TASK / deployment["evaluation"]["component"],
        **{name: _TASK / path for name, path in deployment["task_program"].items()},
        **{f"{name}_urdf": path for name, path in models.items()},
    }
    return {name: file_info(path) for name, path in paths.items()}


def load_commands(
    path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode the accepted source episode and reorder actual targets by joint name."""
    import torch
    from tensordict import TensorDict
    from tensordict._reductions import _make_td

    with torch.serialization.safe_globals([TensorDict, _make_td]):
        recording = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    hz = config["control_hz"]
    substeps = config["substeps"]
    steps = round(config["duration_s"] * hz)
    dof = len(JOINT_NAMES)
    meta = recording["meta"]
    expected = {
        "artifact_purpose": "recording",
        "action_kind": "expert",
        "expert_trajectory_schema_version": 1,
        "joint_command_mode": "position_velocity",
        "robot_uid": "franka",
        "robot_dof": dof,
        "initial_state_step": 0,
        "num_envs": 1,
        "num_steps": steps,
        "lengths": [steps],
        "qpos_slice": [0, dof],
        "qvel_slice": [dof, 2 * dof],
        "control_frequency": hz,
        "sim_steps_per_control": substeps,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Expected RJ45 recording metadata {key}={value!r}.")
    if not math.isclose(meta["step_dt"], 1 / hz, rel_tol=1e-8) or not math.isclose(
        meta["physics_dt"], 1 / (hz * substeps), rel_tol=1e-8
    ):
        raise ValueError("Recording and RJ45 scene use different clocks.")
    names = tuple(meta["joint_names"])
    if (
        len(names) != dof
        or set(names) != set(JOINT_NAMES)
        or sorted(meta["active_joint_ids"]) != list(range(dof))
    ):
        raise ValueError("Expected nine active named FR3 joints in the recording.")
    actions = recording["actions"]
    if (
        actions.shape != (1, steps, 2 * dof)
        or actions.dtype != torch.float32
        or not torch.isfinite(actions).all().item()
    ):
        raise ValueError(
            f"Expected finite float32 targets with shape (1, {steps}, 18)."
        )
    (episode,) = meta["demo_episodes"]
    if (
        episode.get("instruction_source") != "task_program"
        or episode.get("completed") is not True
        or episode.get("success") is not True
        or episode.get("length") != steps
    ):
        raise ValueError("Expected a completed and accepted RJ45 Task Program episode.")
    phase_end = 0
    phases = []
    for segment in episode["segments"]:
        details = segment["metadata"]
        start, end = segment["start_step"], segment["end_step"]
        if (
            details.get("task_program_id") != TASK_ID
            or start != phase_end
            or not start < end <= steps
        ):
            raise ValueError("Expected consecutive Franka RJ45 Task Program stages.")
        phases.append({"name": segment["name"], "start_step": start, "end_step": end})
        phase_end = end
    if phase_end != steps:
        raise ValueError("Task Program stage evidence does not cover every command.")
    order = [names.index(name) for name in JOINT_NAMES]
    targets = actions[0].numpy()
    provenance = {
        "episode_uuid": episode["episode_uuid"],
        "program_hash": meta["program_hash"],
        "seed": episode.get("seed"),
        "task_program_id": TASK_ID,
        "recorded_joint_names": list(names),
        "stages": phases,
    }
    return targets[:, order].copy(), targets[:, np.add(order, dof)].copy(), provenance


def save_asset_manifest(path: Path, models: dict[str, Path]) -> None:
    """Write references to the three prepared URDFs used by the source task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(
            {name: file_info(file) for name, file in models.items()}, stream, indent=2
        )
        stream.write("\n")


def load_asset_manifest(path: Path) -> dict[str, Path]:
    """Verify and resolve the asset references exported by EmbodiChain."""
    manifest = json.loads(path.read_text())
    if set(manifest) != {"robot", "plug", "socket"}:
        raise ValueError("Expected a manifest with robot, plug and socket URDFs.")
    paths = {
        name: Path(entry["path"]).resolve(strict=True)
        for name, entry in manifest.items()
    }
    for name, file in paths.items():
        if manifest[name]["sha256"] != sha256(file):
            raise ValueError(f"The prepared {name} URDF differs from the source task.")
    return paths


def load_measurements(config: dict[str, Any]) -> SimpleNamespace:
    """Load the shared rule graph and contact cadence for this replay."""
    import yaml

    from embodichain.lab.evaluation._episode_geometry import _connector_frame
    from embodichain.lab.evaluation._episode_reduction import (
        _evaluate_scalar_checks,
        _reduce_measurements,
    )

    deployment = yaml.safe_load(TASK_CONFIG.read_text())
    evaluation = yaml.safe_load(
        (TASK_CONFIG.parent / deployment["evaluation"]["component"]).read_text()
    )
    measurements = evaluation["measurements"]
    samples = measurements["samples_per_control"]
    substeps = config["substeps"]
    if type(samples) is not int or samples < 1 or substeps % samples:
        raise ValueError(
            "RJ45 samples_per_control must be a positive integer that divides the physics substeps."
        )
    diagnostic_rate_hz = config["control_hz"] * samples
    parameters = {**measurements["parameters"], "control_hz": config["control_hz"]}
    graph = evaluation["graph"]
    (root,) = graph["roots"]

    def measure(**frame: Any) -> dict[str, Any]:
        return _connector_frame(parameters, **frame)

    def evaluate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        values = _reduce_measurements(rows, measurements["reductions"])
        results = _evaluate_scalar_checks(graph, values)
        checks = {name: results[name] for name in graph["rules"][root]["refs"]}
        passed = results[root]
        pressed = [row for row in rows if 24 <= row["time_s"] <= 25]
        retained = [row["depth_m"] for row in rows if 18 <= row["time_s"] <= 19]
        return {
            "passed": passed,
            "checks": checks,
            "quality_status": "floor_cycle_verified" if passed else "demo_candidate",
            "diagnostic_rate_hz": diagnostic_rate_hz,
            "peak_penetration_m": values.get("peak_contact_penetration"),
            "peak_attachment_error_m": values.get("cable_attachment"),
            "peak_latch_limit_violation_deg": (
                float(np.rad2deg(values["latch_limits"]))
                if "latch_limits" in values
                else None
            ),
            "pressed_latch_min_deg": min(
                (row["latch_angle_deg"] for row in pressed), default=None
            ),
            "press_contact_frames": sum(
                row["finger_latch_contact"] is True for row in pressed
            ),
            "final_return_error_m": rows[-1]["return_error_m"] if rows else None,
            "retained_depth_range_m": (
                [min(retained), max(retained)] if retained else None
            ),
        }

    return SimpleNamespace(
        measure_frame=measure,
        evaluate=evaluate,
        parameters=parameters,
        samples_per_control=samples,
        substeps_per_sample=substeps // samples,
        diagnostic_rate_hz=diagnostic_rate_hz,
    )


def save_results(
    output: Path,
    trajectory_data: dict[str, np.ndarray],
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Persist fresh replay measurements and their provenance."""
    trajectory = output / "trajectory.npz"
    np.savez_compressed(trajectory, **trajectory_data)
    (output / "metrics.json").write_text(
        json.dumps({"rows": rows}, indent=2, allow_nan=False) + "\n"
    )
    report["trajectory_sha256"] = sha256(trajectory)
    summary = output / "result.json"
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Completed {report['control_steps']} controls. Result: {summary}", flush=True
    )
