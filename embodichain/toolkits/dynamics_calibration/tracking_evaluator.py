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

"""Built-in qpos-only trajectory evaluator for robot drive calibration."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


def evaluate(overlay: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one drive overlay on a deterministic multisine trajectory.

    This callable is designed for the isolated calibration worker. Evaluator
    payload options are passed under ``context["payload"]``:

    ``robot_cfg``
        Overrides for :class:`~embodichain.lab.sim.cfg.RobotCfg`. The first
        configured asset is used as ``fpath`` unless explicitly repeated.
    ``control_part``
        Optional Robot control-part name. All active joints are used otherwise.
    ``training_trajectory`` / ``qualification_trajectory``
        Mappings with ``duration_seconds``, ``warmup_seconds``, ``amplitude``
        (scalar or per-joint radians), and ``frequencies_hz`` (scalar or list).
    ``renderer``
        Headless renderer selection, defaulting to ``hybrid``.

    Args:
        overlay: RobotCfg-compatible candidate drive overlay.
        context: Asset, timing, backend, phase, seed, and evaluator payload
            supplied by the calibration worker.

    Returns:
        Raw observations consumed by :func:`compute_tracking_metrics`.
    """
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, RobotCfg

    payload = context.get("payload", {})
    if not isinstance(payload, Mapping):
        raise TypeError("evaluator payload must be a mapping")
    backend = str(context["backend"])
    if backend not in {"default", "physx"}:
        raise ValueError(
            f"built-in tracking evaluator supports 'default'/'physx', got {backend!r}"
        )

    robot_data = copy.deepcopy(dict(payload.get("robot_cfg", {})))
    assets = context.get("assets", [])
    if not assets:
        raise ValueError("evaluation context contains no robot asset")
    robot_data.setdefault("fpath", str(assets[0]["path"]))
    robot_data.setdefault("build_pk_chain", False)
    robot_data.setdefault("solver_cfg", None)
    _merge_drive_properties(robot_data, overlay)

    renderer = str(payload.get("renderer", "hybrid"))
    sim_cfg = SimulationManagerCfg(
        headless=True,
        num_envs=1,
        physics_dt=float(context["physics_dt"]),
        sim_device=str(context.get("device", "cpu")),
        render_cfg=RenderCfg(renderer=renderer),
    )
    simulation = SimulationManager(sim_cfg)
    robot = simulation.add_robot(RobotCfg.from_dict(robot_data))
    if robot is None:
        raise RuntimeError("SimulationManager failed to create the calibration robot")

    control_part_value = payload.get("control_part")
    control_part = None if control_part_value is None else str(control_part_value)
    joint_ids = robot.get_joint_ids(name=control_part)
    if not joint_ids:
        raise ValueError(f"control part {control_part!r} resolved to no active joints")
    joint_names = [robot.joint_names[index] for index in joint_ids]
    initial = robot.get_qpos(name=control_part)[0].detach().clone()
    limits = robot.get_qpos_limits(name=control_part)[0].detach().clone()

    phase = str(context["phase"])
    trajectory_key = f"{phase}_trajectory"
    trajectory = payload.get(trajectory_key, payload.get("trajectory", {}))
    if not isinstance(trajectory, Mapping):
        raise TypeError(f"{trajectory_key} must be a mapping")
    duration = _positive_float(
        trajectory.get("duration_seconds", 4.0 if phase == "qualification" else 3.0),
        f"{trajectory_key}.duration_seconds",
    )
    warmup = _nonnegative_float(
        trajectory.get("warmup_seconds", 0.5),
        f"{trajectory_key}.warmup_seconds",
    )
    requested_hz = float(context["requested_control_hz"])
    actual_hz = float(context["actual_control_hz"])
    physics_steps = int(context["physics_steps_per_control"])
    sample_count = max(1, int(round(duration * actual_hz)))
    warmup_updates = int(math.ceil(warmup / float(context["physics_dt"])))
    amplitudes = _joint_values(
        trajectory.get("amplitude", 0.1), len(joint_ids), "amplitude"
    ).to(device=initial.device, dtype=initial.dtype)
    default_frequency = 0.35 if phase == "qualification" else 0.25
    frequencies = _joint_values(
        trajectory.get("frequencies_hz", default_frequency),
        len(joint_ids),
        "frequencies_hz",
    ).to(device=initial.device, dtype=initial.dtype)
    if torch.any(frequencies <= 0.0):
        raise ValueError("trajectory frequencies must be greater than zero")
    phase_offsets = torch.arange(
        len(joint_ids), dtype=initial.dtype, device=initial.device
    ) * (math.pi / max(1, len(joint_ids)))
    if phase == "qualification":
        phase_offsets = phase_offsets + math.pi / 5.0

    robot.set_qpos(initial.unsqueeze(0), target=False, name=control_part)
    robot.set_qpos(initial.unsqueeze(0), target=True, name=control_part)
    if warmup_updates:
        simulation.update(step=warmup_updates)
    target_qvel_writes_before = robot.target_qvel_write_count

    target_rows: list[list[float]] = []
    actual_rows: list[list[float]] = []
    effort_rows: list[list[float]] = []
    velocity_rows: list[list[float]] = []
    for sample_index in range(sample_count):
        time_seconds = sample_index / actual_hz
        target = initial + amplitudes * torch.sin(
            2.0 * math.pi * frequencies * time_seconds + phase_offsets
        )
        target = torch.minimum(torch.maximum(target, limits[:, 0]), limits[:, 1])
        robot.set_qpos(target.unsqueeze(0), target=True, name=control_part)
        simulation.update(step=physics_steps)
        target_rows.append(target.detach().cpu().tolist())
        actual_rows.append(robot.get_qpos(name=control_part)[0].detach().cpu().tolist())
        effort_rows.append(robot.get_qf(name=control_part)[0].detach().cpu().tolist())
        velocity_rows.append(
            robot.get_qvel(name=control_part)[0].detach().cpu().tolist()
        )

    result: dict[str, Any] = {
        "joint_names": joint_names,
        "target_qpos": target_rows,
        "actual_qpos": actual_rows,
        "requested_control_hz": requested_hz,
        "actual_control_hz": actual_hz,
        "target_qvel_write_count": (
            robot.target_qvel_write_count - target_qvel_writes_before
        ),
        "control_groups": {control_part or "all": joint_names},
        "stable": _rows_are_finite(actual_rows),
        "metadata": {
            "evaluator": "embodichain.multisine_qpos_v1",
            "phase": phase,
            "backend": backend,
            "device": str(context.get("device", "cpu")),
            "renderer": renderer,
            "physics_steps_per_control": physics_steps,
            "target_qvel_instrumentation": "Articulation.set_qvel",
        },
    }
    effort_limits = robot.get_qf_limits(name=control_part)[0].detach().cpu()
    if torch.isfinite(effort_limits).all() and torch.all(effort_limits > 0.0):
        result["effort"] = effort_rows
        result["effort_limits"] = effort_limits.tolist()
    velocity_limits = robot.get_qvel_limits(name=control_part)[0].detach().cpu()
    if torch.isfinite(velocity_limits).all() and torch.all(velocity_limits > 0.0):
        result["qvel"] = velocity_rows
        result["qvel_limits"] = velocity_limits.tolist()
    finite_limits = torch.isfinite(limits).all()
    if finite_limits:
        result["qpos_lower"] = limits[:, 0].detach().cpu().tolist()
        result["qpos_upper"] = limits[:, 1].detach().cpu().tolist()
    return result


def _merge_drive_properties(
    robot_data: dict[str, Any], overlay: Mapping[str, Any]
) -> None:
    drive_overlay = overlay.get("drive_pros")
    if not isinstance(drive_overlay, Mapping):
        raise ValueError("candidate overlay has no drive_pros mapping")
    configured = robot_data.setdefault("drive_pros", {})
    if not isinstance(configured, dict):
        raise TypeError("robot_cfg.drive_pros must be a mapping")
    for field, raw_values in drive_overlay.items():
        if not isinstance(raw_values, Mapping):
            configured[str(field)] = raw_values
            continue
        existing = configured.setdefault(str(field), {})
        if not isinstance(existing, dict):
            existing = {}
            configured[str(field)] = existing
        existing.update({str(key): float(value) for key, value in raw_values.items()})


def _joint_values(raw: Any, count: int, name: str):
    import torch

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != count:
            raise ValueError(f"trajectory {name} must contain {count} values")
        values = [float(value) for value in raw]
    else:
        values = [float(raw)] * count
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"trajectory {name} must contain finite non-negative values")
    return torch.tensor(values, dtype=torch.float32)


def _positive_float(raw: Any, name: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _nonnegative_float(raw: Any, name: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _rows_are_finite(rows: list[list[float]]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row)


__all__ = ["evaluate"]
