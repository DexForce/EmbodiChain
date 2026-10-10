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

"""Compare CobotMagic drive limits and gains with gravity enabled.

Run: python -m scripts.benchmark.robotics.cobotmagic_drives
The profiles run in isolated arenas of one batched Default-physics simulation.
The source URDF limits are simulation references, not measured motor ratings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
import psutil
import torch

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult
    from embodichain.lab.sim.objects import Robot

__all__ = ["main", "run_all_benchmarks"]

_PROFILES = {
    "configured": None,
    "legacy": ([70000] * 6, [1000] * 6, 3e6, 3e3, False),
    "source_limits": ([70000] * 6, [1000] * 6, 100, 10, True),
    "medium": (
        [30000, 30000, 20000, 12000, 8000, 4000],
        [600, 600, 400, 240, 160, 80],
        100,
        10,
        True,
    ),
    "soft": (
        [10000, 10000, 8000, 5000, 3000, 2000],
        [200, 200, 160, 100, 60, 40],
        100,
        10,
        True,
    ),
}
_HOME = [
    -0.3,
    0.3,
    1.0,
    1.0,
    -1.2,
    -1.2,
    0.0,
    0.0,
    0.6,
    0.6,
    0.0,
    0.0,
    0.05,
    0.05,
    0.05,
    0.05,
]
_SOURCE_VELOCITY_LIMITS = np.array([5, 5, 5, 5, 5, 3], dtype=float)


def _audit_urdf(path: Path) -> dict[str, object]:
    """Audit finite positive masses and physically admissible inertia tensors."""
    links = []
    for link in ET.parse(path).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        values = {
            key: float(value) for key, value in inertial.find("inertia").attrib.items()
        }
        tensor = np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ]
        )
        eigenvalues = (
            np.linalg.eigvalsh(tensor)
            if np.isfinite(tensor).all()
            else np.full(3, np.nan)
        )
        origin = inertial.find("origin")
        com = (
            np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            if origin is not None
            else np.zeros(3)
        )
        valid = bool(
            np.isfinite(mass)
            and mass > 0
            and com.shape == (3,)
            and np.isfinite(com).all()
            and np.isfinite(eigenvalues).all()
            and eigenvalues[0] > 0
            and eigenvalues[-1] <= eigenvalues[:2].sum() + 1e-8
        )
        links.append(
            {
                "link": link.get("name"),
                "mass_kg": mass,
                "com_m": com.tolist(),
                "principal_inertia": eigenvalues.tolist(),
                "valid": valid,
            }
        )
    return {
        "valid": bool(links) and all(link["valid"] for link in links),
        "links": links,
    }


def _tracking_metrics(
    measured: np.ndarray,
    velocities: np.ndarray,
    target: np.ndarray,
    *,
    kp: np.ndarray,
    kd: np.ndarray,
    effort_limit: float | np.ndarray,
    velocity_limits: np.ndarray,
    case: str,
    dt: float,
    moved_joint: int | None = None,
) -> dict[str, float | bool]:
    """Measure tracking, speed and an explicitly estimated PD demand proxy."""
    error = target - measured
    velocity_ratio = np.abs(velocities) / velocity_limits
    metrics = {
        "max_joint_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(error**2, axis=0))).max()
        ),
        "p95_error_deg": float(np.rad2deg(np.percentile(np.abs(error), 95))),
        "steady_error_deg": float(
            np.rad2deg(np.abs(error[-max(1, round(0.5 / dt)) :])).max()
        ),
        "peak_velocity_rad_s": float(np.abs(velocities).max()),
        "velocity_limit_ratio": float(velocity_ratio.max()),
        # qf is externally applied effort, not measured drive torque on Default.
        # This post-substep expression is a diagnostic proxy, not a force sensor.
        "estimated_pd_limit_fraction": float(
            (np.abs(kp * error - kd * velocities) >= effort_limit).mean()
        ),
        "estimated_peak_pd_demand": float(np.abs(kp * error - kd * velocities).max()),
        "settling_s": 0.0,
        "overshoot_percent": 0.0,
    }
    if case.startswith("step"):
        assert moved_joint is not None
        joint_error = np.abs(error[:, moved_joint])
        stays_settled = np.maximum.accumulate(joint_error[::-1])[::-1] <= 0.01
        settled = np.flatnonzero(stays_settled)
        metrics["settling_s"] = (
            float((settled[0] + 1) * dt) if len(settled) else float(len(error) * dt)
        )
        metrics["overshoot_percent"] = float(
            max(0, (measured[:, moved_joint] - target[:, moved_joint]).max())
            / 0.15
            * 100
        )
        metrics["success"] = bool(
            metrics["settling_s"] <= 0.5
            and metrics["overshoot_percent"] <= 10
            and metrics["steady_error_deg"] <= 0.25
            and metrics["velocity_limit_ratio"] <= 1.05
        )
    else:
        metrics["success"] = bool(
            metrics["max_joint_rmse_deg"] <= 0.5
            and metrics["p95_error_deg"] <= 1.0
            and metrics["velocity_limit_ratio"] <= 1.05
        )
    return metrics


def _configure_profile(robot: Robot, profile_name: str, env_id: int) -> None:
    """Apply one experimental profile to independent actuators only."""
    if profile_name == "configured":
        return
    kp, kd, arm_limit, gripper_limit, bounded_velocity = _PROFILES[profile_name]
    for part in ("left_arm", "right_arm", "left_eef", "right_eef"):
        ids = robot.get_joint_ids(part, remove_mimic=True)
        arm = part.endswith("arm")
        stiffness = kp if arm else [300] * len(ids)
        damping = kd if arm else [30] * len(ids)
        velocity = _SOURCE_VELOCITY_LIMITS.tolist() if arm else [1] * len(ids)
        robot.set_joint_drive(
            stiffness=torch.tensor([stiffness], device=robot.device),
            damping=torch.tensor([damping], device=robot.device),
            max_effort=torch.full(
                (1, len(ids)), arm_limit if arm else gripper_limit, device=robot.device
            ),
            max_velocity=(
                torch.tensor([velocity], device=robot.device)
                if bounded_velocity
                else torch.full(
                    (1, len(ids)), torch.finfo(torch.float32).max, device=robot.device
                )
            ),
            joint_ids=ids,
            env_ids=[env_id],
        )


def _tcp_positions(robot: Robot, side: str) -> np.ndarray:
    """Read the configured TCP offset from the measured wrist transform."""
    wrist = robot.get_link_pose(f"{side}_link6", to_matrix=True)
    return (wrist[:, :3, 3] + 0.143 * wrist[:, :3, 2]).detach().cpu().numpy()


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the three benchmark tables, including all evaluated profiles."""
    lines = [
        "# CobotMagic drive verification",
        "",
        "Gravity enabled; Default backend; 100 Hz physics, 25 Hz position commands.",
        "Profiles share one batched run. Timing/RSS/Torch VRAM are shared batch measurements, not isolated per-profile performance. Native renderer allocations are not included in Torch VRAM.",
        "Source limits (100 Nm arm, 10 N gripper; 5/3 rad/s arm, 1 m/s gripper) come from the asset URDF and are not measured motor ratings.",
        "Configured profiles use their effective native speed caps; pinned experiments retain source caps, and unbounded legacy motion is checked against source caps. resolved_drives.json records the evaluated gains and limits.",
        "PD demand is an estimate from post-substep position/velocity; it is not measured actuator torque or a measured saturation rate.",
        "",
        "## Time & Memory",
        "",
        "| case | profile | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "case",
                    "profile",
                    "cost_time_ms",
                    "cpu_delta_mb",
                    "gpu_delta_mb",
                    "peak_gpu_mb",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| case | profile | success_rate | max_joint_rmse_deg | steady_error_deg | tcp_error_mm | peak_velocity_rad_s | velocity_limit_ratio | settling_s | overshoot_percent | estimated_pd_limit_fraction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        keys = (
            "case",
            "profile",
            "success_rate",
            "max_joint_rmse_deg",
            "steady_error_deg",
            "tcp_error_mm",
            "peak_velocity_rad_s",
            "velocity_limit_ratio",
            "settling_s",
            "overshoot_percent",
            "estimated_pd_limit_fraction",
        )
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    ranking = []
    for profile in dict.fromkeys(row["profile"] for row in rows):
        values = [row for row in rows if row["profile"] == profile]
        ranking.append(
            (
                profile,
                sum(row["success_rate"] for row in values) / len(values),
                max(row["max_joint_rmse_deg"] for row in values),
            )
        )
    ranking.sort(key=lambda item: (-item[1], item[2]))
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | profile | overall_success_rate | worst_case_joint_rmse_deg |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {rank} | {profile} | {success} | {error} |"
        for rank, (profile, success, error) in enumerate(ranking, 1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _physical_task_metrics(
    poses: np.ndarray,
    initial: np.ndarray,
    runtime_success: list[bool],
    velocity_ratio: np.ndarray,
    idle_displacement_mm: np.ndarray,
) -> list[dict[str, object]]:
    """Check measured lift, tilt and return independently of projected effects."""
    rows = []
    for row, runtime_ok in enumerate(runtime_success):
        if not len(poses):
            rows.append(
                {
                    "runtime_success": bool(runtime_ok),
                    "measured_checks_pass": False,
                    "accepted": False,
                    "max_lift_m": 0.0,
                    "max_tilt_deg": 0.0,
                    "return_error_m": None,
                    "final_tilt_deg": None,
                    "velocity_limit_ratio": float(velocity_ratio[row]),
                    "idle_tcp_displacement_mm": float(idle_displacement_mm[row]),
                }
            )
            continue
        lift = float(poses[:, row, 2, 3].max() - initial[row, 2, 3])
        tilt = np.rad2deg(np.arccos(np.clip(poses[:, row, 2, 2], -1, 1)))
        return_error = float(
            np.linalg.norm(poses[-1, row, :3, 3] - initial[row, :3, 3])
        )
        physical_ok = bool(
            lift >= 0.08
            and tilt.max() >= 45
            and return_error <= 0.05
            and tilt[-1] <= 10
            and velocity_ratio[row] <= 1.05
            and idle_displacement_mm[row] <= 1.0
        )
        rows.append(
            {
                "runtime_success": bool(runtime_ok),
                "measured_checks_pass": physical_ok,
                "accepted": bool(runtime_ok and physical_ok),
                "max_lift_m": lift,
                "max_tilt_deg": float(tilt.max()),
                "return_error_m": return_error,
                "final_tilt_deg": float(tilt[-1]),
                "velocity_limit_ratio": float(velocity_ratio[row]),
                "idle_tcp_displacement_mm": float(idle_displacement_mm[row]),
            }
        )
    return rows


def _held_motion_metrics(
    bottle: np.ndarray,
    eef: np.ndarray,
    cup: np.ndarray,
    contacts: np.ndarray,
    *,
    held_start_s: float | None,
    release_s: float | None,
    dt: float,
    bottle_z_bounds: tuple[float, float],
    bottle_radius: float,
    cup_radius: float,
    dropped_contacts: int = 0,
) -> list[dict[str, object]]:
    """Reject in-hand drift and cup contact despite successful command execution.

    The bottle capsule and cup sphere enclose the actual collision vertices.
    Their separation is a conservative lower bound, not a visual-mesh distance.
    """
    batch = contacts.shape[1]
    if not len(bottle) or held_start_s is None or release_s is None:
        return [
            {
                "grasp_and_clearance_pass": False,
                "max_grasp_drift_mm": None,
                "max_grasp_rotation_deg": None,
                "cup_contact_substeps": int((contacts[:, row] > 0).sum()),
                "minimum_bottle_cup_clearance_mm": None,
                "cup_displacement_mm": None,
                "dropped_contacts": dropped_contacts,
            }
            for row in range(batch)
        ]
    start = max(0, round(held_start_s / dt) - 1)
    stop = min(len(bottle), round(release_s / dt))
    if stop <= start:
        raise ValueError("Held observation window must contain physics samples.")
    relative = np.linalg.inv(eef[start:stop]) @ bottle[start:stop]
    reference = relative[0]
    drift = (
        np.linalg.norm(relative[:, :, :3, 3] - reference[None, :, :3, 3], axis=-1)
        * 1000
    )
    rotation = relative[:, :, :3, :3] @ np.swapaxes(reference[None, :, :3, :3], -1, -2)
    angle = np.rad2deg(
        np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    )
    axis = bottle[start:stop, :, :3, 2]
    origin = bottle[start:stop, :, :3, 3]
    a = origin + axis * bottle_z_bounds[0]
    b = origin + axis * bottle_z_bounds[1]
    segment = b - a
    center = cup[start:stop, :, :3, 3]
    fraction = np.clip(
        np.sum((center - a) * segment, axis=-1)
        / np.maximum(np.sum(segment**2, axis=-1), 1e-12),
        0,
        1,
    )
    clearance = (
        np.linalg.norm(a + fraction[..., None] * segment - center, axis=-1)
        - bottle_radius
        - cup_radius
    ) * 1000
    movement = np.linalg.norm(cup[:, :, :3, 3] - cup[0, :, :3, 3], axis=-1) * 1000
    rows = []
    for row in range(batch):
        maximum_drift = float(drift[:, row].max())
        maximum_angle = float(angle[:, row].max())
        contact_steps = int((contacts[:, row] > 0).sum())
        minimum_clearance = float(clearance[:, row].min())
        displacement = float(movement[:, row].max())
        rows.append(
            {
                "grasp_and_clearance_pass": bool(
                    maximum_drift <= 3.0
                    and maximum_angle <= 3.0
                    and contact_steps == 0
                    and minimum_clearance >= 10.0
                    and displacement <= 2.0
                    and dropped_contacts == 0
                ),
                "max_grasp_drift_mm": maximum_drift,
                "max_grasp_rotation_deg": maximum_angle,
                "cup_contact_substeps": contact_steps,
                "minimum_bottle_cup_clearance_mm": minimum_clearance,
                "cup_displacement_mm": displacement,
                "dropped_contacts": dropped_contacts,
            }
        )
    return rows


def _collision_vertices(body: Any) -> np.ndarray:
    """Copy collision vertices into the rigid body's frame without rescaling."""
    vertices = []
    for shape in body.get_collision_shapes():
        if shape.vertices is None:
            raise ValueError(
                "Pour-water envelope checks require mesh collision shapes."
            )
        pose = shape.local_pose.cpu().numpy()
        vertices.append(shape.vertices.cpu().numpy() @ pose[:3, :3].T + pose[:3, 3])
    if not vertices:
        raise ValueError("No physical collision geometry is available.")
    return np.concatenate(vertices)


def _held_window(
    metadata: dict[str, Any], duration: float
) -> tuple[float | None, float | None]:
    """Locate grip acquisition and release across all compiled program segments."""
    start = None
    release = None
    for segment in metadata["segments"]:
        for call in segment["metadata"].get("runtime", {}).get("calls", []):
            for event in call.get("events", []):
                if event["kind"] != "trajectory_segment_entered":
                    continue
                if (
                    call["semantic_id"] == "pick"
                    and event.get("segment_name") == "lift"
                ):
                    start = float(event["timestamp"])
                if (
                    call["semantic_id"] == "place"
                    and event.get("segment_name") == "release"
                ):
                    release = float(event["timestamp"])
    return start, release if release is not None else duration


def _append_physical_report(path: Path, trials: list[dict[str, object]]) -> None:
    """Describe measured load checks without adding a fourth benchmark table."""
    if not trials:
        with path.open("a", encoding="utf-8") as report:
            report.write(
                "\nNo measured motion trials completed. Inspect task_results.json for executor failures.\n"
            )
        return
    lines = [
        "",
        "## Measured motion checks",
        "",
        "Bottle mass and principal inertia were scaled together. Telemetry was sampled at every physics substep; physical_checks.json records the physics and control intervals.",
        "Acceptance requires lift >= 80 mm, tilt >= 45 degrees, return error <= 50 mm, final tilt <= 10 degrees, idle TCP displacement <= 1 mm and actuator speed <= 1.05 times its effective native limit.",
        "Grasp acceptance also requires bottle-to-gripper translation drift <= 3 mm and rotation drift <= 3 degrees from lift until release. Cup contact substeps and dropped contact records must both be zero; cup displacement must be <= 2 mm. The capsule/sphere envelopes enclose the actual collision geometry and must remain >= 10 mm apart.",
        "",
    ]
    for index, mass in enumerate(trials[0]["bottle_masses_kg"]):
        rows = [trial["rows"][index] for trial in trials]
        errors = [
            row["return_error_m"] for row in rows if row["return_error_m"] is not None
        ]
        error_text = f"{max(errors) * 1000:.2f} mm" if errors else "unavailable"
        measured = [row for row in rows if row.get("max_grasp_drift_mm") is not None]
        grasp_text = (
            f"maximum grasp drift {max(row['max_grasp_drift_mm'] for row in measured):.2f} mm / {max(row['max_grasp_rotation_deg'] for row in measured):.2f} degrees; minimum conservative bottle/cup clearance {min(row['minimum_bottle_cup_clearance_mm'] for row in measured):.2f} mm; cup contact substeps {sum(row['cup_contact_substeps'] for row in measured)}"
            if len(measured) == len(rows)
            else "grasp/contact evidence unavailable"
        )
        lines.append(
            f"- {mass * 1000:g} g: {sum(row['accepted'] for row in rows)}/{len(rows)} accepted; minimum lift {min(row['max_lift_m'] for row in rows) * 1000:.2f} mm; minimum tilt {min(row['max_tilt_deg'] for row in rows):.2f} degrees; maximum return error {error_text}; idle TCP displacement {max(row['idle_tcp_displacement_mm'] for row in rows):.3f} mm; {grasp_text}."
        )
    lines += [
        "",
        "Program acceptance and the measured grasp/contact checks are reported separately. The pour deployment uses projected effects with a measured pre-pour position checkpoint; these checks do not measure water transfer. Native drive caps are recorded in resolved_drives.json. Generalized qf is not an actuator torque sensor, and simulation force budgets are not real-device calibration.",
    ]
    with path.open("a", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")


def _verify_task(
    output: Path,
    config: Path,
    masses: list[float],
    seeds: list[int],
    gripper_gains: list[tuple[float, float]] | None = None,
    *,
    renderer: str = "hybrid",
) -> None:
    """Qualify the production deployment with scaled bottle mass and inertia."""
    import gymnasium
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.sim.sensors.contact_sensor import (
        ArticulationContactFilterCfg,
        ContactSensor,
        ContactSensorCfg,
    )
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
        build_env_cfg_from_args,
    )
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.task_program.integrations._configured_composition import (
        _resolve_task_program_components,
    )
    from scripts.benchmark.task_program.demo_success import (
        DemoSuccessCase,
        _close_gym_demo_success_environment,
        run_demo_success_benchmark,
    )

    launcher = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(launcher, require_gym_config=False)
    args = launcher.parse_args(
        [
            "--gym_config",
            str(config),
            "--headless",
            "--device",
            "cuda",
            "--num_envs",
            str(len(masses)),
            "--filter_dataset_saving",
            "--disable-sensor",
            "--renderer",
            renderer,
        ]
    )
    discover_task_packages()
    execute_init_hooks()
    cfg, payload, _ = build_env_cfg_from_args(args)
    _, _, policy = _resolve_task_program_components(
        payload["task_program"], base_dir=config.parent
    )
    env = gymnasium.make(id=payload["id"], cfg=cfg).unwrapped
    robot = env.robot
    actuator_ids = robot.get_joint_ids(remove_mimic=True)
    # Native properties are already in the final joint order and include
    # configured overrides. Never validate tightened caps against stale numbers.
    limits = robot.get_joint_drive(joint_ids=actuator_ids)[3].clone()
    target_masses = torch.tensor(masses, device=robot.device)
    physical_trials = []
    bottle_body = env.sim.get_rigid_object("bottle")
    cup_body = env.sim.get_rigid_object("cup")
    bottle_vertices = _collision_vertices(bottle_body)
    cup_vertices = _collision_vertices(cup_body)
    bottle_z_bounds = (
        float(bottle_vertices[:, 2].min()),
        float(bottle_vertices[:, 2].max()),
    )
    bottle_radius = float(np.linalg.norm(bottle_vertices[:, :2], axis=1).max())
    cup_radius = float(np.linalg.norm(cup_vertices, axis=1).max())
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            uid="pour_verification_contacts",
            rigid_uid_list=["cup", "bottle"],
            articulation_cfg_list=[
                ArticulationContactFilterCfg(articulation_uid=robot.uid)
            ],
            filter_need_both_actor=True,
            max_contacts_per_env=4096,
        ),
        device=robot.device,
        owner=env.sim,
    )
    cup_actor_ids = contact_sensor.get_actor_ids("cup")[:, 0]
    right_arm_ids = robot.get_joint_ids("right_arm")
    # Record the effective native properties, including mimic lowering.
    (output / "resolved_drives.json").write_text(
        json.dumps(
            {
                "joint_names": robot.joint_names,
                "active_action_width": len(env.active_joint_ids),
                "properties": [
                    value.cpu().tolist() for value in robot.get_joint_drive()
                ],
            },
            indent=2,
        )
        + "\n"
    )

    def execute(environment: EmbodiedEnv, *, episode_index: int) -> DemoEpisodeResult:
        """Apply payloads and collect measured evidence during normal execution."""
        if gripper_gains is not None:
            for row, (stiffness, damping) in enumerate(gripper_gains):
                for part in ("left_eef", "right_eef"):
                    ids = robot.get_joint_ids(part, remove_mimic=True)
                    robot.set_joint_drive(
                        stiffness=torch.full(
                            (1, len(ids)), stiffness, device=robot.device
                        ),
                        damping=torch.full((1, len(ids)), damping, device=robot.device),
                        joint_ids=ids,
                        env_ids=[row],
                    )
            (output / "resolved_drives.json").write_text(
                json.dumps(
                    {
                        "joint_names": robot.joint_names,
                        "active_action_width": len(env.active_joint_ids),
                        "properties": [
                            value.cpu().tolist() for value in robot.get_joint_drive()
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
        bottle = environment.sim.get_rigid_object("bottle")
        base_mass = bottle.get_mass().clone()
        base_inertia = bottle.get_inertia().clone()
        bottle.set_mass(target_masses)
        bottle.set_inertia(base_inertia * (target_masses / base_mass).reshape(-1, 1))
        torch.testing.assert_close(bottle.get_mass(), target_masses)
        for _ in range(10):
            environment.step(ControllerAction(value=robot.get_qpos(target=True)))
        initial = bottle.get_local_pose(to_matrix=True).cpu().numpy().copy()
        initial_idle = _tcp_positions(robot, "left")
        samples = []
        eef_samples = []
        cup_samples = []
        contact_samples = []
        joint_samples = []
        target_samples = []
        velocity_samples = []
        dropped = [0]
        peak_ratio = np.zeros(len(masses))
        idle_displacement = np.zeros(len(masses))
        original_update = environment.sim.update

        def update(
            *args: Any,
            after_substep: Callable[[float], None] | None = None,
            **kwargs: Any,
        ) -> None:
            """Preserve the host observer and sample every physics substep."""

            def sample(dt: float) -> None:
                """Collect measured bottle, actuator and idle-arm motion."""
                if after_substep is not None:
                    after_substep(dt)
                samples.append(
                    bottle.get_local_pose(to_matrix=True).cpu().numpy().copy()
                )
                eef_samples.append(
                    robot.compute_fk(
                        robot.get_qpos()[:, right_arm_ids],
                        name="right_arm",
                        to_matrix=True,
                    )
                    .cpu()
                    .numpy()
                    .copy()
                )
                cup_samples.append(
                    cup_body.get_local_pose(to_matrix=True).cpu().numpy().copy()
                )
                contact_sensor.update()
                contact_data = contact_sensor.get_data()
                cup_contacts = contact_data["is_valid"] & (
                    contact_data["user_ids"] == cup_actor_ids[:, None, None]
                ).any(dim=-1)
                contact_samples.append(cup_contacts.sum(dim=1).cpu().numpy().copy())
                joint_samples.append(robot.get_qpos().cpu().numpy().copy())
                target_samples.append(robot.get_qpos(target=True).cpu().numpy().copy())
                velocity_samples.append(robot.get_qvel().cpu().numpy().copy())
                dropped[0] += contact_sensor.dropped_contacts
                ratio = (
                    (robot.get_qvel()[:, actuator_ids].abs() / limits)
                    .amax(dim=1)
                    .cpu()
                    .numpy()
                )
                np.maximum(peak_ratio, ratio, out=peak_ratio)
                displacement = (
                    np.linalg.norm(_tcp_positions(robot, "left") - initial_idle, axis=1)
                    * 1000
                )
                np.maximum(idle_displacement, displacement, out=idle_displacement)

            original_update(*args, after_substep=sample, **kwargs)

        environment.sim.update = update
        try:
            result = execute_demo_episode(environment, episode_index=episode_index)
        finally:
            environment.sim.update = original_update
        poses = np.asarray(samples)
        metrics = _physical_task_metrics(
            poses, initial, list(result.success), peak_ratio, idle_displacement
        )
        eef_poses = np.asarray(eef_samples)
        cup_poses = np.asarray(cup_samples)
        contact_counts = np.asarray(contact_samples).reshape(-1, len(masses))
        physics_dt = environment.sim.sim_config.physics_dt
        start, release = _held_window(result.to_metadata(), len(poses) * physics_dt)
        safety = _held_motion_metrics(
            poses,
            eef_poses,
            cup_poses,
            contact_counts,
            held_start_s=start,
            release_s=release,
            dt=physics_dt,
            bottle_z_bounds=bottle_z_bounds,
            bottle_radius=bottle_radius,
            cup_radius=cup_radius,
            dropped_contacts=dropped[0],
        )
        for row, checks in zip(metrics, safety):
            row.update(checks)
            row["accepted"] = bool(
                row["accepted"] and checks["grasp_and_clearance_pass"]
            )
        physical_trials.append(
            {
                "seed": seeds[episode_index],
                "bottle_masses_kg": masses,
                "rows": metrics,
                "steps": result.length,
                "effect_assurance": policy["effect_assurance"],
                "renderer": environment.sim.sim_config.render_cfg.renderer,
                "physics_dt_s": environment.sim.sim_config.physics_dt,
                "control_dt_s": environment.step_dt,
                "water_transfer_measured": False,
                "gripper_gain_overrides": gripper_gains,
            }
        )
        np.savez_compressed(
            output / f"task_seed_{seeds[episode_index]}.npz",
            poses=poses,
            initial=initial,
            eef=eef_poses,
            cup=cup_poses,
            cup_contacts=contact_counts,
            qpos=np.asarray(joint_samples),
            targets=np.asarray(target_samples),
            qvel=np.asarray(velocity_samples),
        )
        (output / "physical_checks.json").write_text(
            json.dumps(physical_trials, indent=2) + "\n"
        )
        print(json.dumps(physical_trials[-1]), flush=True)
        return result

    try:
        run_demo_success_benchmark(
            [DemoSuccessCase("configured_cobotmagic_limits", tuple(seeds))],
            lambda case: env,
            episode_executor=execute,
            raw_json_path=output / "task_results.json",
            report_path=output / "task_report.md",
        )
    finally:
        _close_gym_demo_success_environment(env)
    _append_physical_report(output / "task_report.md", physical_trials)
    if len(physical_trials) != len(seeds) or not all(
        row["accepted"] for trial in physical_trials for row in trial["rows"]
    ):
        raise RuntimeError(
            "Measured task qualification failed; inspect physical_checks.json."
        )
    print(f"Markdown report saved: {output / 'task_report.md'}", flush=True)


def run_all_benchmarks(output: Path, profiles: list[str]) -> None:
    """Run gravity holds, per-joint steps and smooth tracking comparisons.

    Args:
        output: Directory for raw traces, inertia audit and the Markdown report.
        profiles: Names of configured or pinned experimental drive profiles.
    """
    print("=== CobotMagic joint drive verification ===", flush=True)
    from embodichain.data import get_data_path
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg
    from embodichain.lab.sim.robots import CobotMagicCfg

    audit = _audit_urdf(
        Path(get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf"))
    )
    if not audit["valid"]:
        raise ValueError(
            "Source mass/inertia audit failed; inspect audit.json before gain tuning."
        )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    sim_cfg = SimulationManagerCfg(
        headless=True,
        device="cuda",
        num_envs=len(profiles),
        physics_cfg=DefaultPhysicsCfg(),
        render_cfg=RenderCfg(renderer="hybrid"),
        startup_summary="off",
    )
    sim = SimulationManager(sim_cfg)
    try:
        robot = sim.add_robot(
            CobotMagicCfg.from_dict({"init_pos": [0, 0, 0.7775], "init_qpos": _HOME})
        )
        sim.prepare()
        home = robot.get_qpos(target=True).clone()
        ids = robot.get_joint_ids("right_arm")
        dt = sim_cfg.physics_dt
        if not np.isclose(dt, 0.01):
            raise ValueError("This protocol requires 0.01 s physics steps.")
        rows = []
        for case in (
            ["hold_home", "hold_extended"]
            + [f"step_{index + 1}" for index in range(6)]
            + ["sine"]
        ):
            robot.reset()
            for index, profile in enumerate(profiles):
                _configure_profile(robot, profile, index)
            effective_drives = robot.get_joint_drive(joint_ids=ids)
            if case == "hold_home":
                (output / "resolved_drives.json").write_text(
                    json.dumps(
                        {
                            "profiles": profiles,
                            "joint_names": [robot.joint_names[index] for index in ids],
                            "properties": [
                                value.cpu().tolist() for value in effective_drives
                            ],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            reference = home.clone()
            if case == "hold_extended":
                for side in ("left_arm", "right_arm"):
                    reference[:, robot.get_joint_ids(side)] = torch.tensor(
                        [0, 0.35, -0.7, 0, 0.35, 0], device=robot.device
                    )
            robot.set_qpos(reference, target=False)
            robot.set_qpos(reference)
            sim.sync_render_state()
            ideal_tcp = _tcp_positions(robot, "right")
            sim.update(step=50, render_final_step=False)
            moved_joint = int(case[-1]) - 1 if case.startswith("step") else None
            if moved_joint is not None:
                reference[:, ids[moved_joint]] += 0.15
            duration = 8.0 if case == "sine" else 3.0
            commands = []
            measured = []
            velocities = []
            tcp_errors = []
            before_cpu = psutil.Process().memory_info().rss / 2**20
            before_gpu = torch.cuda.memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for control_step in range(round(duration / (4 * dt))):
                target = reference.clone()
                if case == "sine":
                    target[:, ids] += 0.15 * np.sin(
                        2 * np.pi * 0.25 * control_step * 4 * dt
                    )
                robot.set_qpos(target)

                def sample(_: float) -> None:
                    """Collect synchronized post-substep position and velocity."""
                    measured.append(
                        robot.get_qpos()[:, ids].detach().cpu().numpy().copy()
                    )
                    velocities.append(
                        robot.get_qvel()[:, ids].detach().cpu().numpy().copy()
                    )
                    commands.append(target[:, ids].detach().cpu().numpy().copy())
                    if case.startswith("hold"):
                        tcp_errors.append(
                            np.linalg.norm(
                                _tcp_positions(robot, "right") - ideal_tcp, axis=1
                            )
                            * 1000
                        )

                sim.update(step=4, render_final_step=False, after_substep=sample)
            elapsed = (time.perf_counter() - started) * 1000
            memory = {
                "cost_time_ms": elapsed,
                "cpu_delta_mb": psutil.Process().memory_info().rss / 2**20 - before_cpu,
                "gpu_delta_mb": torch.cuda.memory_allocated() / 2**20 - before_gpu,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
            }
            q = np.asarray(measured)
            v = np.asarray(velocities)
            targets = np.asarray(commands)
            np.savez_compressed(
                output / f"{case}.npz",
                measured=q,
                velocity=v,
                target=targets,
                profiles=profiles,
                dt=dt,
            )
            for index, profile in enumerate(profiles):
                kp = effective_drives[0][index].cpu().numpy()
                kd = effective_drives[1][index].cpu().numpy()
                effort = effective_drives[2][index].cpu().numpy()
                metrics = _tracking_metrics(
                    q[:, index],
                    v[:, index],
                    targets[:, index],
                    kp=kp,
                    kd=kd,
                    effort_limit=effort,
                    # The deliberately unbounded legacy profile is judged
                    # against source limits, so disabling caps cannot pass.
                    velocity_limits=(
                        _SOURCE_VELOCITY_LIMITS
                        if profile == "legacy"
                        else effective_drives[3][index].cpu().numpy()
                    ),
                    case=case,
                    dt=dt,
                    moved_joint=moved_joint,
                )
                tcp_error = (
                    float(np.asarray(tcp_errors)[-50:, index].max())
                    if tcp_errors
                    else 0.0
                )
                if case.startswith("hold"):
                    metrics["success"] = bool(metrics["success"] and tcp_error <= 1.0)
                row = {
                    "case": case,
                    "profile": profile,
                    **memory,
                    **metrics,
                    "tcp_error_mm": tcp_error,
                    "success_rate": float(metrics["success"]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        _write_report(output / "report.md", rows)
        print(f"Markdown report saved: {output / 'report.md'}", flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> int:
    """Run the requested profile comparison and write raw and Markdown results.

    Returns:
        Zero after completing the benchmark or reporting unavailable CUDA.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/benchmarks/cobotmagic_drives")
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(_PROFILES), default=list(_PROFILES)
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        help="Qualify the production Task Program instead of scanning joint gains.",
    )
    parser.add_argument(
        "--bottle-masses", type=float, nargs="+", default=[0.01, 0.1, 0.25]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gripper-stiffness", type=float, nargs="+")
    parser.add_argument("--gripper-damping", type=float, nargs="+")
    parser.add_argument(
        "--renderer",
        choices=("auto", "hybrid", "fast-rt", "rt", "no-render"),
        default="hybrid",
        help="Task-qualification renderer; pin Hybrid to match camera-enabled runs.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("Skipped: live CobotMagic drive verification requires CUDA and DexSim.")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task_config:
        if any(not np.isfinite(mass) or mass <= 0 for mass in args.bottle_masses):
            parser.error("Bottle masses must be finite and positive.")
        gains = None
        if args.gripper_stiffness is not None or args.gripper_damping is not None:
            if args.gripper_stiffness is None or args.gripper_damping is None:
                parser.error("Specify both gripper stiffness and damping.")
            stiffness = (
                args.gripper_stiffness * len(args.bottle_masses)
                if len(args.gripper_stiffness) == 1
                else args.gripper_stiffness
            )
            damping = (
                args.gripper_damping * len(args.bottle_masses)
                if len(args.gripper_damping) == 1
                else args.gripper_damping
            )
            if (
                len(stiffness) != len(args.bottle_masses)
                or len(damping) != len(args.bottle_masses)
                or any(
                    not np.isfinite(value) or value <= 0
                    for value in stiffness + damping
                )
            ):
                parser.error(
                    "Provide finite positive gains, one value or one per bottle mass."
                )
            gains = list(zip(stiffness, damping))
        _verify_task(
            args.output_dir,
            args.task_config,
            args.bottle_masses,
            args.seeds,
            gains,
            renderer=args.renderer,
        )
    else:
        run_all_benchmarks(args.output_dir, args.profiles)
    return 0


if __name__ == "__main__":
    # Native resources have been explicitly closed above. Avoid interpreter-order
    # teardown of vendor CUDA/Vulkan libraries in this standalone benchmark.
    import os
    import sys
    import traceback

    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
