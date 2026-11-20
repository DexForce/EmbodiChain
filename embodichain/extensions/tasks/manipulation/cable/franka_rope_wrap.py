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
"""Scene, asset and reset handling for the Franka rope-wrapping task."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import shutil
import numpy as np
from scipy.interpolate import make_interp_spline

if TYPE_CHECKING:
    import torch
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "cable_points",
    "load_config",
    "materialize_deployment",
    "settle_franka_rope",
    "task_directory",
]


def task_directory() -> Path:
    """Locate the co-installed physical scene and task components."""
    from embodichain.extensions.tasks.configs import get_config_path

    return get_config_path("tasks/manipulation/cable/franka_rope_wrap")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load scene settings and derive evaluation windows from the motion clock."""
    config = json.loads(
        (
            Path(path) if path is not None else task_directory() / "config.json"
        ).read_text()
    )
    config.setdefault("open_half_gap_m", 0.04)
    config.setdefault("grip_half_gap_m", config["cable_radius_m"] - 0.0005)
    config.setdefault("floor_friction", 0.2)
    scale = config["duration_s"] / 36.0
    config["grasp_complete_s"] = 5.0 * scale
    config["lift_complete_s"] = 7.0 * scale
    _validate_config(config)
    return config


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _validate_config(config: dict[str, Any]) -> None:
    """Reject inputs that break the standalone control and geometry contracts."""

    def finite_values(value: Any, name: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                finite_values(item, f"{name}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite_values(item, f"{name}[{index}]")
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    finite_values(config, "config")
    for name, minimum in (
        ("control_hz", 1),
        ("substeps", 4),
        ("iterations", 1),
        ("cable_segments", 8),
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["substeps"] % 4:
        raise ValueError("substeps must be divisible by four for contact sampling")
    for name in (
        "cable_radius_m",
        "cable_density_kg_m3",
        "post_radius_m",
        "post_height_m",
        "finger_pad_width_m",
        "duration_s",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
    for name in (
        "pre_roll_s",
        "linear_stiffness_N_m",
        "bend_stiffness_Nm_rad",
        "twist_stiffness_Nm_rad",
        "bend_damping_Nms_rad",
        "contact_ke_N_m",
        "contact_kd_Ns_m",
        "contact_gap_m",
        "contact_match_distance_m",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative number")
    if config["duration_s"] < 36:
        raise ValueError("duration_s must include the full 36-second motion and hold")
    if not np.isclose(
        config["duration_s"] * config["control_hz"],
        round(config["duration_s"] * config["control_hz"]),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("duration_s must align to the control grid")
    if (
        not 0
        <= config["grasp_complete_s"]
        < config["lift_complete_s"]
        <= config["duration_s"] - 2
    ):
        raise ValueError("Grasp and lift checks must precede the final two-second hold")
    for name, shape in (
        ("home_qpos", (7,)),
        ("post_center_xy_m", (2,)),
        ("finger_pad_center_m", (3,)),
        ("finger_pad_size_m", (3,)),
    ):
        vector = np.asarray(config[name], dtype=float)
        if vector.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain finite values")
    if np.any(np.asarray(config["finger_pad_size_m"], dtype=float) <= 0):
        raise ValueError("finger_pad_size_m must contain positive dimensions")
    points = np.asarray(config["cable_waypoints_m"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("cable_waypoints_m must contain at least three XYZ waypoints")
    if not np.isfinite(points).all():
        raise ValueError("cable_waypoints_m must contain finite values")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
        raise ValueError("Consecutive cable waypoints must be distinct")
    if (
        np.linalg.norm(
            points[0, :2] - np.asarray(config["post_center_xy_m"], dtype=float)
        )
        <= 0
    ):
        raise ValueError(
            "The cable anchor and post center must have distinct XY coordinates"
        )
    total_length = float(
        np.linalg.norm(np.diff(cable_points(config), axis=0), axis=1).sum()
    )
    inset = config["grasp_inset_m"]
    if (
        isinstance(inset, bool)
        or not isinstance(inset, (int, float))
        or not 0 < inset < total_length
    ):
        raise ValueError(
            "grasp_inset_m must lie strictly inside the sampled cable length"
        )
    thresholds = config["validation"]
    if not isinstance(thresholds, dict):
        raise ValueError("validation must be a mapping of acceptance thresholds")
    for name, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"validation.{name} must be a finite nonnegative number")
    if not isinstance(config["robot_urdf"], str) or not config["robot_urdf"].strip():
        raise ValueError("robot_urdf must be an asset reference or absolute URDF path")


def _uses_packaged_geometry(config: dict[str, Any]) -> bool:
    defaults = json.loads((task_directory() / "config.json").read_text())
    return all(
        config[key] == defaults[key]
        for key in (
            "cable_waypoints_m",
            "post_center_xy_m",
            "floor_height_m",
            "cable_radius_m",
            "cable_segments",
        )
    )


def cable_points(config: dict[str, Any]) -> np.ndarray:
    """Read the packaged centerline or sample the user's custom geometry."""
    if _uses_packaged_geometry(config):
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(str(task_directory() / "cable.usda"))
        curve = layer.GetPrimAtPath("/World/cable")
        return np.array(curve.attributes["points"].default, dtype=np.float32)
    return _sample_custom_centerline(config)


def _sample_custom_centerline(config: dict[str, Any]) -> np.ndarray:
    points = np.asarray(config["cable_waypoints_m"], float)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    curve = make_interp_spline(distances, points, k=2)(
        np.linspace(0, distances[-1], 4000)
    )
    curve[:, 2] = np.maximum(
        curve[:, 2], config["floor_height_m"] + config["cable_radius_m"]
    )
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    result = np.column_stack(
        [
            np.interp(
                np.linspace(0, arc[-1], config["cable_segments"] + 1), arc, curve[:, i]
            )
            for i in range(3)
        ]
    )
    # The fixed root capsule exits its clamp toward the post. Preserve the
    # sampled first-segment length while removing spline endpoint tangent error.
    direction = np.r_[np.asarray(config["post_center_xy_m"]) - result[0, :2], 0.0]
    direction /= np.linalg.norm(direction)
    result[1] = result[0] + direction * np.linalg.norm(result[1] - result[0])
    return result.astype(np.float32)


def _write_cable_asset(config: dict[str, Any], path: Path) -> None:
    """Copy the packaged rope, updating its geometry for a custom scenario."""
    source = task_directory() / "cable.usda"
    if _uses_packaged_geometry(config):
        shutil.copyfile(source, path)
        return

    from pxr import Sdf, Vt

    layer = Sdf.Layer.OpenAsAnonymous(str(source))
    curve = layer.GetPrimAtPath("/World/cable")
    points = _sample_custom_centerline(config)
    curve.attributes["points"].default = Vt.Vec3fArray.FromNumpy(points)
    curve.attributes["curveVertexCounts"].default = Vt.IntArray([len(points)])
    curve.attributes["physics:thicknesses"].default = Vt.FloatArray(
        [2.0 * config["cable_radius_m"]]
    )
    layer.Export(str(path))


def settle_franka_rope(
    env: EmbodiedEnv, env_ids: torch.Tensor | None, scenario_path: str | None = None
) -> None:
    """Settle the free cable against the slab before observing task time zero."""
    import torch

    if env_ids is not None and len(env_ids) != env.num_envs:
        raise ValueError(
            "Franka rope pre-roll advances the world and requires a full reset"
        )
    from .wrap_motion import _JOINT_NAMES

    if tuple(env.robot.joint_names) != _JOINT_NAMES:
        raise ValueError(f"Unexpected Franka joint order {env.robot.joint_names!r}.")
    config = load_config(Path(scenario_path) if scenario_path is not None else None)
    env.franka_rope_config = config
    arm_home = [*config["home_qpos"], *([config["open_half_gap_m"]] * 2)]
    qpos = torch.tensor(arm_home, dtype=torch.float32, device=env.device).expand(
        env.num_envs, -1
    )
    env.robot.set_qpos(qpos, target=True)
    steps = round(config["pre_roll_s"] * config["control_hz"] * config["substeps"])
    for _ in range(steps):
        env.sim.update(step=1, render_final_step=False)


def _contact(config: dict[str, Any], friction: float) -> dict[str, Any]:
    return {
        "collision_props": {
            "backend": "newton",
            "margin": 0.0,
            "gap": config["contact_gap_m"],
        },
        "material_props": {
            "backend": "newton",
            "dynamic_friction": friction,
            "ke": config["contact_ke_N_m"],
            "kd": config["contact_kd_Ns_m"],
        },
    }


def _environment(
    config: dict[str, Any], scenario_path: Path | None = None
) -> dict[str, Any]:
    floor = config["floor_height_m"]
    center = config["post_center_xy_m"]
    anchor = np.asarray(config["cable_waypoints_m"][0], dtype=float)
    direction = np.r_[np.array(center) - anchor[:2], 0.0]
    direction /= np.linalg.norm(direction)

    def rigid(
        uid: str,
        shape: dict,
        position: list,
        friction: float,
        color: list[float],
        roughness: float,
    ) -> dict:
        return {
            "uid": uid,
            "body_type": "static",
            "init_pos": position,
            "shape": {
                **shape,
                "visual_material": {
                    "uid": uid + "_mat",
                    "base_color": color,
                    "roughness": roughness,
                },
            },
            "attrs": _contact(config, friction),
        }

    slab = rigid(
        "table",
        {"shape_type": "Cube", "size": [1.2, 1.05, 0.05]},
        [0.35, 0, floor - 0.025],
        config["floor_friction"],
        [0.33, 0.37, 0.41, 1],
        0.65,
    )
    clamp = rigid(
        "anchor_clamp",
        {"shape_type": "Cube", "size": [0.028, 0.018, 0.008]},
        (anchor - 0.006 * direction - [0, 0, 0.004]).tolist(),
        config["floor_friction"],
        [0.08, 0.11, 0.15, 1],
        0.65,
    )
    clamp["init_rot"] = [
        0,
        0,
        float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
    ]
    post = rigid(
        "post",
        {
            "shape_type": "Cylinder",
            "radius": config["post_radius_m"],
            "half_height": config["post_height_m"] / 2,
        },
        [*center, floor + config["post_height_m"] / 2],
        config["post_friction"],
        [0.025, 0.45, 0.39, 1],
        0.38,
    )
    cable = {
        "uid": "cable",
        "fixed_segments": [0],
        "collision_filter_span": 2,
        "attrs": {
            "density": config["cable_density_kg_m3"],
            "stretch_stiffness": config["linear_stiffness_N_m"],
            "shear_stiffness": config["linear_stiffness_N_m"],
            "bend_stiffness": config["bend_stiffness_Nm_rad"],
            "twist_stiffness": config["twist_stiffness_Nm_rad"],
            "stretch_damping": 0.03,
            "shear_damping": 0.03,
            "bend_damping": config["bend_damping_Nms_rad"],
            "twist_damping": config["bend_damping_Nms_rad"],
            **_contact(config, config["rope_friction"]),
        },
    }
    return {
        "environment_id": "franka_rope_wrap",
        "enable_ground_plane": True,
        "physics": "newton",
        "physics_config": {
            "physics_dt": 1.0 / (config["control_hz"] * config["substeps"]),
            "num_substeps": 1,
            "collision_cfg": None,
            "sync_to_renderer": False,
            "use_cuda_graph": True,
            "solver_cfg": {
                "solver_type": "dexuni",
                "joint_mode": "dynamic",
                "coupling": "two_way",
                "mujoco_articulation_paths": ["arena_0/franka"],
                "vbd_options": {
                    "iterations": config["iterations"],
                    "rigid_contact_history": config.get("rigid_contact_history", True),
                    "rigid_contact_hard": config.get("rigid_contact_hard", False),
                    "rigid_joint_linear_ke": config.get(
                        "rigid_joint_linear_ke", 100000.0
                    ),
                    "rigid_joint_angular_ke": config.get(
                        "rigid_joint_angular_ke", 10.0
                    ),
                    "friction_epsilon": config.get("friction_epsilon", 1e-5),
                },
                "collision_options": {
                    "contact_matching": "sticky",
                    "contact_matching_pos_threshold": config[
                        "contact_match_distance_m"
                    ],
                    "rigid_contact_max": 4096,
                },
            },
        },
        "max_episode_steps": round(config["duration_s"] * config["control_hz"]) + 1,
        "num_envs": 1,
        "simulation": {
            "background": [slab, clamp, post],
            "usd_scene": {
                "name": "cable_scene",
                "source": {"file_path": "./cable.usda"},
                "asset_cfgs": {"/World/cable": cable},
            },
        },
        "env": {
            "sim_steps_per_control": config["substeps"],
            "ignore_terminations": True,
            "expert_trajectory": {"joint_command_mode": "position_velocity"},
            "task_instruction": "Pick up the rope, wrap it around the post, tighten and hold it.",
            "events": {
                "settle_rope": {
                    "func": "settle_franka_rope",
                    "mode": "reset",
                    "params": (
                        {"scenario_path": str(scenario_path)} if scenario_path else {}
                    ),
                }
            },
        },
    }


def materialize_deployment(config: dict[str, Any], robot_path: Path, out: Path) -> Path:
    """Write an equivalent configured Gym deployment for a user-supplied scene."""
    import yaml

    _validate_config(config)
    folder = out.resolve() / "deployment"
    folder.mkdir()
    _write_cable_asset(config, folder / "cable.usda")
    reference = task_directory()
    deployment = yaml.safe_load((reference / "task.franka.yaml").read_text())
    embodiment_path = reference / deployment["embodiment"]["component"]
    embodiment = yaml.safe_load(embodiment_path.read_text())
    robot = embodiment["simulation"]
    robot["fpath"] = str(robot_path.resolve())
    robot["init_pos"] = [0, 0, config["floor_height_m"]]
    robot["init_qpos"] = [*config["home_qpos"], *([config["open_half_gap_m"]] * 2)]
    robot["attrs"] = _contact(config, 1.2)
    commands = embodiment["skill_profile"]["command_presets"][0]["commands"]
    commands.update(
        open=[config["open_half_gap_m"]] * 2, grasp=[config["grip_half_gap_m"]] * 2
    )
    scenario_path = folder / "config.json"
    _json(scenario_path, config)
    environment = _environment(config, scenario_path)
    for section in ("task_program", "evaluation"):
        for key, path in deployment[section].items():
            deployment[section][key] = str((reference / path).resolve())
    # The orbit is one atomic action. Its timeout uses simulation seconds and
    # must cover all its samples, including the boundary acknowledgement.
    from embodichain.lab.sim.atomic_skills.policies import RecoveryPolicy
    from .wrap_motion import PHASE_END_S

    ends = np.asarray([0.0, *PHASE_END_S.values()])
    longest_phase_s = float(np.diff(ends).max()) * config["duration_s"] / ends[-1]
    required_timeout = float(longest_phase_s + 2 / config["control_hz"])
    policy_path = deployment["task_program"]["execution_policy"]
    recovery = yaml.safe_load(Path(policy_path).read_text())["recovery"]
    if required_timeout > recovery.get("skill_timeout", RecoveryPolicy().skill_timeout):
        deployment["task_program"]["execution_policy"] = {
            "component": policy_path,
            "overrides": {"recovery": {"skill_timeout": required_timeout}},
        }
    deployment["embodiment"]["component"] = "embodiment.yaml"
    for name, values in [
        ("env.yaml", environment),
        ("embodiment.yaml", embodiment),
        ("task.franka.yaml", deployment),
    ]:
        (folder / name).write_text(yaml.safe_dump(values, sort_keys=False))
    return folder / "task.franka.yaml"
