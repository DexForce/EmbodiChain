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

"""Download and replay a RoboCasa365 episode with physical PandaOmron commands."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim._runtime_controls import _DexUniAllJointOwnership
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    RobotCfg,
    UsdImportCfg,
    UsdSourceCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.lab.sim.sensors.contact.sensor import (
    ArticulationContactFilterCfg,
    ContactSensor,
    ContactSensorCfg,
)

if __package__:
    from ._sources import _TASKS, _cached_episode
else:
    from _sources import _TASKS, _cached_episode

__all__ = ["main"]

_ROBOT_PRIM = "/KitchenEpisode/Geometry/robot0_base"
_BASE_LINK = "mobilebase0_base"
_GRIP_LINK = "gripper0_right_eef"
_COMMAND_DT = 0.004
_SOURCE_CONTROL_DT = 0.05
_STIFFNESS = np.array(
    [6500, 6500, 5000, 5000, 3500, 2200, 1800, 12000, 12000, 5000, 8000, 2200, 2200]
)
_DAMPING = np.array([150, 150, 130, 130, 100, 55, 45, 1000, 1000, 1500, 500, 22, 22])
_EFFORT = np.array([80, 80, 80, 80, 80, 12, 12, 600, 600, 600, 5000, 20, 20])
_DEVICE = "cuda:0"


def _tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def _free_pose(asset: Articulation | RigidObject, name: str) -> np.ndarray:
    if isinstance(asset, Articulation):
        return asset.get_link_pose(name, to_matrix=True)[0].cpu().numpy()
    return asset.get_local_pose(to_matrix=True)[0].cpu().numpy()


def _restore(
    assets: dict[str, Articulation | RigidObject], episode: dict[str, Any]
) -> None:
    scalar = episode["initial_scalar_joints"]
    free = episode["initial_free_bodies"]
    for name, asset in assets.items():
        if isinstance(asset, Articulation):
            joints = [
                (i, scalar[j]) for i, j in enumerate(asset.joint_names) if j in scalar
            ]
            if joints:
                ids, values = zip(*joints, strict=True)
                q, v = np.asarray(values).T
                asset.set_state(
                    qpos=_tensor(q),
                    qvel=_tensor(v),
                    joint_ids=list(ids),
                    clamp_qpos=False,
                )
            if name in free:
                asset.set_state(
                    root_pose=_tensor([free[name]["pose"]]),
                    root_velocity=_tensor([free[name]["velocity"]]),
                )
        elif name in free:
            asset.set_local_pose(_tensor([free[name]["pose"]]))
            velocity = _tensor([free[name]["velocity"]])
            asset.set_velocity(lin_vel=velocity[:, :3], ang_vel=velocity[:, 3:])


def _contact_sensor(
    sim: SimulationManager,
    assets: dict[str, Articulation | RigidObject],
    target_names: list[str],
) -> ContactSensor | None:
    if not target_names:
        return None
    targets = [assets[name] for name in target_names]
    return sim.add_sensor(
        ContactSensorCfg(
            uid="task_contacts",
            rigid_uid_list=[
                asset.uid for asset in targets if isinstance(asset, RigidObject)
            ],
            articulation_cfg_list=[
                ArticulationContactFilterCfg(
                    articulation_uid=asset.uid,
                    link_name_list=list(asset.body_data.link_names),
                )
                for asset in targets
                if isinstance(asset, Articulation)
            ],
            filter_need_both_actor=False,
            max_contacts_per_env=512,
        )
    )


def _contact_pairs(sensor: ContactSensor) -> set[tuple[str, str]]:
    sensor.update()
    if sensor.get_latest_contact_loss_mask().any().item():
        raise RuntimeError("Task contact evidence exceeded the sensor capacity.")
    data = sensor.get_data()
    contacts = data["user_ids"][0, data["is_valid"][0]].tolist()
    names = {}
    for actor_id in {
        actor_id for pair in contacts for actor_id in pair if actor_id >= 0
    }:
        actor = sensor.get_actor_info(actor_id)
        names[actor_id] = actor.link_name or actor.path.rsplit("/", 1)[-1]
    return {
        (names[left], names[right])
        for left, right in contacts
        if left in names and right in names
    }


def _touches(pairs: set[tuple[str, str]], first: str, second: str) -> bool:
    a = f"{first.removesuffix('_main')}_"
    b = f"{second.removesuffix('_main')}_"
    return any(
        (left.startswith(a) and right.startswith(b))
        or (right.startswith(a) and left.startswith(b))
        for left, right in pairs
    )


def _door_open_fractions(
    asset: Articulation, limits: dict[str, list[float]]
) -> list[float]:
    positions = asset.get_qpos()[0].cpu().numpy()
    result = []
    for joint, (low, high) in limits.items():
        fraction = float(
            (positions[asset.joint_names.index(joint)] - low) / (high - low)
        )
        result.append(fraction if low >= 0 else 1 - fraction)
    return result


def _in_rack(pose: np.ndarray, points: list[list[float]], rack: dict[str, Any]) -> bool:
    center = np.asarray(rack["center"])
    half = np.asarray(rack["half_size"])
    bottom = center[2] - half[2]
    if not bottom <= pose[2, 3] <= bottom + 1.8 * half[2]:
        return False
    corners = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    coordinates = (corners - center) @ np.asarray(rack["axes"])
    margin = 0.05 / (2 * half)
    return bool(np.all(np.abs(coordinates) <= half + margin))


def _terminal_evidence(
    contact_sensor: ContactSensor | None,
    assets: dict[str, Articulation | RigidObject],
    robot: Articulation,
    episode: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, list[float]]]:
    task = episode["task"]
    door_fractions = []
    if "door" in episode:
        door = episode["door"]
        fixture = assets[f"{door['fixture']}_main"]
        assert isinstance(fixture, Articulation)
        door_fractions = _door_open_fractions(fixture, door["limits"])
    if task == "OpenMicrowave":
        return {
            "microwave_open": all(fraction >= 0.90 for fraction in door_fractions)
        }, {}

    names = episode["target_objects"]
    if contact_sensor is None:
        raise RuntimeError("This task requires native contact evidence.")
    pairs = _contact_pairs(contact_sensor)
    poses = {name: _free_pose(assets[name], name) for name in names}
    positions = {name: pose[:3, 3] for name, pose in poses.items()}
    gripper = robot.get_link_pose(_GRIP_LINK)[0, :3].cpu().numpy()
    tongs = task == "TongBuffetSetup"
    retreat_threshold = 0.15 if tongs else 0.25
    predicates: dict[str, bool] = {}

    def in_receptacle(obj: str, receptacle: str) -> bool:
        offset = positions[obj][:2] - positions[receptacle][:2]
        threshold = episode["receptacle_thresholds"][receptacle]
        return bool(
            _touches(pairs, obj, receptacle) and np.linalg.norm(offset) < threshold
        )

    if task == "SetBowlsForSoup":
        first, second = ("bowl1_main", "bowl2_main")
        plate1, plate2 = ("plate1_main", "plate2_main")
        predicates["bowls_on_distinct_plates"] = (
            in_receptacle(first, plate1) and in_receptacle(second, plate2)
        ) or (in_receptacle(first, plate2) and in_receptacle(second, plate1))
        names = (first, second)
    elif task == "LoadPreparedFood":
        container = "tupperware_main"
        meat, vegetables = ("meat_in_tupperware_main", "veg_in_tupperware_main")
        racks = episode["fridge_racks"]
        box = episode["object_bbox_points"]
        predicates["container_on_target_shelf"] = _in_rack(
            poses[container], box[container], racks[episode["target_rack"]]
        )
        meat_in_box = in_receptacle(meat, container)
        vegetables_in_box = in_receptacle(vegetables, container)
        predicates["food_in_container"] = (meat_in_box and vegetables_in_box) or (
            (meat_in_box or vegetables_in_box) and _touches(pairs, meat, vegetables)
        )
        predicates["original_items_remain_in_fridge"] = all(
            any(
                0.50 <= rack["center"][2] - rack["half_size"][2] <= 1.50
                and _in_rack(poses[name], box[name], rack)
                for rack in racks.values()
            )
            for name in names
            if name.startswith(("existing_", "target_rack_"))
        )
        predicates["fridge_closed"] = all(
            fraction <= 0.005 for fraction in door_fractions
        )
        names = ()
    else:
        support = episode["support"]
        for name in names:
            predicates[f"{name}_on_{support}"] = _touches(pairs, name, support)
    for name in names:
        predicates[f"{name}_gripper_clear"] = bool(
            np.linalg.norm(positions[name] - gripper) > retreat_threshold
        )
    if tongs:
        offset = positions["tongs_main"][:2] - positions["food_container_main"][:2]
        predicates["tongs_near_tray"] = bool(np.linalg.norm(offset) <= 0.5)
    if "sink_bounds" in episode:
        sink_min, sink_max = np.asarray(episode["sink_bounds"])
        for name, pose in poses.items():
            local = np.asarray(episode["object_bbox_points"][name])
            corners = local @ pose[:3, :3].T + pose[:3, 3]
            separation = np.maximum(
                0,
                np.maximum(
                    sink_min - corners.max(axis=0), corners.min(axis=0) - sink_max
                ),
            )
            predicates[f"{name}_near_sink"] = bool(np.linalg.norm(separation) < 0.2)
    return predicates, {name: xyz.tolist() for name, xyz in positions.items()}


def _drive_targets(
    reference: Path, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(reference) as archive:
        time, position, velocity, bias, gripper = (
            archive[name]
            for name in ("time", "position", "velocity", "bias", "gripper")
        )
    command = 0.0
    finger_target = np.empty_like(gripper)
    for i, direction in enumerate(np.sign(gripper)):
        command = float(np.clip(command + 0.2 * direction, -1, 1))
        finger_target[i] = 0.02 - 0.02 * command
    # Sample the archived 20 Hz reference on the control grid before uploading.
    steps = int(round(time[-1] / _COMMAND_DT))
    elapsed = np.arange(1, steps + 1) * _COMMAND_DT
    frame = np.minimum((elapsed / _SOURCE_CONTROL_DT).astype(int), len(time) - 2)
    weight = np.clip((elapsed - time[frame]) / _SOURCE_CONTROL_DT, 0, 1)[:, None]
    targets = (1 - weight) * position[frame] + weight * position[frame + 1]
    velocities = (1 - weight) * velocity[frame] + weight * velocity[frame + 1]
    # RoboSuite's mirrored finger sliders are open at [0.04, -0.04].
    targets[:, -2:] = finger_target[frame, None] * np.array([1, -1])
    velocities[:, -2:] = 0
    bias_target = (1 - weight) * bias[frame] + weight * bias[frame + 1]
    return (
        torch.from_numpy(targets).to(device),
        torch.from_numpy(velocities).to(device),
        torch.from_numpy(bias_target).to(device),
    )


def _dexuni_solver_path(sim: SimulationManager) -> tuple[str, int]:
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    backend = get_newton_backend(sim.spawn_result.world)
    features = getattr(getattr(backend, "solver", None), "features", None)
    rigid_solver = getattr(backend, "mujoco_solver", None)
    if (
        features is None
        or features.backend != "pure_mujoco"
        or not features.mujoco_solve_enabled
        or features.mujoco_joint_count != backend.model.joint_count
        or rigid_solver is None
        or rigid_solver.use_mujoco_cpu
    ):
        raise RuntimeError(
            "RoboCasa365 requires DexUni with GPU MJ Warp ownership of all joints."
        )
    return features.backend, features.mujoco_joint_count


@contextmanager
def _replay_stage(source: Path) -> Iterator[Path]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(source))
    root = stage.GetDefaultPrim()
    key = "mujoco:solverOptions:disableflags"
    flags = int(root.GetCustomDataByKey(key) or 0)
    # Legacy CPU bundles disabled the mid-phase as a contact workaround. The
    # CUDA solver requires it. Author the compatibility override in a session
    # layer and export a temporary composed stage with the original assets.
    disable_midphase = 1 << 14
    if not (flags & disable_midphase):
        yield source
        return
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root.SetCustomDataByKey(key, flags & ~disable_midphase)
    with TemporaryDirectory(prefix="robocasa365-stage-") as directory:
        path = Path(directory) / source.name
        if not stage.Flatten().Export(str(path)):
            raise RuntimeError(
                f"Could not export the CUDA-compatible USD stage {path}."
            )
        yield path


def _replay(directory: Path) -> dict[str, Any]:
    with _replay_stage(directory / "KitchenEpisode.usdc") as source:
        return _run_scene(directory, source)


def _run_scene(directory: Path, source: Path) -> dict[str, Any]:
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    positions, velocities, bias = _drive_targets(directory / "reference.npz", _DEVICE)

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=_DEVICE,
            num_envs=1,
            enable_ground_plane=False,
            physics_dt=_COMMAND_DT,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device=_DEVICE,
                num_substeps=2,
                use_cuda_graph=True,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "joint_mode": "dynamic",
                    "mujoco_options": {
                        "use_mujoco_cpu": False,
                        "impratio": 20.0,
                        "cone": "elliptic",
                        "integrator": "euler",
                        "density": 1.2,
                        "viscosity": 2.0e-5,
                        "nconmax": 2048,
                        "njmax": 8192,
                    },
                },
            ),
        )
    )
    try:
        result = sim.add_usd(
            UsdImportCfg(
                name="robocasa365",
                source=UsdSourceCfg(file_path=str(source)),
                robot_cfgs={
                    _ROBOT_PRIM: RobotCfg(
                        uid="panda_omron", control_parts=episode["parts"]
                    )
                },
            )
        )
        sim.add_runtime_control(_DexUniAllJointOwnership())
        sim.prepare()
        solver_path, mujoco_joint_count = _dexuni_solver_path(sim)
        robot = result[_ROBOT_PRIM]
        assert isinstance(robot, Articulation)
        assets = {
            path.rsplit("/", 1)[-1]: asset
            for path, asset in result.items()
            if isinstance(asset, (Articulation, RigidObject))
        }
        sensor = _contact_sensor(sim, assets, episode["target_objects"])
        _restore(assets, episode)
        joint_names = [j for names in episode["parts"].values() for j in names]
        indices = torch.as_tensor(
            [robot.joint_names.index(j) for j in joint_names], device=_DEVICE
        )
        stiffness = torch.as_tensor(_STIFFNESS, dtype=torch.float64, device=_DEVICE)
        damping = torch.as_tensor(_DAMPING, dtype=torch.float64, device=_DEVICE)
        max_effort = torch.as_tensor(_EFFORT, dtype=torch.float64, device=_DEVICE)
        initial_base = robot.get_link_pose(_BASE_LINK)[0, :2].clone()
        base_excursion = torch.zeros((), device=_DEVICE)

        opening = episode.get("opening")
        fixture = assets[f"{opening['fixture']}_main"] if opening else None
        if isinstance(fixture, Articulation):
            opening_index = fixture.joint_names.index(opening["joint"])
            opening_sign = np.sign(opening["reference_excursion"])
        else:
            opening_index = None
        opening_excursion = torch.zeros((), device=_DEVICE)
        steps = len(positions)
        for step in range(steps):
            actual = robot.get_qpos()[0].index_select(0, indices)
            speed = robot.get_qvel()[0].index_select(0, indices)
            effort = torch.clamp(
                stiffness * (positions[step] - actual)
                + damping * (velocities[step] - speed)
                + bias[step],
                -max_effort,
                max_effort,
            )
            robot.set_state(qf=effort, joint_ids=indices)
            sim.update(step=1, render_final_step=False)

            if step % 10 == 0 or step == steps - 1:
                base = robot.get_link_pose(_BASE_LINK)[0, :2]
                base_excursion = torch.maximum(
                    base_excursion, torch.linalg.vector_norm(base - initial_base)
                )
                if opening_index is not None:
                    delta = fixture.get_qpos()[0, opening_index] - opening["start"]
                    opening_excursion = torch.maximum(
                        opening_excursion, opening_sign * delta
                    )

        base_excursion = float(base_excursion.item())
        opening_excursion = float(opening_excursion.item())
        if not np.isfinite([base_excursion, opening_excursion]).all():
            raise RuntimeError("Non-finite pose measured during the DexUni replay.")
        predicates, final_positions = _terminal_evidence(sensor, assets, robot, episode)
        source_success = all(predicates.values())
        moved = base_excursion >= 0.25
        opened = opening_index is None or opening_excursion >= 0.5 * abs(
            opening["reference_excursion"]
        )
        return {
            "task": episode["task"],
            "solver": "dexuni",
            "solver_path": solver_path,
            "rigid_solver": "mjwarp_cuda",
            "mujoco_joint_count": mujoco_joint_count,
            "episode": episode["episode"],
            "layout": episode["layout"],
            "style": episode["style"],
            "duration_s": sim.simulation_time,
            "base_excursion_m": base_excursion,
            "opening_excursion": (
                opening_excursion if opening_index is not None else None
            ),
            "source_predicates": predicates,
            "source_success": source_success,
            "base_moved": moved,
            "fixture_motion_qualified": (
                bool(opened) if opening_index is not None else None
            ),
            "success": bool(source_success and moved and opened),
            "final_positions_m": final_positions,
            "final_reference_error_m": {
                name: float(
                    np.linalg.norm(
                        position
                        - np.asarray(episode["reference_final_positions"][name])
                    )
                )
                for name, position in final_positions.items()
            },
        }
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run the original scene from its archived initial state and report physics outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--task", choices=_TASKS, help="Public task (default: OpenMicrowave)."
    )
    source.add_argument(
        "--episode-dir", type=Path, help="Use an existing prepared episode."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "robocasa365",
        help="Cache for downloaded sources and prepared episodes.",
    )
    parser.add_argument("--report", type=Path, help="Write a JSON validation report.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("RoboCasa365 replay requires an NVIDIA CUDA device with MJ Warp.")
    directory = (
        args.episode_dir.expanduser().resolve()
        if args.episode_dir
        else _cached_episode(
            args.task or "OpenMicrowave", args.cache_dir.expanduser().resolve()
        )
    )
    report = _replay(directory)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        args.report.write_text(encoded + "\n", encoding="utf-8")
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
