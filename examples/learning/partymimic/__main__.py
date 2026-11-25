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

"""Run the RP1 carrying demos with live EmbodiChain physics."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from hashlib import file_digest
import json
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from dexsim.scene import ContactQuery
    from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy
    from embodichain.lab.sim import SimulationManager

__all__ = ["main"]

_CACHE = Path.home() / ".cache/embodichain/partymimic"
_ROBOT_XML = _CACHE / "upstream/data/rp1_28dof_mjlab/rp1_28dof.xml"
_CONVERTER = _CACHE / "converter/bin/python"
_SCENARIOS = ("walk", "step-up")
_VIDEO_RESOLUTION = (1280, 720)
_VIDEO_FPS = 50
_CAMERA_OFFSET = (1.18, -2.18, 0.42)
_STEP_CAMERA_OFFSET = (1.36, -2.51, 0.48)
_RP_STEP_CAMERA_OFFSET = (1.65, -1.73, 0.49)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _run(
    args: argparse.Namespace, scenario: str, policy: PartyMimicPolicy
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np
    from scipy.spatial.transform import Rotation
    import torch

    from dexsim import spawn
    from embodichain.extensions.models.policies.partymimic import (
        PartyMimicState,
        sparse5_window,
    )
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        LightCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        UsdImportCfg,
        UsdSourceCfg,
    )
    from embodichain.lab.sim.objects import Articulation

    from ._reference import (
        CONTROL_DT,
        FOOT_SOLE_OFFSET,
        PHYSICS_DT,
        SagittalMirror,
        make_reference,
        mirror_sparse_poses,
    )
    from ._scene import WristCrateClearance, convert_scene, write_mjcf

    output = args.output_dir
    asset_dir = output / "scenes"
    asset_dir.mkdir(exist_ok=True)
    stem = scenario.replace("-", "_")
    video_step = scenario == "step-up" and args.step_reference == "video"
    rp_hand_step = video_step and args.hand == "rp"
    reference = make_reference(
        scenario,
        args.robot_xml,
        policy,
        video_step=video_step,
        rp_hands=args.hand == "rp",
    )
    payload_kg = 0.0 if args.unloaded else reference.payload_kg
    mjcf = asset_dir / f"{stem}.xml"
    usd = mjcf.with_suffix(".usdc")
    model = write_mjcf(
        args.robot_xml,
        mjcf,
        scenario,
        reference,
        policy,
        payload_kg=payload_kg,
        hand_model=args.hand,
    )
    convert_scene(mjcf, model, usd, args.converter_python, policy, reference)
    native_wrist_to_crate = model.eq_data[:, 3:6].copy()
    reflection = SagittalMirror(model, policy.joint_names) if video_step else None
    policy_reference = (
        mirror_sparse_poses(reference.poses)
        if reflection is not None
        else reference.poses
    )

    want_video = not args.no_video
    cpu_physics = args.device == "cpu"
    use_dexuni = args.solver == "dexuni"
    mujoco_options = {
        "use_mujoco_cpu": cpu_physics,
        "iterations": 100,
        "ls_iterations": 50,
        "nconmax": 128,
        "njmax": 500,
    }
    if use_dexuni:
        # This single-world MJCF contains every physical joint in the scene.
        # Keep the floating crate and RP1 in the same DexUni partition so its
        # compact MuJoCo model retains both cross-tree welds.
        solver_cfg = {
            "solver_type": "dexuni",
            "mujoco_joints": tuple(range(model.njnt)),
            "mujoco_options": mujoco_options,
            "request_body_acceleration": False,
        }
    else:
        solver_cfg = {
            "solver_type": "mujoco_warp",
            "rne_mode": "disabled",
            **mujoco_options,
        }
    resolution = _VIDEO_RESOLUTION
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            headless=True,
            device=args.device,
            gpu_id=torch.device(args.device).index or 0,
            width=resolution[0],
            height=resolution[1],
            enable_ground_plane=False,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(
                renderer="fast-rt" if want_video else "no-render", spp=8
            ),
            physics_cfg=NewtonPhysicsCfg(
                device=args.device,
                num_substeps=1,
                physics_dt=PHYSICS_DT,
                sync_to_renderer=False,
                solver_cfg=solver_cfg,
            ),
        )
    )
    camera = None
    try:
        assets = sim.add_usd(
            UsdImportCfg(name="rp1_carry", source=UsdSourceCfg(file_path=str(usd)))
        )
        robot = next(
            asset for asset in assets.values() if isinstance(asset, Articulation)
        )
        if want_video:
            for name, direction, intensity in (
                ("key", (-0.52, 0.46, -0.71), 3.0),
                ("fill", (0.38, -0.46, -0.80), 0.9),
            ):
                sim.add_light(
                    LightCfg(
                        uid=name,
                        light_type="direction",
                        direction=direction,
                        intensity=intensity,
                    )
                )
        sim.prepare()
        scene = sim.spawn_result
        solver_features = (scene.get_solver_diagnostics() or {}).get("features", {})
        native_path = {path.rsplit("/", 1)[-1]: path for path in scene.get_paths()}
        crate = next(
            (asset for asset in assets.values() if asset.uid == "carry_crate"), None
        )
        if payload_kg > 0 and (crate is None or len(native_wrist_to_crate) != 2):
            raise RuntimeError(
                "The payload scene requires a rigid crate and two welds."
            )
        if use_dexuni and crate is not None:
            _verify_payload_welds(sim, policy.keypoint_names[2:4], crate.uid)
        measured_mass = 0.0 if crate is None else float(crate.get_mass()[0].item())
        if abs(measured_mass - payload_kg) > 1e-4:
            raise RuntimeError(
                f"Expected {payload_kg} kg, got {measured_mass} kg in Newton."
            )
        target_joints = policy.joint_indices(robot.joint_names).tolist()
        wrist_indices = [
            robot.link_names.index(name) for name in policy.keypoint_names[2:4]
        ]
        clearance = (
            WristCrateClearance(model, robot.link_names) if crate is not None else None
        )
        platform_contacts: Callable[[], tuple[bool, ...]] | None = None
        swing_contacts = None
        platform = reference.platform
        if platform is not None:
            platform_path, height = native_path["platform"], platform.size[2]
            if cpu_physics:
                platform_contacts = _cpu_platform_contacts(
                    sim, policy.keypoint_names[4:], platform_path, height
                )
            else:
                contacts = [
                    scene.create_contact_query(
                        [(native_path[robot.uid], name), platform_path],
                        match="all",
                        capacity=32,
                        device="cpu",
                    )
                    for name in policy.keypoint_names[4:]
                ]

                def warp_platform_contacts() -> tuple[bool, ...]:
                    return tuple(_supported(query, height) for query in contacts)

                platform_contacts = warp_platform_contacts
                if rp_hand_step and crate is not None:
                    robot_path = native_path[robot.uid]
                    right_ankle = policy.keypoint_names[5]
                    right_ankle_index = robot.link_names.index(right_ankle)
                    swing_contacts = _StepSwingContacts(
                        scene.create_contact_query(
                            [robot_path, platform_path],
                            match="all",
                            capacity=64,
                            device="cpu",
                        ),
                        scene.create_contact_query(
                            [(robot_path, right_ankle), native_path["floor"]],
                            match="all",
                            capacity=32,
                            device="cpu",
                        ),
                        scene.create_contact_query(
                            [native_path[crate.uid], platform_path],
                            match="all",
                            capacity=32,
                            device="cpu",
                        ),
                    )

        q0 = torch.as_tensor(reference.initial_joints[None], device=sim.device)
        robot.set_state(
            root_pose=torch.as_tensor(reference.poses[0, 0:1], device=sim.device),
            root_velocity=torch.zeros((1, 6), device=sim.device),
            qpos=q0,
            qvel=torch.zeros_like(q0),
            target_qpos=q0,
            joint_ids=target_joints,
        )

        def read() -> tuple[PartyMimicState, np.ndarray]:
            snapshot = robot.body_data.fetch_state()
            root = snapshot["root_pose"][0].cpu().numpy()
            return (
                PartyMimicState(
                    snapshot["qpos"][0].cpu().numpy()[target_joints],
                    snapshot["qvel"][0].cpu().numpy()[target_joints],
                    root[3:],
                    snapshot["root_ang_vel"][0].cpu().numpy(),
                ),
                root,
            )

        state, initial_root = read()
        start_xyz = initial_root[:3].copy()
        policy.reset(
            reflection.state(state) if reflection is not None else state,
            sparse5_window(policy_reference, 0),
        )
        root = initial_root
        loop_ms: list[float] = []
        max_weld_error = 0.0
        initial_wrist_gap = None
        initial_hand_gap = None
        if clearance is not None:
            initial_wrist_gap, initial_hand_gap = clearance.measure(
                robot.body_data.body_link_pose[0, clearance.link_indices].cpu().numpy(),
                crate.get_local_pose()[0].cpu().numpy(),
            )
        min_wrist_gap = initial_wrist_gap
        min_hand_gap = initial_hand_gap
        stable_frames = 0
        max_stable_frames = 0
        fall = False
        min_root_height = float(root[2])
        contact_streak = [0, 0]
        first_contact_s: list[float | None] = [None, None]
        torso_yaw_deg = 0.0
        peak_torso_yaw_deg = 0.0
        if platform_contacts is not None:
            torso_index = robot.link_names.index(policy.keypoint_names[1])
            initial_torso_rotation = Rotation.from_quat(
                robot.body_data.body_link_pose[0, torso_index, 3:].cpu().numpy()
            )
        start_time = sim.simulation_time
        physics_steps_per_command = round(CONTROL_DT / PHYSICS_DT)

        with ExitStack() as stack:
            video = None
            if want_video:
                # The source platform's projected edges give a roughly 865 px focal
                # length at this resolution. Allow some margin around the box.
                step_intrinsics = (
                    np.array(
                        [[830.0, 0.0, 640.0], [0.0, 830.0, 360.0], [0.0, 0.0, 1.0]]
                    )
                    if rp_hand_step
                    else None
                )
                camera = scene.add_camera(
                    spawn.CameraDesc(
                        name=f"{stem}_camera",
                        resolution=resolution,
                        fov=None if rp_hand_step else 70.0,
                        intrinsic=step_intrinsics,
                    )
                )
                camera.open_camera()
                video = stack.enter_context(
                    imageio.get_writer(
                        output / f"{stem}.mp4",
                        fps=_VIDEO_FPS,
                        codec="libx264",
                        pixelformat="yuv420p",
                        quality=8,
                        ffmpeg_log_level="error",
                    )
                )
            for frame in range(reference.control_steps):
                start = perf_counter()
                target = policy.step(
                    reflection.state(state) if reflection is not None else state,
                    sparse5_window(policy_reference, frame),
                )
                if reflection is not None:
                    target = reflection.joints(target)
                robot.set_qpos(
                    torch.as_tensor(target[None], device=sim.device),
                    joint_ids=target_joints,
                    clamp_target_qpos=False,
                )
                sim.update(
                    step=physics_steps_per_command,
                    physics_dt=PHYSICS_DT,
                    render_final_step=False,
                )
                state, root = read()
                loop_ms.append(1000 * (perf_counter() - start))

                min_root_height = min(min_root_height, float(root[2]))
                up_z = Rotation.from_quat(root[3:]).as_matrix()[2, 2]
                fall |= root[2] < 0.45 or up_z < 0.50
                if clearance is not None:
                    links = robot.body_data.body_link_pose[0].cpu().numpy()
                    wrists = links[wrist_indices]
                    crate_pose = crate.get_local_pose()[0].cpu().numpy()
                    predicted = wrists[:, :3] + Rotation.from_quat(wrists[:, 3:]).apply(
                        native_wrist_to_crate
                    )
                    error = np.linalg.norm(predicted - crate_pose[:3], axis=-1).max()
                    max_weld_error = max(max_weld_error, float(error))
                    wrist_gap, hand_gap = clearance.measure(
                        links[clearance.link_indices], crate_pose
                    )
                    min_wrist_gap = min(min_wrist_gap, wrist_gap)
                    if min_hand_gap is not None and hand_gap is not None:
                        min_hand_gap = min(min_hand_gap, hand_gap)
                if platform_contacts is not None:
                    on_platform = platform_contacts()
                    stable_frames = stable_frames + 1 if all(on_platform) else 0
                    max_stable_frames = max(max_stable_frames, stable_frames)
                    for foot, supported in enumerate(on_platform):
                        contact_streak[foot] = (
                            contact_streak[foot] + 1 if supported else 0
                        )
                        if contact_streak[foot] == 3 and first_contact_s[foot] is None:
                            first_contact_s[foot] = round((frame - 1) * CONTROL_DT, 9)
                    if swing_contacts is not None:
                        ankle_z = float(links[right_ankle_index, 2])
                        swing_contacts.sample(
                            (frame + 1) * CONTROL_DT,
                            height,
                            ankle_z > FOOT_SOLE_OFFSET + 0.10,
                            first_contact_s[1] is not None,
                        )
                    torso_rotation = Rotation.from_quat(
                        robot.body_data.body_link_pose[0, torso_index, 3:].cpu().numpy()
                    )
                    relative = (
                        torso_rotation * initial_torso_rotation.inv()
                    ).as_matrix()
                    torso_yaw_deg = float(
                        np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))
                    )
                    if abs(torso_yaw_deg) > abs(peak_torso_yaw_deg):
                        peak_torso_yaw_deg = torso_yaw_deg
                if video is not None:
                    if rp_hand_step:
                        center = np.array(
                            [
                                0.28 + 0.66 * root[0],
                                0.60 * root[1] - 0.26,
                                0.76 + 0.90 * max(root[2] - start_xyz[2], 0.0),
                            ]
                        )
                        camera_offset = _RP_STEP_CAMERA_OFFSET
                    elif scenario == "step-up":
                        center = np.array(
                            [
                                0.18 + 0.55 * root[0],
                                0.60 * root[1],
                                0.72 + 0.44 * max(root[2] - start_xyz[2], 0.0),
                            ]
                        )
                        camera_offset = _STEP_CAMERA_OFFSET
                    else:
                        center = np.array([root[0] + 0.10, root[1], 0.69])
                        camera_offset = _CAMERA_OFFSET
                    camera.look_at(
                        (center + camera_offset).tolist(),
                        center.tolist(),
                        [0, 0, 1],
                    )
                    sim.sync_render_state()
                    camera.render()
                    rgb = np.array(camera.get_rgb_map(), copy=True)[..., :3]
                    if np.issubdtype(rgb.dtype, np.floating):
                        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    video.append_data(rgb)

        dx, dy = (float(x) for x in root[:2] - start_xyz[:2])
        duration = sim.simulation_time - start_time
        welded = payload_kg == 0 or max_weld_error <= 0.03
        if scenario == "walk":
            task_passed = abs(dx - 3.5) <= 0.15 and abs(duration - 8.0) <= PHYSICS_DT
        else:
            task_passed = stable_frames * CONTROL_DT >= 0.50
            if video_step:
                left, right = first_contact_s
                task_passed &= left is not None and right is not None and right < left
        return {
            "scenario": scenario,
            "resolved_solver_mode": solver_features.get("backend"),
            "vbd_dynamic_bodies": solver_features.get("vbd_dynamic_body_count"),
            "payload_kg": measured_mass,
            "payload_size_m": (
                (2 * model.geom("crate_box").size).tolist()
                if crate is not None
                else None
            ),
            "distance_forward_m": dx,
            "distance_lateral_m": dy,
            "duration_s": round(duration, 9),
            "control_steps": len(loop_ms),
            "control_loop_p95_ms": float(np.percentile(loop_ms, 95)),
            "min_root_height_m": min_root_height,
            "fell": bool(fall),
            "max_weld_position_error_m": max_weld_error,
            "welded": bool(welded),
            "initial_wrist_crate_clearance_m": initial_wrist_gap,
            "minimum_wrist_crate_clearance_m": min_wrist_gap,
            "initial_hand_crate_clearance_m": initial_hand_gap,
            "minimum_hand_crate_clearance_m": min_hand_gap,
            "final_two_foot_support_s": stable_frames * CONTROL_DT,
            "longest_two_foot_support_s": max_stable_frames * CONTROL_DT,
            "step_motion": (
                {
                    "reference": args.step_reference,
                    "first_platform_contact_s": dict(
                        zip(("left", "right"), first_contact_s, strict=True)
                    ),
                    "peak_torso_yaw_change_deg": peak_torso_yaw_deg,
                    "final_torso_yaw_change_deg": torso_yaw_deg,
                    "swing_contacts": (
                        swing_contacts.summary() if swing_contacts is not None else None
                    ),
                }
                if platform_contacts is not None
                else None
            ),
            "platform_height_m": platform.size[2] if platform is not None else None,
            "platform_size_m": platform.size if platform is not None else None,
            "scene": str(usd),
            "video": str(output / f"{stem}.mp4") if want_video else None,
            "passed": bool(
                task_passed
                and welded
                and not fall
                and (min_wrist_gap is None or min_wrist_gap >= 0)
                and (min_hand_gap is None or min_hand_gap >= 0)
                and (swing_contacts is None or swing_contacts.passed)
            ),
        }
    finally:
        camera = None
        sim.destroy(exit_process=False)


def _verify_payload_welds(
    sim: SimulationManager, wrists: tuple[str, ...], crate_name: str
) -> None:
    """Check both enabled welds survived DexUni scene partitioning."""
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend
    import mujoco

    solver = get_newton_backend(sim.get_world()).mujoco_solver
    if solver is None:
        raise RuntimeError("DexUni did not prepare the MuJoCo weld partition.")
    model = solver.mj_model
    weld_pairs = [
        (
            model.body(int(model.eq_obj1id[index])).name,
            model.body(int(model.eq_obj2id[index])).name,
        )
        for index in range(model.neq)
        if model.eq_active0[index] and model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD
    ]
    for wrist in wrists:
        if not any(
            (first.endswith(wrist) and second.endswith(crate_name))
            or (second.endswith(wrist) and first.endswith(crate_name))
            for first, second in weld_pairs
        ):
            raise RuntimeError(f"DexUni did not retain the payload weld for {wrist}.")


def _supported(query: ContactQuery, height: float) -> bool:
    buffer = query.fetch_solver_report()
    rows = buffer.data[: buffer.count]
    return bool(
        ((rows[:, 2] > height - 0.03) & (rows[:, 5].abs() > 0.7) & (rows[:, 9] > 0))
        .any()
        .item()
    )


def _loaded_contacts(query: ContactQuery) -> torch.Tensor:
    buffer = query.fetch_solver_report()
    rows = buffer.data[: buffer.count]
    return rows[buffer.impulse_valid[: buffer.count] & (rows[:, 9] > 5e-5)]


class _StepSwingContacts:
    """Track contacts that prevent a clean ascent at the control sampling rate."""

    def __init__(
        self, platform: ContactQuery, floor: ContactQuery, crate: ContactQuery
    ) -> None:
        self.platform = platform
        self.floor = floor
        self.crate = crate
        self.lift_s: float | None = None
        self.floor_samples = 0
        self.side_samples = 0
        self.crate_samples = 0

    def sample(self, time_s: float, height: float, lifted: bool, planted: bool) -> None:
        if self.lift_s is None and lifted:
            self.lift_s = round(time_s, 9)
        if self.lift_s is not None and not planted:
            self.floor_samples += len(_loaded_contacts(self.floor)) > 0
        rows = _loaded_contacts(self.platform)
        self.side_samples += bool(
            ((rows[:, 2] < height - 0.018) & (rows[:, 5].abs() < 0.6)).any().item()
        )
        self.crate_samples += len(_loaded_contacts(self.crate)) > 0

    @property
    def passed(self) -> bool:
        return self.lift_s is not None and not (
            self.floor_samples or self.side_samples or self.crate_samples
        )

    def summary(self) -> dict[str, float | int | None]:
        return {
            "right_ankle_lift_s": self.lift_s,
            "right_floor_recontact_samples": self.floor_samples,
            "platform_side_contact_samples": self.side_samples,
            "crate_platform_contact_samples": self.crate_samples,
        }


def _cpu_platform_contacts(
    sim: SimulationManager,
    feet: tuple[str, ...],
    platform_path: str,
    height: float,
) -> Callable[[], tuple[bool, ...]]:
    """Read active contacts from Newton's host MuJoCo solver."""
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend
    import mujoco
    import numpy as np

    solver = get_newton_backend(sim.get_world()).mujoco_solver
    model, data = solver.mj_model, solver.mj_data
    foot_matches = tuple(
        [body for body in range(model.nbody) if model.body(body).name.endswith(name)]
        for name in feet
    )
    platform_geoms = {
        geom
        for geom in range(model.ngeom)
        if model.geom(geom).name == platform_path
        or model.geom(geom).name.startswith(f"{platform_path}_")
    }
    if any(len(matches) != 1 for matches in foot_matches) or len(platform_geoms) != 1:
        raise RuntimeError("MuJoCo scene must contain both feet and one platform geom.")
    foot_bodies = tuple(matches[0] for matches in foot_matches)
    platform_geom = next(iter(platform_geoms))
    force = np.zeros(6)

    def supported() -> tuple[bool, ...]:
        feet_in_contact = set()
        for index, contact in enumerate(data.contact):
            if (
                contact.efc_address < 0
                or platform_geom not in contact.geom
                or contact.pos[2] <= height - 0.03
                or abs(contact.frame[2]) <= 0.7
            ):
                continue
            other_geom = contact.geom[int(contact.geom[0] == platform_geom)]
            mujoco.mj_contactForce(model, data, index, force)
            if force[0] > 0:
                feet_in_contact.add(int(model.geom_bodyid[other_geom]))
        return tuple(body in feet_in_contact for body in foot_bodies)

    return supported


def main(argv: list[str] | None = None) -> None:
    """Run one or both reconstructed PartyMimic carrying scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", required=True, type=Path, help="Official sparse5 ONNX"
    )
    parser.add_argument("--scenario", choices=(*_SCENARIOS, "all"), default="all")
    parser.add_argument(
        "--hand",
        choices=("proxy", "rp"),
        default="proxy",
        help="Use simplified visuals or the official RP Hands in a fixed carrying pose.",
    )
    parser.add_argument(
        "--step-reference",
        choices=("baseline", "video"),
        default="baseline",
        help="Use the baseline or the video-matched step motion.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--robot-xml", type=Path, default=_ROBOT_XML)
    parser.add_argument("--converter-python", type=Path, default=_CONVERTER)
    parser.add_argument(
        "--solver", choices=("mujoco_warp", "dexuni"), default="mujoco_warp"
    )
    parser.add_argument(
        "--device", help="Physics device (MuJoCo defaults to cpu, DexUni to cuda:0)"
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Run physics without a renderer"
    )
    parser.add_argument(
        "--unloaded", action="store_true", help="Run without the welded payload"
    )
    args = parser.parse_args(argv)
    if args.device is None:
        args.device = "cuda:0" if args.solver == "dexuni" else "cpu"
    if not args.robot_xml.is_file() or not args.converter_python.is_file():
        parser.error("Prepare the pinned RP1 and USD converter with prepare.py first.")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("Use cpu or a CUDA device for Newton/MuJoCo physics.")
    if args.solver == "dexuni" and args.device == "cpu":
        parser.error("This scene's DexUni USD collision masks require a CUDA device.")
    # Native headless rendering needs no display server on Linux.
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort

    from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy
    from embodichain.lab.sim import SimulationManager

    policy = PartyMimicPolicy.from_onnx(args.policy)
    provenance_path = args.policy.with_suffix(".provenance.json")
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    hashes = {
        "onnx": _sha256(args.policy),
        "yaml": _sha256(args.policy.with_suffix(".yaml")),
    }
    recorded = provenance.get("artifacts", {})
    if recorded and any(
        recorded.get(args.policy.with_suffix(f".{suffix}").name) != digest
        for suffix, digest in hashes.items()
    ):
        raise RuntimeError("The ONNX artifacts differ from their export provenance.")
    scenarios = _SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = []
    for scenario in scenarios:
        try:
            result = _run(args, scenario, policy)
        finally:
            SimulationManager.flush_cleanup_queue()
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.solver == "dexuni":
        physics_backend = "Newton/DexUni"
    else:
        physics_backend = (
            "Newton/MuJoCo CPU" if args.device == "cpu" else "Newton/MuJoCo Warp"
        )
    hand_asset = None
    if args.hand == "rp":
        from embodichain.data import get_data_path
        from embodichain.extensions.robots.roboparty.rp1.assets import RoboPartyRP1

        from ._rp_hand import RP_HAND_ASSET

        hand_asset = {
            "source": RP_HAND_ASSET,
            "revision": RoboPartyRP1.revision,
            "mjcf_sha256": _sha256(Path(get_data_path(RP_HAND_ASSET))),
        }
    metrics = {
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "onnx_sha256": hashes["onnx"],
        "deployment_sha256": hashes["yaml"],
        "partymimic_revision": provenance.get("partymimic_revision"),
        "rp1_description_revision": provenance.get("rp1_description_revision"),
        "onnxruntime": ort.__version__,
        "execution_provider": "CPUExecutionProvider",
        "onnx_max_abs_error": provenance.get("onnx_max_abs_error"),
        "physics_backend": physics_backend,
        "physics_device": args.device,
        "hand_model": args.hand,
        "hand_asset": hand_asset,
        "reference_source": "Reconstructed RP1 sparse5 references with authored footholds",
        "scene_source": "Geometry and appearance reconstructed from Carrying loads footage",
        "video_resolution": _VIDEO_RESOLUTION,
        "video_fps": _VIDEO_FPS,
        "scenarios": results,
    }
    path = args.output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if any(not result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
