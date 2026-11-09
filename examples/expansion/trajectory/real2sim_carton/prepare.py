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

"""Qualify the pinned DROID input and prepare the carton demo deployment.

Only this selected episode is downloaded. Manual measurements are frozen in
annotations; admission recomputes their residuals rather than fitting to the
desired simulation output. Run ``python prepare.py --help`` for the two steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil

import cv2
import gymnasium
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import torch
import yaml

from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim._geometry import (
    _canonical_robot_geometry_sha256,
)
from embodichain.data_pipeline.real2sim import (
    JointSpec,
    PhaseAnnotation,
    prepare_prototype,
)
from embodichain.data_pipeline.real2sim.droid import (
    build_droid_trajectory,
    load_droid_episode,
    save_droid_unit_evidence,
    verify_droid_unit_evidence,
)
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import (
    _create_parser,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    SampledRobotCollisionChecker,
)
from embodichain.lab.task_program.integrations.nominal import (
    BoundTaskProgramNominalSources,
)
from embodichain_robots.robotiq.two_finger import robotiq85_joint_positions
from embodichain.utils.config_paths import resolve_config_path
from embodichain.toolkits.graspkit.box_region import (
    BoxSideGraspRegionCfg,
    project_box_side_grasp,
)

__all__: list[str] = []

_TASK_CONFIG = "extensions/embodichain_tasks/configs/tasks/manipulation/carton_stocking/task.panda.yaml"


def task_config() -> Path:
    """Locate the SDK-owned carton task in a checkout or installed wheel."""
    return resolve_config_path(_TASK_CONFIG).resolve()


ROOT = Path(__file__).resolve().parent
DT = 1 / 30


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def require_prepared(prepared: Path) -> None:
    """Reject incomplete or unqualified preparation before scheduling work."""
    path = prepared / "qualification.json"
    if not path.is_file():
        raise ValueError(
            f"prepared input is incomplete: missing {path}; run preparation first"
        )
    qualification = json.loads(path.read_text(encoding="utf-8"))
    if qualification.get("admitted") is not True:
        failed = [
            name
            for name, passed in qualification.get("checks", {}).items()
            if not passed
        ]
        raise ValueError(
            f"prepared input failed qualification: {', '.join(failed) or 'admitted is false'}; "
            f"inspect {path} and rebuild preparation"
        )
    for name in ("task.measured.yaml", "profile.measured.yaml"):
        if not (prepared / name).is_file():
            raise ValueError(
                f"prepared input is incomplete: missing {prepared / name}; rebuild preparation"
            )


def acquire(directory: Path) -> None:
    """Copy pinned raw files from the shared demo asset cache when needed."""
    lock = json.loads((ROOT / "source.lock.json").read_text())
    missing = []
    for item in lock["files"]:
        path = directory / item["path"]
        if not path.exists():
            missing.append(item)
        elif path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError(f"pinned DROID source differs: {path}")
    if not missing:
        return
    resource = Path(get_data_path("Real2SimCarton/raw"))
    for item in missing:
        original = resource / item["path"]
        path = directory / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".copy")
        try:
            shutil.copyfile(original, temporary)
            if (
                temporary.stat().st_size != item["size"]
                or digest(temporary) != item["sha256"]
            ):
                raise ValueError(f"cached DROID source differs: {item['path']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def matrix_layouts() -> list[dict]:
    return yaml.safe_load(
        (task_config().parent / "real2sim/collection.yaml").read_text()
    )["layouts"]


def make_environment(task: Path):
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(task),
            "--device",
            "cpu",
            "--headless",
            "--renderer",
            "no-render",
            "--disable-sensor",
            "--filter_dataset_saving",
        ]
    )
    discover_task_packages()
    execute_init_hooks()
    cfg, raw, actions = build_env_cfg_from_args(args)
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    return gymnasium.make(id=raw["id"], cfg=cfg, **actions)


def robot_evidence(robot, bodies) -> tuple[dict, tuple[JointSpec, ...]]:
    names = tuple(robot.joint_names)
    mimic = dict(
        zip(
            robot.mimic_ids,
            zip(robot.mimic_parents, robot.mimic_multipliers, robot.mimic_offsets),
        )
    )
    joints = tuple(
        JointSpec(
            name,
            "revolute",
            mimic_driver=names[int(mimic[i][0])] if i in mimic else None,
            mimic_multiplier=float(mimic[i][1]) if i in mimic else 1.0,
            mimic_offset=float(mimic[i][2]) if i in mimic else 0.0,
        )
        for i, name in enumerate(names)
    )
    checker = SampledRobotCollisionChecker(
        robot.cfg.fpath,
        joint_names=names,
        bodies=bodies,
        world_from_base=robot.get_local_pose(to_matrix=True)[0].cpu().numpy(),
    )
    return (
        dict(
            joint_names=list(names),
            robot_geometry_sha256=checker.geometry_sha256,
            object_geometry_sha256=bodies[0].geometry_sha256,
            target_geometry_sha256=bodies[1].geometry_sha256,
            urdf_path=robot.cfg.fpath,
            joints=[asdict(joint) for joint in joints],
            link_masses_kg=dict(
                zip(robot.link_names, robot.body_data.mass[0].cpu().tolist())
            ),
            qpos_limits=robot.get_qpos_limits()[0].cpu().tolist(),
            qvel_limits=robot.get_qvel_limits()[0].cpu().tolist(),
            physics_parameters=dict(
                link_inertials={
                    key: value.attrs.mass_props.to_dict()
                    for key, value in robot.cfg.link_attrs.items()
                },
                drive=robot.cfg.joint_drive_props.to_dict(),
                root=robot.cfg.root_props.to_dict(),
                scope="identified Panda, nominal Robotiq CAD, 5 g pad assumption; no real dynamics qualification",
            ),
        ),
        joints,
    )


def _calibration_checks(robot, episode, annotations, box) -> dict:
    cfg = robot.cfg
    q = torch.as_tensor(np.array(episode.arm_qpos), dtype=torch.float32)
    fk = cfg.build_pk_serial_chain()["arm"].forward_kinematics(q).get_matrix().numpy()
    reported = np.repeat(np.eye(4)[None], len(q), axis=0)
    reported[:, :3, :3] = Rotation.from_quat(episode.reported_poses[:, 3:]).as_matrix()
    reported[:, :3, 3] = episode.reported_poses[:, :3]
    camera = json.loads((ROOT / "annotations/camera1_registration.json").read_text())
    intr = json.loads((ROOT / "annotations/intrinsics.selected.json").read_text())[
        "entry"
    ]
    model, geometry, _ = pin.buildModelsFromUrdf(
        cfg.fpath, package_dirs=[str(Path(cfg.fpath).parent)]
    )
    data, gdata = model.createData(), geometry.createData()
    names = list(model.names[1:])
    gripper = robotiq85_joint_positions(episode.gripper_closed_fraction)

    def tips(index):
        values = dict(zip(cfg.control_parts["arm"], episode.arm_qpos[index]))
        values.update(zip(cfg.control_parts["eef"], gripper[index]))
        pin.forwardKinematics(model, data, np.array([values[name] for name in names]))
        pin.updateGeometryPlacements(model, data, geometry, gdata)
        points = []
        for link in ("left_inner_finger_pad", "right_inner_finger_pad"):
            i = next(
                i
                for i, shape in enumerate(geometry.geometryObjects)
                if model.frames[shape.parentFrame].name == link
            )
            shape, transform = geometry.geometryObjects[i], gdata.oMg[i].homogeneous
            points.append(
                transform[:3, :3] @ [0, 0, float(shape.geometry.halfSide[2])]
                + transform[:3, 3]
            )
        return np.array(points)

    def project(points, serial, camera_pose):
        transform = np.linalg.inv(camera_pose)
        local = points @ transform[:3, :3].T + transform[:3, 3]
        if np.any(local[:, 2] <= 0):
            raise ValueError("calibration puts evidence behind camera")
        fx, cx, fy, cy = intr[serial]["cameraMatrix"]
        return local[:, :2] / local[:, 2, None] * [fx, fy] + [cx, cy]

    c1 = np.array(camera["camera_to_base"])
    errors = np.concatenate(
        [
            project(tips(int(index)), camera["camera_serial"], c1) - points
            for index, points in camera["pixel_landmarks"].items()
        ]
    )
    holdout = annotations["holdout"]
    point = project(tips(holdout["original_index"])[1:], camera["camera_serial"], c1)[0]
    x = np.array(box["parameters"])
    vertices = (
        np.array(list(product([-1, 1], repeat=3)))
        * x[4:]
        / 2
        @ Rotation.from_euler("z", x[3]).as_matrix().T
        + x[:3]
    )
    box_errors = []
    for serial, points in box["pixel_bboxes"].items():
        uv = project(vertices, serial, np.array(box["camera_to_base"][serial]))
        box_errors.extend(np.r_[uv.min(axis=0), uv.max(axis=0)] - points)
    metrics = dict(
        flange_fk_position_m=float(
            np.linalg.norm(fk[:, :3, 3] - reported[:, :3, 3], axis=1).max()
        ),
        flange_fk_rotation_rad=float(
            Rotation.from_matrix(fk[:, :3, :3] @ reported[:, :3, :3].transpose(0, 2, 1))
            .magnitude()
            .max()
        ),
        camera_fit_rms_px=float(np.sqrt(np.mean(errors**2))),
        camera_fit_max_coordinate_px=float(np.abs(errors).max()),
        camera_holdout_px=float(np.linalg.norm(point - holdout["pixel"])),
        box_bbox_rms_px=float(np.sqrt(np.mean(np.asarray(box_errors) ** 2))),
        box_bbox_max_coordinate_px=float(np.abs(box_errors).max()),
    )
    checks = {
        name: value <= annotations["admission_limits"][name]
        for name, value in metrics.items()
    }
    checks.update(
        physical_box_within_opening=bool(0 < x[5] < 0.085),
        source_commands_submitted=bool(episode.submitted[25:93].all()),
        video_offset_explicit=annotations["video_state_offset"] == 0,
        original_robot_model=(
            getattr(cfg, "arm_model", None) == "panda"
            and Path(cfg.urdf_cfg.components["hand"]["urdf_path"]).resolve()
            == Path(
                get_data_path("Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf")
            ).resolve()
            and tuple(cfg.control_parts["arm"])
            == tuple(f"Joint{i}" for i in range(1, 8))
            and tuple(cfg.control_parts["eef"])
            == (
                "finger_joint",
                "left_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_inner_finger_joint",
            )
        ),
    )
    return dict(
        metrics=metrics,
        checks=checks,
        admitted=all(checks.values()),
        scope=annotations["scope"],
        geometry_uncertainty="Multiview approximate cuboid; weak measured-aperture prior sigma 4 mm. Mass/friction are simulation assumptions.",
        calibration_uncertainty="Conditional coaxial mount registration, manual landmarks approximately 4 px; no independent optical TCP ground truth.",
    )


def prepare(source_directory: Path, output: Path) -> None:
    acquire(source_directory)
    lock = json.loads((ROOT / "source.lock.json").read_text())
    for item in lock["annotations"]:
        if digest(ROOT / "annotations" / item["path"]) != item["sha256"]:
            raise ValueError(f"annotation revision changed: {item['path']}")
    output.mkdir(parents=True, exist_ok=False)
    annotation = json.loads((ROOT / "annotations/staples.json").read_text())
    box = json.loads((ROOT / "annotations/box_registration.json").read_text())
    episode = load_droid_episode(source_directory / "trajectory.h5")
    x = np.array(box["parameters"])
    source_object = np.eye(4)
    source_object[:3, :3] = Rotation.from_euler("z", x[3]).as_matrix()
    source_object[:3, 3] = x[:3]
    tcp = np.repeat(np.eye(4)[None], len(episode.timestamps), axis=0)
    tcp[:, :3, :3] = Rotation.from_quat(episode.reported_poses[:, 3:]).as_matrix()
    tcp[:, :3, 3] = episode.reported_poses[:, :3]
    link_to_tcp = np.eye(4)
    link_to_tcp[2, 3] = annotation["robot_mount_z_m"] + annotation["tcp_offset_m"]
    tcp = tcp @ link_to_tcp
    source_grasp = np.linalg.inv(source_object) @ tcp[60]
    release_object = tcp[83] @ np.linalg.inv(source_grasp)
    target_yaw = float(np.arctan2(release_object[1, 0], release_object[0, 0]))
    support_height = float(x[2] - x[6] / 2)
    source_target = np.eye(4)
    source_target[:3, :3] = Rotation.from_euler("z", target_yaw).as_matrix()
    source_target[:3, 3] = [*release_object[:2, 3], support_height / 2]
    object_pose, target_pose = source_object.copy(), source_target.copy()
    object_pose[:2, 3] += annotation["scene_xy_shift_m"]
    target_pose[:2, 3] += annotation["scene_xy_shift_m"]
    grasp_frame = np.eye(4)
    grasp_frame[:3, :3] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    region = BoxSideGraspRegionCfg(longitudinal_margin_m=0.010, grasp_height_m=-0.005)
    real_grasp = (
        project_box_side_grasp(source_grasp @ np.linalg.inv(grasp_frame), x[4:], region)
        @ grasp_frame
    )
    calibration = dict(
        world_from_base=np.eye(4).tolist(),
        reported_link_to_tcp=link_to_tcp.tolist(),
        gripper_mapping="Observed DROID width=0.085*(1-g); Robotiq linkage inverse; no command-as-observation substitution",
        revision=annotation["revision"],
    )
    official = task_config().parent
    environment = yaml.safe_load((official / "env.yaml").read_text())
    for uid, pose, size in (
        ("carton", object_pose, x[4:]),
        ("worktop", target_pose, [0.65, 0.65, support_height]),
    ):
        entry = next(
            item
            for kind in ("rigid_object", "background")
            for item in environment["simulation"][kind]
            if item["uid"] == uid
        )
        if not (
            np.allclose(entry["shape"]["size"], size, atol=1e-10, rtol=0)
            and np.allclose(entry["init_pos"], pose[:3, 3], atol=1e-10, rtol=0)
            and np.allclose(
                Rotation.from_euler("xyz", entry["init_rot"], degrees=True).as_matrix(),
                pose[:3, :3],
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError(
                "source registration differs from the SDK-owned physical scene"
            )
    (output / "environment.yaml").write_bytes((official / "env.yaml").read_bytes())
    (output / "program.yaml").write_bytes(
        (official / "task_program/program.yaml").read_bytes()
    )
    policy_path = resolve_config_path(
        "../../../components/execution_policies/real2sim_verified.yaml",
        base_dir=official,
    )
    (output / "execution_policy.yaml").write_bytes(policy_path.read_bytes())
    integration = yaml.safe_load(
        (official / "task_program/integration.yaml").read_text()
    )
    integration["runtime_services"].pop("nominal_motion_sources")
    integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"] = (
        real_grasp.reshape(-1).tolist()
    )
    write_yaml(output / "integration.qualification.yaml", integration)
    task = yaml.safe_load(task_config().read_text())
    task.update(
        id="CartonStockingQualification-v1",
        collection={"target_episodes": 1, "max_attempts": 1},
    )
    task.pop("real2sim")
    task["environment"]["component"] = "environment.yaml"
    task["embodiment"]["component"] = str(
        resolve_config_path(
            task["embodiment"]["component"], base_dir=official
        ).resolve()
    )
    task["task_program"] = dict(
        program="program.yaml",
        integration="integration.qualification.yaml",
        execution_policy="execution_policy.yaml",
    )
    write_yaml(output / "task.qualification.yaml", task)
    env = make_environment(output / "task.qualification.yaml")
    try:
        env.reset(seed=0, options={"save_data": False})
        target = env.unwrapped
        bodies = tuple(
            BoxCollisionBody.from_rigid_object(target.sim.get_rigid_object(uid))
            for uid in ("carton", "worktop")
        )
        snapshot, joints = robot_evidence(target.robot, bodies)
        fingerprint = BoundTaskProgramNominalSources(
            target.compile_task_program(target.cfg.task_program), {}
        ).program_fingerprint
        qualification = _calibration_checks(target.robot, episode, annotation, box)
        qualification["checks"].update(
            source_joint_layout=set(target.robot.joint_names)
            == {
                *snapshot["joint_names"][:7],
                "finger_joint",
                "left_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_inner_finger_joint",
            }
            and tuple(snapshot["joint_names"][:7])
            == tuple(f"Joint{i}" for i in range(1, 8)),
            # The shared geometry identity supports explicitly pinned legacy
            # aliases; source and physical calibration still require qualification.
            source_geometry_identity=snapshot["robot_geometry_sha256"]
            == _canonical_robot_geometry_sha256(
                "6b08aff1b8de77edfd1deab3da3f139d2ecfb20838bd21cf2d6207058c0d4d82"
            ),
        )
        qualification["admitted"] = all(qualification["checks"].values())
        qualification.update(
            source_id=lock["source_id"],
            source_revision=episode.sha256,
            annotation_sha256=digest(ROOT / "annotations/staples.json"),
            original_sample_count=len(episode.timestamps),
            video_frame_count=95,
            original_velocity_available=episode.arm_qvel is not None,
            frame_clock_map=dict(
                acquisition_frame=list(range(95)),
                source_seconds=episode.timestamps[:95].tolist(),
                encoding_seconds=(np.arange(95) / 60).tolist(),
            ),
            semantic_geometry=dict(
                object_pose=source_object.tolist(),
                target_pose=source_target.tolist(),
                object_size_m=x[4:].tolist(),
                target_yaw_rad=target_yaw,
            ),
            robot=snapshot,
        )
        write_json(output / "qualification.json", qualification)
        if not qualification["admitted"]:
            raise ValueError(
                "real input failed qualification; inspect qualification.json"
            )
        bindings, artifacts = [], {}
        hand = robotiq85_joint_positions(episode.gripper_closed_fraction)
        for index, unit_annotation in enumerate(annotation["phases"]):
            start, stop = unit_annotation["start"], unit_annotation["stop"]
            phases = tuple(
                PhaseAnnotation(
                    p["name"],
                    p["start"] - start,
                    p["stop"] - start,
                    p["kind"],
                    p["reference"],
                    (
                        ("joint_residual", "task_space_residual", "retime")
                        if p["kind"] == "free"
                        else ()
                    ),
                )
                for p in unit_annotation["phases"]
            )
            object_anchor = (
                source_object
                if index == 0
                else tcp[start] @ np.linalg.inv(source_grasp)
            )
            unit = build_droid_trajectory(
                episode,
                source_id=lock["source_id"],
                unit_id=f"call_{index}_{unit_annotation['skill_id']}",
                skill_id=unit_annotation["skill_id"],
                start=start,
                stop=stop,
                arm_joint_names=target.robot.cfg.control_parts["arm"],
                gripper_joint_names=target.robot.cfg.control_parts["eef"],
                gripper_joint_units=("rad",) * 6,
                observed_gripper_qpos=hand,
                phases=phases,
                world_from_base=np.eye(4),
                reported_link_to_tcp=link_to_tcp,
                object_pose=object_anchor,
                target_pose=None if index == 0 else source_target,
                calibration_revision=annotation["revision"],
                annotation_revision=annotation["revision"],
                **{
                    name: snapshot[name]
                    for name in ("robot_geometry_sha256", "object_geometry_sha256")
                },
                target_geometry_sha256=(
                    snapshot["target_geometry_sha256"] if index else None
                ),
            )
            manifest = prepare_prototype(
                unit,
                joints=joints,
                control_dt=DT,
                event_tolerance_s=1 / 60,
                directory=output / f"measured/prototypes/{index}",
            )
            evidence = save_droid_unit_evidence(
                episode,
                output / f"measured/evidence/{index}.npz",
                source_id=unit.source_id,
                unit_id=unit.unit_id,
                start=start,
                stop=stop,
                arm_joint_names=target.robot.cfg.control_parts["arm"],
                calibrated_gripper_qpos=hand,
                annotation={
                    **annotation,
                    "unit": unit_annotation,
                    "object_anchor_origin": (
                        "manual_multiview"
                        if index == 0
                        else "derived_rigid_hold_from_original_fk"
                    ),
                },
                calibration=calibration,
            )
            verify_droid_unit_evidence(
                evidence, source_directory / "trajectory.h5", unit
            )
            artifacts[index] = {
                "original_recording": dict(
                    path=str(source_directory / "trajectory.h5"), sha256=episode.sha256
                ),
                "source_evidence": dict(path=str(evidence), sha256=digest(evidence)),
            }
            binding = dict(
                call_index=index,
                program_fingerprint=fingerprint,
                provider_id="real2sim.prototype",
                revision="1",
                prototype_manifest=str(manifest),
                prototype_sha256=digest(manifest),
                object_entity_id="carton",
                authority="eef_pose",
                gripper_mode="semantic",
                require_collision_validation=True,
                refinement=dict(
                    max_translation_m=0.07,
                    max_rotation_rad=float(np.deg2rad(60)),
                    project_pick_lift=True,
                    project_pick_approach=True,
                    project_free_boundaries=True,
                    pick_entry_projection_samples=24,
                    smoothing_weight=20.0,
                    seed=7,
                ),
            )
            if index:
                binding["target_entity_id"] = "worktop"
            bindings.append(binding)
        integration["runtime_services"]["nominal_motion_sources"] = bindings
        write_yaml(output / "integration.measured.yaml", integration)
        task["task_program"]["integration"] = "integration.measured.yaml"
        task["task_program"]["execution_policy"] = "execution_policy.yaml"
        task["id"] = "CartonStockingPanda-v1"
        task["real2sim"] = {"profile": "profile.measured.yaml"}
        task["collection"]["target_episodes"] = 6
        write_yaml(output / "task.measured.yaml", task)
        profile = yaml.safe_load((official / "real2sim/collection.yaml").read_text())
        profile["source_artifacts"] = artifacts
        write_yaml(output / "profile.measured.yaml", profile)
        single = {**profile, "layouts": matrix_layouts()[:1], "recipes": ["nominal"]}
        write_yaml(output / "profile.measured.single.yaml", single)
        write_json(output / "joints.json", snapshot["joints"])
        write_json(
            output / "scene.lock.json",
            dict(
                schema="real2sim-carton-scene/v1",
                source_lock_sha256=digest(ROOT / "source.lock.json"),
                geometry=snapshot,
                object_size_m=x[4:].tolist(),
                object_pose=object_pose.tolist(),
                target_pose=target_pose.tolist(),
                support_height_m=support_height,
                source_to_scene_xy_m=annotation["scene_xy_shift_m"],
                source_grasp=source_grasp.tolist(),
                calibration=calibration,
            ),
        )
    finally:
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "prepare"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.source_dir.resolve())
    else:
        if args.output is None:
            parser.error("prepare requires --output")
        prepare(args.source_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
