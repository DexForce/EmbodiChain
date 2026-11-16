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
"""Native physical-model smoke checks; does not attest demonstration success."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np

__all__: list[str] = []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_physical_robot(
    source: Path, output: Path, *, device: str = "cpu"
) -> dict[str, Any]:
    """Measure prepared robot import, gripper motion, and bilateral pad contacts.

    The diagnostic holds the recording's initial arm/body pose and commands the
    existing gripper calibration at 90 mm and 10 mm. Static 15 mm-radius spheres are
    placed between the observed open pad centers only after an empty closing
    cycle. Both pads must then contact their sphere and release on reopening.
    This verifies collision and actuator plumbing, not task success, grasp
    stability, hardware validity, or accuracy of the source inertial estimates.

    Args:
        source: Prepared recording directory with its repaired physical URDF.
        output: Fresh directory receiving the native trace and qualification report.
        device: Default-physics compute device. CPU matches diagnostic replay defaults.

    Returns:
        Passed report binding exact physical inputs and the measured native run.

    Raises:
        ValueError: Calibration or physical input is incomplete.
        RuntimeError: Native geometry, motion, contact, or release checks fail.
    """
    from embodichain.lab.sim import SimulationManager

    failure: BaseException | None = None
    try:
        report = _qualify_physical_robot(source, output, device=device)
    except BaseException as error:
        # Constructor failure can queue cleanup without returning a manager.
        # Release native views retained by unwound frames before draining it.
        pending = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            traceback.clear_frames(current.__traceback__)
            pending.extend(
                chained
                for chained in (current.__cause__, current.__context__)
                if chained is not None
            )
        failure = error.with_traceback(None)
    try:
        SimulationManager.flush_cleanup_queue()
    except BaseException as error:
        if failure is None:
            raise
        failure.add_note(f"Deferred robot qualification cleanup failed: {error!r}")
    if failure is not None:
        raise failure
    return report


def _qualify_physical_robot(
    source: Path, output: Path, *, device: str
) -> dict[str, Any]:
    """Run and persist native checks before the outer caller drains cleanup."""
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationRootPropertiesCfg,
        CollisionPropertiesCfg,
        DefaultPhysicsCfg,
        JointDrivePropertiesCfg,
        RenderCfg,
        RigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        RobotCfg,
    )
    from embodichain.lab.sim.shapes import SphereCfg
    from embodichain.lab.sim.sensors.contact import (
        ArticulationContactFilterCfg,
        ContactSensorCfg,
    )
    from embodichain.data_pipeline.real2sim._geometry import _robot_geometry_sha256

    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source / "robot.json").read_text())
    urdf = (source / meta["urdf"]).resolve()
    with np.load(source / "trajectory_original.npz", allow_pickle=False) as data:
        names = data["joint_names"].tolist()
        original = np.array(data["qpos"][0], dtype=float)
    tree = ET.parse(urdf).getroot()
    authored = {
        l.get("name"): len(l.findall("collision"))
        for l in tree.findall("link")
        if l.findall("collision")
    }
    if not authored:
        raise ValueError("Physical robot URDF has no collision geometry")
    resources = {urdf, source / "robot.json", source / "trajectory_original.npz"}
    resources.update(
        (urdf.parent / node.get("filename")).resolve()
        for node in tree.findall(".//mesh") + tree.findall(".//texture")
    )
    fingerprints = {str(path): _sha256(path) for path in sorted(resources)}
    if any(not path.is_relative_to(source) for path in resources):
        raise ValueError("Native qualification requires package-local robot resources")
    collision_meshes = tuple(
        (urdf.parent / node.get("filename")).resolve()
        for node in tree.findall("./link/collision/geometry/mesh")
    )
    geometry_identity = _robot_geometry_sha256(urdf, collision_meshes)
    calibrations = meta.get("gripper_calibrations", [])
    if len(calibrations) != len(meta["side_names"]) or not calibrations:
        raise ValueError(
            "Each gripper requires explicit calibrated joint and pad identities"
        )
    for calibration in calibrations:
        if (
            len(calibration["pad_links"]) != 2
            or len(set(calibration["pad_links"])) != 2
            or not set(calibration["pad_links"]) <= set(authored)
        ):
            raise ValueError("Each calibrated pad needs explicit collision geometry")
        widths = np.asarray(calibration["width_m"])
        if widths.min() > 0.01 or widths.max() < 0.09:
            raise ValueError(
                "Qualification requires the authored 10–90 mm calibration range"
            )
    joints = {j.get("name"): j for j in tree.findall("joint")}
    stiffness, damping, effort = {}, {}, {}
    for name in names:
        joint = joints[name]
        finger = (
            joint.get("type") == "prismatic" or "gripper" in name or "finger" in name
        )
        stiffness[name], damping[name] = (2000, 70) if finger else (30000, 600)
        limit = joint.find("limit")
        effort[name] = (
            max(0.1, float(limit.get("effort", "40" if finger else "100")))
            if limit is not None
            else 100
        )

    def command(width: float) -> np.ndarray:
        q = original.copy()
        for c in meta["gripper_calibrations"]:
            widths = np.array(c["width_m"])
            grid = np.array(c["qpos"])
            order = np.argsort(widths)
            for column, name in enumerate(c["joint_names"]):
                q[names.index(name)] = np.interp(
                    width, widths[order], grid[order, column]
                )
        for i, name in enumerate(names):
            limit = joints[name].find("limit")
            if limit is not None and joints[name].get("type") != "continuous":
                q[i] = np.clip(
                    q[i],
                    float(limit.get("lower", "-inf")),
                    float(limit.get("upper", "inf")),
                )
        return q

    report = dict(
        schema="prepared-robot-native-qualification/v1",
        passed=False,
        status="failed",
        task_success_verified=False,
        scope="Native authored-collision import, stable stepping, calibrated opening/closing, independent pad contact probe",
        python=sys.version.split()[0],
        dexsim=importlib.metadata.version("dexsim-engine"),
        torch=torch.__version__,
        backend="default",
        device=device,
        physics_dt_s=1 / 600,
        urdf=urdf.relative_to(source).as_posix(),
        source_urdf_sha256=_sha256(urdf),
        authored_collision_counts=authored,
        input_sha256={
            Path(path).relative_to(source).as_posix(): checksum
            for path, checksum in fingerprints.items()
        },
        robot_geometry_sha256=geometry_identity,
        qualification_code_sha256=_sha256(Path(__file__)),
        runtime={
            "python": sys.version.split()[0],
            "dexsim": importlib.metadata.version("dexsim-engine"),
            "torch": torch.__version__,
            "backend": "default",
            "device": device,
        },
    )
    sim = None
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                physics_dt=1 / 600,
                physics_cfg=DefaultPhysicsCfg(device=device),
                render_cfg=RenderCfg(renderer="no-render"),
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
            )
        )
        robot = sim.add_robot(
            RobotCfg(
                uid="robot",
                fpath=str(urdf),
                build_pk_chain=False,
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True, min_position_iters=32, min_velocity_iters=8
                ),
                joint_drive_props=JointDrivePropertiesCfg(
                    drive_type="force",
                    stiffness=stiffness,
                    damping=damping,
                    max_effort=effort,
                ),
                attrs=RigidBodyPhysicsCfg(
                    collision_props=CollisionPropertiesCfg(
                        contact_offset=0.001, rest_offset=0.0
                    ),
                    material_props=RigidBodyMaterialCfg(
                        static_friction=0.95, dynamic_friction=0.9
                    ),
                ),
            )
        )
        probes = {
            side: sim.add_rigid_object(
                RigidObjectCfg(
                    uid="probe_" + side,
                    body_type="static",
                    shape=SphereCfg(radius=0.015),
                    init_pos=(10.0, i * 1.0, 10.0),
                    attrs=RigidBodyPhysicsCfg(),
                )
            )
            for i, side in enumerate(meta["side_names"])
        }
        sim.prepare()
        sim.spawn_result.get_actor("default_plane").enable_collision(False)
        ids = [robot.joint_names.index(name) for name in names]
        qopen = command(0.09)
        robot.set_qpos(
            torch.tensor(qopen[None], dtype=torch.float32, device=robot.device),
            joint_ids=ids,
            target=False,
        )
        robot.set_qpos(
            torch.tensor(qopen[None], dtype=torch.float32, device=robot.device),
            joint_ids=ids,
        )
        pads = [p for c in meta["gripper_calibrations"] for p in c["pad_links"]]
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["probe_" + s for s in meta["side_names"]],
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid="robot", link_name_list=pads
                    )
                ],
                max_contacts_per_env=256,
            )
        )
        pad_ids = {p: int(sensor.get_actor_ids("robot", [p])[0, 0]) for p in pads}
        probe_ids = {
            s: int(sensor.get_actor_ids("probe_" + s)[0, 0]) for s in meta["side_names"]
        }
        actual = {l: len(robot.get_link_collision_shapes(l)) for l in authored}
        report["native_collision_counts"] = actual
        if any(n == 0 for n in actual.values()):
            raise RuntimeError(
                "An authored collision link has no native collision shapes"
            )

        def pad_cloud(link: str) -> np.ndarray:
            arrays = []
            pose = robot.get_link_pose(link, to_matrix=True)[0].cpu().numpy()
            for shape in robot.get_link_collision_shapes(link):
                if shape.vertices is None:
                    raise RuntimeError("Expected authored pad mesh collision")
                v = shape.vertices.cpu().numpy()
                local = shape.local_pose.cpu().numpy()
                world = pose @ local
                arrays.append(v @ world[:3, :3].T + world[:3, 3])
            return np.concatenate(arrays)

        trace: list[dict[str, Any]] = []
        report["stages"] = []

        def drive(width: float, steps: int, label: str) -> dict[str, Any]:
            q = command(width)
            target = torch.tensor(q[None], dtype=torch.float32, device=robot.device)
            robot.set_qpos(target, joint_ids=ids)
            robot.set_qvel(torch.zeros_like(target), joint_ids=ids)
            hit = {p: 0 for p in pads}
            peak = {p: 0.0 for p in pads}
            terminal_hit = {p: 0 for p in pads}
            dropped = 0
            for step in range(steps):
                sensor.begin_control_step()
                sim.update(1 / 600, 1)
                sensor.update()
                values = (
                    robot.get_qpos(),
                    robot.get_qvel(),
                    robot.body_data.body_link_pose,
                    robot.body_data.body_link_vel,
                )
                if any(not bool(torch.isfinite(v).all()) for v in values):
                    raise FloatingPointError("Non-finite robot state")
                dropped += int(sensor.evidence_dropped_contacts)
                data = sensor.get_data()
                mask = data["is_valid"][0]
                pairs = data["user_ids"][0][mask].cpu().numpy()
                impulse = data["impulse"][0][mask].cpu().numpy()
                sample_contacts: set[str] = set()
                for side, c in zip(meta["side_names"], meta["gripper_calibrations"]):
                    for pad in c["pad_links"]:
                        for pair, value in zip(pairs, impulse):
                            if set(map(int, pair)) == {pad_ids[pad], probe_ids[side]}:
                                sample_contacts.add(pad)
                                peak[pad] = max(
                                    peak[pad], float(np.linalg.norm(value)) * 600
                                )
                for pad in sample_contacts:
                    hit[pad] += 1
                    if step >= steps - 60:
                        terminal_hit[pad] += 1
                trace.append(
                    dict(
                        time=float(sim.simulation_time),
                        stage=label,
                        qpos=robot.get_qpos()[0, ids].cpu().tolist(),
                        contact=[pad in sample_contacts for pad in pads],
                    )
                )
            row = dict(
                stage=label,
                requested_width_m=width,
                steps=steps,
                max_joint_tracking_error_rad_m=float(
                    abs(robot.get_qpos()[0, ids].cpu().numpy() - q).max()
                ),
                pad_probe_contact_samples=hit,
                terminal_contact_samples=terminal_hit,
                dropped_contact_rows=dropped,
                peak_pad_probe_force_n=peak,
                pad_centers={p: pad_cloud(p).mean(0).tolist() for p in pads},
            )
            row["pad_center_distance_m"] = {
                side: float(
                    np.linalg.norm(
                        np.asarray(row["pad_centers"][c["pad_links"][0]])
                        - np.asarray(row["pad_centers"][c["pad_links"][1]])
                    )
                )
                for side, c in zip(meta["side_names"], calibrations)
            }
            report["stages"].append(row)
            print(json.dumps(row), flush=True)
            return row

        stages = [
            drive(0.09, 300, "free_open"),
            drive(0.01, 300, "free_close"),
            drive(0.09, 300, "free_reopen"),
        ]
        for side, c in zip(meta["side_names"], meta["gripper_calibrations"]):
            center = np.mean([pad_cloud(p).mean(0) for p in c["pad_links"]], axis=0)
            probes[side].set_local_pose(
                torch.tensor(
                    [[*center, 0, 0, 0, 1]], dtype=torch.float32, device=robot.device
                )
            )
        stages += [
            drive(0.09, 120, "probe_open"),
            drive(0.01, 600, "probe_close"),
            drive(0.09, 300, "probe_release"),
        ]
        report["steps"] = sum(stage["steps"] for stage in stages)
        close = stages[-2]
        release = stages[-1]
        report["terminal_contact_window_s"] = 0.1
        report["pad_center_distance_change_m"] = {
            side: stages[0]["pad_center_distance_m"][side]
            - stages[1]["pad_center_distance_m"][side]
            for side in meta["side_names"]
        }
        report["checks"] = dict(
            finite_stepping=True,
            native_collisions_present=all(count > 0 for count in actual.values()),
            calibrated_pad_motion=all(
                x > 0.05 for x in report["pad_center_distance_change_m"].values()
            ),
            free_tracking=all(
                stage["max_joint_tracking_error_rad_m"] < 0.05 for stage in stages[:4]
            ),
            no_initial_probe_contact=all(
                n == 0 for n in stages[3]["pad_probe_contact_samples"].values()
            ),
            bilateral_sustained_contact=all(
                close["terminal_contact_samples"][p] >= 50 for p in pads
            ),
            positive_contact_impulse=all(
                close["peak_pad_probe_force_n"][p] > 1e-6 for p in pads
            ),
            contact_released=all(
                release["terminal_contact_samples"][p] == 0 for p in pads
            ),
            reopened_tracking=release["max_joint_tracking_error_rad_m"] < 0.05,
            complete_contact_capture=all(
                stage["dropped_contact_rows"] == 0 for stage in stages
            ),
            unchanged_inputs=all(
                _sha256(Path(path)) == expected
                for path, expected in fingerprints.items()
            ),
        )
        report["passed"] = all(report["checks"].values())
        report["status"] = "passed" if report["passed"] else "failed"
        np.savez_compressed(
            output / "trace.npz",
            timestamps=[r["time"] for r in trace],
            qpos=[r["qpos"] for r in trace],
            stage=[r["stage"] for r in trace],
            joint_names=names,
            contact_found=[r["contact"] for r in trace],
            pad_links=pads,
        )
        report["traces"] = {
            "path": "trace.npz",
            "sha256": _sha256(output / "trace.npz"),
        }
        if not report["passed"]:
            raise RuntimeError("Gripper motion or pad-contact qualification failed")
        return report
    except BaseException as error:
        report.update(status="failed", passed=False, error=str(error))
        raise
    finally:
        try:
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        finally:
            if sim is not None:
                sim.destroy(exit_process=False)
