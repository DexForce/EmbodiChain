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

"""Freeze independently captured synthetic inputs with explicit controlled errors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from embodichain.data_pipeline.real2sim import (
    JointSpec,
    load_prototype,
    prepare_prototype,
)
from embodichain.toolkits.graspkit.box_region import (
    BoxSideGraspRegionCfg,
    project_box_side_grasp,
)
from examples.expansion.trajectory.real2sim_carton.prepare import (
    DT,
    digest,
    write_json,
    write_yaml,
)

__all__: list[str] = []


def prepare(prepared: Path, capture: Path, output: Path) -> None:
    report = json.loads((capture / "report.json").read_text())
    if not report["baseline_verified"] or not report["source_artifacts_complete"]:
        raise ValueError("independent synthetic capture did not pass native execution")
    qualification = json.loads((prepared / "qualification.json").read_text())
    if not qualification["admitted"]:
        raise ValueError("the shared physical scene has not been qualified")
    output.mkdir(parents=True, exist_ok=False)
    joints = tuple(
        JointSpec(**item) for item in json.loads((prepared / "joints.json").read_text())
    )
    integration = yaml.safe_load((prepared / "integration.measured.yaml").read_text())
    revisions = [digest(capture / f"prototypes/{i}/prototype.json") for i in range(3)]
    revision = hashlib.sha256(
        json.dumps(
            dict(
                native=revisions,
                bias=[0.006, -0.004, 0.003],
                yaw_deg=3.0,
                free_drift_m=0.003,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for i, binding in enumerate(
        integration["runtime_services"]["nominal_motion_sources"]
    ):
        prototype = load_prototype(capture / f"prototypes/{i}/prototype.json")
        source = prototype.unit.source
        if (
            source.origin != "synthetic_reference"
            or source.robot_geometry_sha256
            != qualification["robot"]["robot_geometry_sha256"]
        ):
            raise ValueError("synthetic origin or frozen physical model differs")
        poses = np.array(prototype.unit.world_tcp, copy=True)
        poses[:, :3, 3] += [0.006, -0.004, 0.003]
        poses[:, :3, :3] = (
            Rotation.from_euler("z", 3, degrees=True).as_matrix() @ poses[:, :3, :3]
        )
        for phase in prototype.unit.phases:
            if phase.kind == "free":
                progress = np.linspace(0, 1, phase.stop - phase.start)
                poses[phase.start : phase.stop, 1, 3] += (
                    0.003 * np.sin(np.pi * progress) ** 2
                )
        corrupted = replace(
            source,
            source_id="synthetic_panda_carton_controlled_error",
            source_revision=revision,
            eef_poses=np.column_stack(
                (poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat())
            ),
            observed_link_to_tcp=np.eye(4),
            world_from_pose_frame=np.eye(4),
            pose_origin="transformed",
            annotation_revision="carton-bias-drift-v1",
        )
        if i == 0:
            calibration = np.array(
                integration["profile"]["action_options"]["pick"]["grasp_frame_to_eef"]
            ).reshape(4, 4)
            hint = (
                np.linalg.inv(source.object_pose)
                @ poses[prototype.unit.phases[2].start]
                @ np.linalg.inv(calibration)
            )
            region = BoxSideGraspRegionCfg(
                longitudinal_margin_m=0.010, grasp_height_m=-0.005
            )
            grasp = (
                project_box_side_grasp(
                    hint, qualification["semantic_geometry"]["object_size_m"], region
                )
                @ calibration
            )
            integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"] = (
                grasp.reshape(-1).tolist()
            )
        manifest = prepare_prototype(
            corrupted,
            joints=joints,
            control_dt=DT,
            directory=output / f"prototypes/{i}",
        )
        binding.update(
            prototype_manifest=str(manifest),
            prototype_sha256=digest(manifest),
            refinement=dict(
                max_translation_m=0.02,
                max_rotation_rad=float(np.deg2rad(10)),
                project_pick_approach=True,
                project_pick_lift=True,
                project_free_boundaries=True,
                smoothing_weight=0.2,
                seed=7,
            ),
        )
    write_yaml(output / "integration.yaml", integration)
    task = yaml.safe_load((prepared / "task.measured.yaml").read_text())
    task["environment"]["component"] = str(prepared / "environment.yaml")
    task["task_program"].update(
        program=str(prepared / "program.yaml"),
        integration="integration.yaml",
        execution_policy=str(prepared / "execution_policy.yaml"),
    )
    write_yaml(output / "task.yaml", task)
    profile = yaml.safe_load((prepared / "profile.measured.yaml").read_text())
    profile["source_artifacts"] = {}
    profile["retime_phase_allowlist"] = {
        "pick_up": ["approach"],
        "move_held_object": ["transport"],
        "place": ["approach", "retract"],
    }
    write_yaml(output / "profile.yaml", profile)
    write_yaml(
        output / "profile.single.yaml",
        {**profile, "layouts": profile["layouts"][:1], "recipes": ["nominal"]},
    )
    write_json(
        output / "source.lock.json",
        dict(
            schema="carton-independent-synthetic/v1",
            origin="synthetic_reference",
            source_revision=revision,
            native_prototypes=revisions,
            native_capture=dict(
                path=str(capture),
                report_sha256=digest(capture / "report.json"),
                measured_rollout_sha256=digest(capture / "actual_rollout.npz"),
            ),
            injection=dict(
                translation_m=[0.006, -0.004, 0.003],
                yaw_deg=3.0,
                free_phase_drift_m=0.003,
            ),
            real_arrays_used=False,
            oracle_exposed_to_refiner=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prepared.resolve(), args.capture.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
