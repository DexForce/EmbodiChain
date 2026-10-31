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

"""Export deterministic corrupted synthetic references and a normal run-task deployment.

Run: python examples/expansion/trajectory/real2sim/prepare_refinement.py --output /tmp/refinement --error bias --refine
The canonical cube grasp is authored from the fixture's TCP calibration; the
refiner never reads clean reference arrays or benchmark oracle output.
"""

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

__all__: list[str] = []


def main() -> None:
    """Prepare immutable input bundles without constructing a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--error", choices=("clean", "bias", "drift", "over_budget"), default="bias"
    )
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--repeated", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    layout = json.loads((root / "joints.ur5.json").read_text())
    joints = tuple(JointSpec(**item) for item in layout)
    integration = yaml.safe_load(
        (
            root
            / (
                "integration.repeated.yaml"
                if args.repeated
                else "integration.recorded.yaml"
            )
        ).read_text()
    )
    # The physical fixture is a 4 cm cube. Its authored top grasp has opening
    # axis +object Y, approach -object Z and a -1.7 mm calibrated TCP offset.
    grasp = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, -0.0017], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"] = (
        grasp.reshape(-1).tolist()
    )
    for index, binding in enumerate(
        integration["runtime_services"]["nominal_motion_sources"]
    ):
        prototype = load_prototype(root / binding["prototype_manifest"])
        source = prototype.unit.source
        poses = np.array(prototype.unit.world_tcp, copy=True)
        if args.error in {"bias", "over_budget"}:
            offset = [0.006, -0.004, 0.003] if args.error == "bias" else [0.05, 0, 0]
            poses[:, :3, 3] += offset
            poses[:, :3, :3] = (
                Rotation.from_euler("z", 3, degrees=True).as_matrix() @ poses[:, :3, :3]
            )
        elif args.error == "drift":
            poses[:, :3, 3] += [0.003, -0.002, 0.001]
            for phase in prototype.unit.phases:
                if phase.kind == "free":
                    fraction = np.linspace(0, 1, phase.stop - phase.start)
                    poses[phase.start : phase.stop, 1, 3] += (
                        0.003 * np.sin(np.pi * fraction) ** 2
                    )
        local = (
            np.linalg.inv(source.world_from_pose_frame)
            @ poses
            @ np.linalg.inv(source.observed_link_to_tcp)
        )
        positions = local[:, :3, 3] * (1000 if source.pose_length_unit == "mm" else 1)
        quaternion = Rotation.from_matrix(local[:, :3, :3]).as_quat()
        if source.quaternion_order == "wxyz":
            quaternion = quaternion[:, [3, 0, 1, 2]]
        corrupted = replace(
            source,
            source_id="synthetic_refinement_fixture",
            source_revision=f"1-{args.error}",
            eef_poses=np.column_stack((positions, quaternion)),
            pose_origin="transformed",
            annotation_revision=f"synthetic_error_injection_v1:{args.error}",
        )
        path = prepare_prototype(
            corrupted,
            joints=joints,
            control_dt=0.04,
            directory=directory / f"prototypes/{index}",
        )
        binding.update(
            prototype_manifest=str(path),
            prototype_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if args.refine:
            binding["refinement"] = {
                "max_translation_m": 0.02,
                "max_rotation_rad": float(np.deg2rad(10)),
                "smoothing_weight": 0.2,
            }
    (directory / "integration.yaml").write_text(
        yaml.safe_dump(integration, sort_keys=False)
    )
    task = yaml.safe_load(
        (
            root
            / ("task.ur5.repeated.yaml" if args.repeated else "task.ur5.recorded.yaml")
        ).read_text()
    )
    task["id"] = "Real2SimRefinement-v0"
    task["environment"]["component"] = str(
        (root / task["environment"]["component"]).resolve()
    )
    task["embodiment"]["component"] = str(
        Path(task["embodiment"]["component"]).resolve()
    )
    task["task_program"].update(
        program=str(
            root / ("program.repeated.yaml" if args.repeated else "program.yaml")
        ),
        integration="integration.yaml",
        execution_policy=str(root / "execution_policy.yaml"),
    )
    task["collection"]["max_attempts"] = 1
    (directory / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    (directory / "fixture.json").write_text(
        json.dumps(
            {
                "schema": "real2sim-refinement-fixture/v1",
                "origin": "synthetic_reference",
                "error": args.error,
                "refinement_enabled": args.refine,
                "grasp_object_to_tcp": grasp.tolist(),
                "oracle_exposed_to_refiner": False,
            },
            indent=2,
        )
    )
    print(directory / "task.yaml")


if __name__ == "__main__":
    main()
