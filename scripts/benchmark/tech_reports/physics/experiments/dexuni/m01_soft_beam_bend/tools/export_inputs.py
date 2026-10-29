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
"""Freeze the approved native M01 scene as inputs for EmbodiChain and Isaac Lab.

Run once after the user approves the DexSim preview:
    /home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/m01_soft_beam_bend/tools/export_inputs.py

Writes inputs/m01_config.json and inputs/joint_commands.npz (the DexSim joint
targets replayed by the other platforms), copies the prepared W1 USD from S04
(same URDF, verified by hash) and records the approval in manifest.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_common import (
    OUTPUT,
    TABLE_Z,
)

S04_INPUTS = OUTPUT.parent / "s04_rigid_soft_squeeze_grasp/inputs"
APPROVAL = "ok，继续做embodichain和issacLab的场景吧，做完我看没问题再测指标"
__all__ = ["main"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joint_names(urdf: Path) -> list[str]:
    """Joint order of the DexSim target rows (the IK model's actuated joints)."""
    from dexsim.kit.ik import KinematicModel

    model = KinematicModel.from_urdf(
        urdf, floating=False, hide_visuals=True, collapse_fixed_joints=False
    )
    starts = model.joint_q_start.numpy()
    return [
        label.rsplit("/", 1)[-1]
        for j, label in enumerate(model.joint_label)
        if starts[j + 1] > starts[j]
    ]


def main() -> None:
    """Write the frozen config/commands and record the scene approval."""
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed":
        raise ValueError("The native preview did not complete")
    args = manifest["arguments"]
    scene = manifest["scene"]
    urdf = OUTPUT / "inputs/w1_v030/robot.urdf"
    states = np.load(OUTPUT / "preview/states.npz")
    names = joint_names(urdf)
    targets = states["target_q"].astype(np.float32)  # as recorded by DexSim
    time = states["time"].astype(np.float64)
    steps = int(manifest["frames"])
    if targets.shape != (steps + 1, len(names)):
        raise ValueError(
            f"Unexpected target layout {targets.shape} for {len(names)} joints"
        )
    np.savez_compressed(
        OUTPUT / "inputs/joint_commands.npz",
        joint_names=np.asarray(names),
        targets=targets,
        time=np.arange(steps + 1) / 60.0,
        recorded_time=time,
    )
    beam = np.asarray(scene["beam_size_xyz"], dtype=float)
    spawn_z = TABLE_Z + scene["spawn_gap"] + beam[2] / 2
    carry_x = scene["carry_x"]
    x = scene["beam_center_xy"][0]
    config = {
        "task": "M01",
        "dt": 1 / 60,
        "steps": steps,
        "substeps": int(args["substeps"]),
        "vbd_iterations": int(args["iterations"]),
        "use_cuda_graph": args["no_cuda_graph"] == "False",
        "coupling": "two_way",
        "arm_kp": float(args["arm_kp"]),
        "arm_kd": float(args["arm_kd"]),
        "finger_kp": float(args["finger_kp"]),
        "finger_kd": float(args["finger_kd"]),
        "finger_armature": float(args["finger_armature"]),
        "finger_passive_damping": float(args["finger_damping"]),
        "gravity_compensation": float(args["gravity_compensation"]),
        "proxy_mass_scale": float(args["proxy_mass_scale"]),
        "proxy_relaxation": float(args["proxy_relaxation"]),
        "shape_ke": 2.0e4,
        "shape_kd": 100.0,
        "shape_gap": 0.002,
        "shape_margin": 0.001,
        "robot_mu": 1.0,
        "static_mu": 0.8,
        "particle_ke": 2.0e4,
        "particle_kd": 10.0,
        "particle_mu": 1.0,
        "particle_radius": 0.003,
        "soft_mu": float(args["k_mu"]),
        "soft_lambda": float(args["k_lambda"]),
        "soft_damping": float(args["k_damp"]),
        "soft_density": float(args["density"]),
        "soft_center": [x, 0.0, spawn_z],
        "beam_size_xyz": beam.tolist(),
        "marks": [
            {
                "name": "start_mark",
                "center": [x, 0.0, TABLE_Z - 0.0045],
                "size": [0.07, 0.50, 0.01],
                "color": [0.35, 0.55, 0.35],
            },
            {
                "name": "target_mark",
                "center": [x + carry_x, 0.0, TABLE_Z - 0.0045],
                "size": [0.07, 0.50, 0.01],
                "color": [0.55, 0.40, 0.30],
            },
        ],
        "camera": {
            "eye": manifest["camera"]["eye"],
            "target": manifest["camera"]["target"],
            "resolution": manifest["camera"]["resolution"],
            "fps": 30,
            "hfov": 60.0,
        },
        "robot_urdf": "w1_v030/robot.urdf",
        "native_preview_sha256": _sha(OUTPUT / "preview/dexsim.mp4"),
    }
    (OUTPUT / "inputs/m01_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Isaac Lab needs the converted W1 USD; S04 already prepared it from the
    # same URDF (with GLB visuals restored), so copy it instead of reconverting.
    if _sha(S04_INPUTS / "w1_v030/robot.urdf") != _sha(urdf):
        raise ValueError("S04 W1 URDF differs; reconvert the USD for M01")
    usd = OUTPUT / "inputs/w1_usd"
    if not usd.exists():
        shutil.copytree(S04_INPUTS / "w1_usd", usd)
    (OUTPUT / "inputs/w1_usd_path.json").write_text(
        json.dumps(
            {
                "usd_path": str(usd / "robot/robot.usda"),
                "urdf": str(urdf),
                "source": "copied from S04 inputs/w1_usd (same URDF sha256)",
                "urdf_sha256": _sha(urdf),
            },
            indent=2,
        )
        + "\n"
    )
    manifest.update(
        user_scene_approved=True,
        user_approval=APPROVAL,
        approved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(names)} joints x {steps + 1} frames; approval recorded.")


if __name__ == "__main__":
    main()
