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
"""Freeze the approved native M02 scene as inputs for EmbodiChain and Isaac Lab.

Run once after the user approves the DexSim preview:
    python3 scripts/benchmark/tech_reports/physics/experiments/dexuni/m02_cloth_single_fold/tools/export_inputs.py

The DexSim script already wrote inputs/joint_commands.npz and cloth_mesh.npz.
This writes inputs/m02_config.json (scene, cloth, contact and solver values of
the approved run), copies the prepared W1 USD from M01 (same URDF, verified by
hash) and records the approval in manifest.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = (
    ROOT.parents[2] / "outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold"
)
M01_INPUTS = (
    ROOT.parents[2] / "outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/inputs"
)
APPROVAL = (
    "可以，移植到 EmbodiChain（同样使用 surface-fast）和 Isaac Lab（使用自带 VBD）"
)
__all__ = ["main"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write the frozen config and record the scene approval."""
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed":
        raise ValueError("The native preview did not complete")
    args, scene = manifest["arguments"], manifest["scene"]
    effective = manifest["effective"]
    urdf = OUTPUT / "inputs/w1_v030/robot.urdf"
    # Values below are the ones m02.py passed to DexSim for the approved run.
    config = {
        "task": "M02",
        "dt": effective["dt"],
        "steps": int(manifest["frames"]),
        "substeps": effective["num_substeps"],
        "use_cuda_graph": args["no_cuda_graph"] == "False",
        "coupling": manifest["coupling"],
        "joint_mode": manifest["joint_mode"],
        "contact_mode": scene["contact_mode"],
        "vbd_preset": scene["vbd_preset"],
        "native_vbd_iterations": effective["vbd"]["iterations"],
        "friction_epsilon": 1e-4,
        "self_contact": manifest["self_contact"],
        "soft_contact_margin": 0.003,
        "shape_ke": 2.0e4,
        "shape_kd": 100.0,
        "shape_gap": 0.002,
        "shape_margin": 0.001,  # NewtonCollisionDesc default
        "robot_mu": float(scene["robot_mu"]),
        "static_mu": 0.8,
        "cloth": {
            "surface_density": 0.002,
            "tri_ke": 1000.0,
            "tri_ka": 1000.0,
            "tri_kd": 1e-5,
            "tri_drag": 0.0,
            "tri_lift": 0.0,
            "edge_ke": 0.1,
            "edge_kd": 0.05,
            "particle_radius": 0.003,
            "contact_ke": 1e4,
            "contact_kd": 0.01,
            "contact_mu": 0.3,
            "color": [0.25, 0.65, 0.9],
        },
        "camera": {**manifest["camera"], "fps": 30},
        "tcp_offset": [0.0, 0.0, 0.125],
        "robot_urdf": "w1_v030/robot.urdf",
        "native_preview_sha256": _sha(OUTPUT / "preview/dexsim.mp4"),
        "input_sha256": {
            name: _sha(OUTPUT / "inputs" / name)
            for name in ("joint_commands.npz", "cloth_mesh.npz")
        },
    }
    (OUTPUT / "inputs/m02_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Isaac Lab needs the converted W1 USD; M01 holds one made from the same URDF.
    if _sha(M01_INPUTS / "w1_v030/robot.urdf") != _sha(urdf):
        raise ValueError("M01 W1 URDF differs; reconvert the USD for M02")
    usd = OUTPUT / "inputs/w1_usd"
    if not usd.exists():
        shutil.copytree(M01_INPUTS / "w1_usd", usd)
    (OUTPUT / "inputs/w1_usd_path.json").write_text(
        json.dumps(
            {
                "usd_path": str(usd / "robot/robot.usda"),
                "urdf": str(urdf),
                "source": "copied from M01 inputs/w1_usd (same URDF sha256)",
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
    print(f"Exported M02 config ({config['steps']} steps); approval recorded.")


if __name__ == "__main__":
    main()
