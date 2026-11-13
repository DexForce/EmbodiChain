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

"""Review L-03 model preflights without importing a simulation runtime.

Run with --root the completed batch, --manifest the common inputs, and --output.
All metrics are model/state readback differences, not loading performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

__all__ = ["main"]


def main() -> None:
    """Match links/joints by name and compare inertia tensors in link frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", nargs="+")
    parser.add_argument("--overlap-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(args.manifest.read_text())["robots"]
    if args.robots:
        inputs = {name: inputs[name] for name in args.robots}
    shutil.copy2(__file__, args.output / "analysis-source.py")
    rows = []
    details = []
    for robot, cfg in inputs.items():
        stage = Usd.Stage.Open(cfg["usd_path"])
        bounded = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
                limits = [
                    prim.GetAttribute("physics:" + k).Get()
                    for k in ("lowerLimit", "upperLimit")
                ]
                if np.isfinite(limits).all():
                    bounded.add(prim.GetName())
        for count in (1, 4):
            pa = args.root / f"{robot}_{count:04d}_isaaclab"
            pb = args.root / f"{robot}_{count:04d}_embodichain"
            a = json.loads((pa / "run.json").read_text())
            b = json.loads((pb / "run.json").read_text())
            assert a["status"] == b["status"] == "completed"
            assert a["quaternion_order"] == b["quaternion_order"] == "xyzw"
            assert set(a["link_names"]) == set(b["link_names"])
            assert set(a["joint_names"]) == set(b["joint_names"])
            ma, mb = np.load(pa / "model.npz"), np.load(pb / "model.npz")
            sa, sb = np.load(pa / "states.npz"), np.load(pb / "states.npz")
            links = [b["link_names"].index(k) for k in a["link_names"]]
            joints = [b["joint_names"].index(k) for k in a["joint_names"]]
            finite_ids = [i for i, k in enumerate(a["joint_names"]) if k in bounded]
            rotations = (
                Rotation.from_quat(mb["com_pose"][:, links, 3:].reshape(-1, 4))
                .as_matrix()
                .reshape(count, len(links), 3, 3)
            )
            inertia_b = (
                rotations
                @ (np.eye(3)[None, None] * mb["inertia_principal"][:, links, None, :])
                @ rotations.swapaxes(-1, -2)
            )
            inertia_a = ma["inertia_matrix"].reshape(inertia_b.shape)
            qa = Rotation.from_quat(sa["first_link_pose"][..., 3:].reshape(-1, 4))
            qb = Rotation.from_quat(sb["first_link_pose"][:, links, 3:].reshape(-1, 4))
            initial_root_delta = (
                sa["initial_root_pose"][..., :3] - sb["initial_root_pose"][..., :3]
            )
            delta_link = (
                sa["first_link_pose"][..., :3] - sb["first_link_pose"][:, links, :3]
            )
            row = {
                "robot": robot,
                "num_envs": count,
                "joints_per_env": len(joints),
                "bodies_per_env": len(links),
                "max_mass_abs_kg": float(
                    np.max(np.abs(ma["mass"] - mb["mass"][:, links]))
                ),
                "max_inertia_component_abs_kgm2": float(
                    np.max(np.abs(inertia_a - inertia_b))
                ),
                "max_com_position_component_abs_m": float(
                    np.max(
                        np.abs(ma["com_pose"][..., :3] - mb["com_pose"][:, links, :3])
                    )
                ),
                "max_bounded_joint_limit_abs": float(
                    np.max(
                        np.abs(
                            ma["joint_limits"][:, finite_ids]
                            - mb["joint_limits"][:, joints][:, finite_ids]
                        )
                    )
                ),
                "max_initial_root_position_distance_m": float(
                    np.linalg.norm(initial_root_delta, axis=-1).max()
                ),
                "max_first_link_position_distance_m": float(
                    np.linalg.norm(delta_link, axis=-1).max()
                ),
                "max_first_link_rotation_difference_deg": float(
                    np.degrees((qa.inv() * qb).magnitude()).max()
                ),
                "max_initial_qpos_abs": float(
                    np.abs(sa["initial_qpos"] - sb["initial_qpos"][:, joints]).max()
                ),
                "max_first_qpos_abs": float(
                    np.abs(sa["first_qpos"] - sb["first_qpos"][:, joints]).max()
                ),
                "isaaclab_max_first_step_translation_m": a[
                    "max_link_translation_change_m"
                ],
                "embodichain_max_first_step_translation_m": b[
                    "max_link_translation_change_m"
                ],
            }
            if "native_shapes_per_articulation" in a:
                source_colliders = json.loads(
                    (pa / "source-colliders.json").read_text()
                )
                enabled = [x for x in source_colliders if x["enabled"]]
                native_counts = [
                    int((mb["native_shape_world"] == i).sum()) for i in range(count)
                ]
                assert native_counts == [len(enabled)] * count
                assert a["native_shapes_per_articulation"] == len(enabled)
                row.update(
                    source_colliders_per_env=len(enabled),
                    isaac_shapes_per_env=a["native_shapes_per_articulation"],
                    embodichain_shapes_per_env=native_counts[0],
                )
            if args.overlap_root and count == 4:
                for platform in ("isaaclab", "embodichain"):
                    reference_path = args.root / f"{robot}_0004_{platform}"
                    overlap_path = args.overlap_root / f"{robot}_0004_{platform}"
                    overlap_meta = json.loads((overlap_path / "run.json").read_text())
                    assert overlap_meta["status"] == "completed"
                    assert overlap_meta["overlap_isolation_probe"]
                    assert np.all(
                        np.load(overlap_path / "model.npz")["env_origins"] == 0
                    )
                    reference = np.load(reference_path / "initialization-trace.npz")
                    overlap = np.load(overlap_path / "initialization-trace.npz")
                    steps = [
                        s for s in overlap_meta["trace_stages"] if s.startswith("step_")
                    ]
                    positions = np.stack(
                        [overlap[s + "_link_pose"][..., :3] for s in steps]
                    )
                    reference_positions = np.stack(
                        [reference[s + "_link_pose"][..., :3] for s in steps]
                    )
                    assert np.isfinite(positions).all()
                    row[platform + "_overlap_vs_separated_link_max_m"] = float(
                        np.linalg.norm(positions - reference_positions, axis=-1).max()
                    )
                    row[platform + "_overlap_qpos_max_difference"] = float(
                        max(
                            np.abs(overlap[s + "_qpos"] - reference[s + "_qpos"]).max()
                            for s in steps
                        )
                    )
                    row[platform + "_overlap_trajectory_steps"] = len(steps)
            rows.append(row)
            details.append(
                {
                    **row,
                    "isaaclab": str(pa.resolve()),
                    "embodichain": str(pb.resolve()),
                    "initial_root_delta_m": initial_root_delta.tolist(),
                    "isaaclab_initial_root_pose": sa["initial_root_pose"].tolist(),
                    "embodichain_initial_root_pose": sb["initial_root_pose"].tolist(),
                    "joint_names": a["joint_names"],
                    "link_names": a["link_names"],
                    "unbounded_source_joints": {
                        name: {
                            "isaaclab_raw": ma["joint_limits"][:, i].tolist(),
                            "embodichain_raw": mb["joint_limits"][
                                :, joints[i]
                            ].tolist(),
                        }
                        for i, name in enumerate(a["joint_names"])
                        if name not in bounded
                    },
                    "vbd_active": b["vbd_active"],
                    "articulation_solver": b["articulation_solver"],
                }
            )
    with (args.output / "model-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "model-comparison.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
