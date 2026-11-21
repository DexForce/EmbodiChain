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
"""Prepare source-owned Wuji assets and FK-fitted grasp candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from _asset import _compile_usd_hand
from scipy.optimize import least_squares

__all__ = ["main"]


def _scene(source: Path, destination: Path) -> None:
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent.parent / "meshes/right"))
    option = root.find("option")
    option.set("integrator", "Euler")
    option.set("timestep", str(1 / 240))
    option.set("gravity", "0 0 0")
    palm = root.find("worldbody/body")
    palm.set("pos", "0 0 .5")
    # Wuji's palmar normal is local +X; rotate it onto world +Z.
    palm.set("quat", ".7071067811865476 0 -.7071067811865476 0")
    for body in root.iter("body"):
        for index, geom in enumerate(body.findall("geom")):
            geom.set("name", body.get("name") + "_geom_" + str(index))
    obj = ET.SubElement(root.find("worldbody"), "body", name="object", pos="-.1 0 .55")
    ET.SubElement(obj, "freejoint", name="object_joint")
    ET.SubElement(
        obj,
        "geom",
        name="object",
        type="cylinder",
        size=".03 .024",
        mass=".05",
        friction="2 .005 .0001",
        rgba=".9 .15 .08 1",
        condim="6",
    )
    temporary = destination.with_suffix(".unfiltered.xml")
    ET.ElementTree(root).write(temporary)
    model = mujoco.MjModel.from_xml_path(str(temporary))
    contact = ET.SubElement(root, "contact")
    for first in range(1, model.nbody):
        if model.body(first).name == "object":
            continue
        for second in range(first + 1, model.nbody):
            if model.body(second).name == "object":
                continue
            allowed = False
            for a in range(
                int(model.body_geomadr[first]),
                int(model.body_geomadr[first] + model.body_geomnum[first]),
            ):
                for b in range(
                    int(model.body_geomadr[second]),
                    int(model.body_geomadr[second] + model.body_geomnum[second]),
                ):
                    allowed |= bool(
                        (model.geom_contype[a] & model.geom_conaffinity[b])
                        or (model.geom_contype[b] & model.geom_conaffinity[a])
                    )
            if (
                not allowed
                or model.body_parentid[first] == second
                or model.body_parentid[second] == first
            ):
                ET.SubElement(
                    contact,
                    "exclude",
                    body1=model.body(first).name,
                    body2=model.body(second).name,
                )
    ET.indent(root)
    ET.ElementTree(root).write(destination)


def main() -> None:
    """Fit candidate fingertip poses; native physical screening is separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--center-x", type=float, default=-0.09)
    parser.add_argument("--bottom-height", type=float, default=0.521)
    parser.add_argument("--center-height", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = args.output / "scene.xml"
    _scene(args.asset.resolve() / "mjcf/right.xml", scene)
    _, profile = _compile_usd_hand(scene, args.output / "compiled")
    model = mujoco.MjModel.from_xml_path(str(scene))
    profile["tip_geometry_centers"] = [
        model.geom(f"right_finger{i}_link4_geom_3").pos.tolist() for i in range(1, 6)
    ]
    (args.output / "compiled/source.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    data = mujoco.MjData(model)
    addresses = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    lower, upper = model.actuator_ctrlrange.T
    tips = [model.geom(f"right_finger{i}_link4_geom_3").id for i in range(1, 6)]
    default = np.array([0.8, 0.5, 0.7, 0.7] + [1.0, 0, 0.8, 0.3] * 4)
    directions = np.array([[1, 0.6], [-0.5, 1], [-1, 0.03], [-0.55, -0.8], [0.2, -1.0]])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    rng = np.random.default_rng(20261104)
    reports = []
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        rows, errors = [], []
        for _ in range(args.count):
            center = np.array(
                [
                    args.center_x,
                    0,
                    (
                        args.center_height
                        if args.center_height is not None
                        else args.bottom_height + 0.016 * scale
                    ),
                ]
            ) + rng.uniform([-0.015, -0.008, -0.004], [0.015, 0.008, 0.004])
            targets = np.zeros((5, 3))
            targets[:, :2] = center[:2] + directions * (0.02 * scale + 0.008)
            targets[:, 2] = center[2] + rng.uniform(-0.003, 0.003, 5)
            initial = np.clip(
                default + rng.normal(0, 0.08, 20), lower + 1e-6, upper - 1e-6
            )

            def residual(
                q: np.ndarray,
                targets: np.ndarray = targets,
                initial: np.ndarray = initial,
            ) -> np.ndarray:
                data.qpos[addresses] = q
                mujoco.mj_forward(model, data)
                return np.r_[
                    (data.geom_xpos[tips] - targets).ravel(), 0.001 * (q - initial)
                ]

            def jacobian(q: np.ndarray) -> np.ndarray:
                data.qpos[addresses] = q
                mujoco.mj_forward(model, data)
                blocks = []
                for tip in tips:
                    linear = np.empty((3, model.nv))
                    mujoco.mj_jacGeom(model, data, linear, None, tip)
                    blocks.append(linear[:, dofs])
                return np.vstack([*blocks, 0.001 * np.eye(20)])

            result = least_squares(
                residual,
                initial,
                jac=jacobian,
                bounds=(lower + 1e-6, upper - 1e-6),
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            errors.append(
                float(
                    np.max(
                        np.linalg.norm(residual(result.x)[:15].reshape(5, 3), axis=-1)
                    )
                )
            )
            # Begin near the contact surface; a separate bounded controller
            # preload supplies grip force without resetting into deep overlap.
            targets[:, :2] = center[:2] + directions * (0.02 * scale + 0.002)
            preload = least_squares(
                residual,
                result.x,
                jac=jacobian,
                bounds=(lower + 1e-6, upper - 1e-6),
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            rows.append(np.r_[result.x, center, [1, 0, 0, 0], preload.x])
        path = args.output / f"candidates_{scale:g}.npy"
        np.save(path, np.asarray(rows, dtype=np.float32))
        reports.append(
            {
                "scale": scale,
                "count": len(rows),
                "median_max_tip_error_m": float(np.median(errors)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    (args.output / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "method": "FK-only fingertip fitting; no successful physical grasp claim until native screening",
                "source_asset": str(args.asset.resolve()),
                "source_xml_sha256": hashlib.sha256(
                    (args.asset / "mjcf/right.xml").read_bytes()
                ).hexdigest(),
                "scales": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
