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
"""Validate P-03 geometry, transient-event detection, inputs and model readback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .p03_metrics import convex_gap, metrics, relative_pose


def main() -> None:
    root = Path("outputs/benchmarks/P-03")
    result = {}
    box = np.array(
        [[a, b, c] for a in [-0.5, 0.5] for b in [-0.5, 0.5] for c in [-0.5, 0.5]]
    )
    p = np.array([[0, 0, 0, 0, 0, 0, 1.0]])
    q = p.copy()
    q[0, 0] = 1.2
    assert np.allclose(convex_gap(box, box, p, q), 0.2, atol=1e-8)
    q[0, 0] = 0.9
    assert np.allclose(convex_gap(box, box, p, q), -0.1, atol=1e-8)
    result["known_box_gap_and_penetration"] = (
        "verified against 0.2 m gap and 0.1 m penetration"
    )
    rng = np.random.default_rng(19)
    tcp = np.c_[
        rng.normal(size=(50, 3)), Rotation.random(50, random_state=rng).as_quat()
    ]
    # Construct objects with one fixed relative pose despite arbitrary rigid motion.
    rel = np.array([0.02, -0.01, 0.1, 0.1, 0.2, 0.3, 0.9])
    rel[3:] /= np.linalg.norm(rel[3:])
    rt = Rotation.from_quat(tcp[:, 3:])
    obj = np.c_[
        tcp[:, :3] + rt.apply(rel[:3]), (rt * Rotation.from_quat(rel[3:])).as_quat()
    ]
    measured = relative_pose(tcp, obj)
    assert np.max(abs(measured[:, :3] - rel[:3])) < 1e-12
    obj[::2, 3:] *= -1
    flipped = relative_pose(tcp, obj)
    assert np.max(abs(flipped[:, :3] - measured[:, :3])) < 1e-12
    assert np.allclose(abs(np.sum(flipped[:, 3:] * measured[:, 3:], axis=1)), 1)
    result["rigid_motion_and_quaternion_sign"] = (
        "verified: no artificial slip from common motion or quaternion sign"
    )
    fingers = dict(np.load(root / "prepared/collision-v1/fingers.npz"))
    e = next((root / "runs").glob("*quality_v3_embodichain_cube*"))
    i = next((root / "runs").glob("*quality_v3_isaaclab_cube*"))
    x = np.load(e / "episode_00_state.npy")[:, 0]
    t = np.load(e / "episode_00_time.npy")
    assert metrics(x, t, "cube", fingers)[0]["cycle_completed"]
    dropped = x.copy()
    dropped[(t >= 4.9) & (t < 5), 41] = 0.775
    events = metrics(dropped, t, "cube", fingers)[0]
    assert events["support_recontact_3_7"] and not events["cycle_completed"]
    assert events["first_support_recontact_3_7_s"] == 4.9
    deep = x.copy()
    deep[(t >= 4.9) & (t < 5), 41] = 0.73
    assert not metrics(deep, t, "cube", fingers)[0]["cycle_completed"]
    result["transient_drop_and_penetration"] = (
        "verified: a drop at 4.9 s followed by recovery remains incomplete; below-table excursion retained"
    )
    E = json.loads((e / "run.json").read_text())["model_readback"]
    I = json.loads((i / "run.json").read_text())["model_readback"]
    mass, com, inertia = [], [], []
    for name in E["link_names"]:
        a, b = E["link_names"].index(name), I["link_names"].index(name)
        r = Rotation.from_quat(E["robot_com"][0][a][3:]).as_matrix()
        ie = r @ np.diag(E["robot_inertia"][0][a]) @ r.T
        ii = np.array(I["robot_inertia"][0][b]).reshape(3, 3)
        inertia.append(float(abs(ie - ii).max()))
        mass.append(abs(E["robot_mass"][0][a] - I["robot_mass"][0][b]))
        com.append(
            float(
                abs(
                    np.array(E["robot_com"][0][a][:3])
                    - np.array(I["robot_com"][0][b][:3])
                ).max()
            )
        )
    result["model_readback"] = dict(
        source_runs=[e.name, i.name],
        max_link_mass_difference_kg=max(mass),
        max_link_com_difference_m=max(com),
        max_link_inertia_tensor_component_difference_kg_m2=max(inertia),
    )
    regenerated = root / "prepared/rebuild-20261023"
    checks = {}
    for name, old in [
        ("commands.npz", root / "prepared/grasp-v1/commands.npz"),
        ("fingers.npz", root / "prepared/collision-v1/fingers.npz"),
    ]:
        a, b = dict(np.load(old)), dict(np.load(regenerated / name))
        assert a.keys() == b.keys()
        differences = {k: float(np.max(abs(a[k] - b[k]))) for k in a}
        assert max(differences.values()) < 1e-12, differences
        checks[name] = differences
    result["offline_input_rebuild"] = checks
    runtime = root / "runtime/p03-frozen-v1"
    manifest = json.loads((runtime / "manifest.json").read_text())
    mismatches = [
        f["path"]
        for f in manifest["files"]
        if hashlib.sha256((runtime / f["path"]).read_bytes()).hexdigest() != f["sha256"]
    ]
    assert not mismatches, mismatches
    result["frozen_runtime_verified_files"] = len(manifest["files"])
    out = root / "analysis/grasp-v1/validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
