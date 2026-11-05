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

"""Independent offline checks of L-02 inputs, readbacks and measurement windows."""

from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main() -> None:
    root = Path("outputs/benchmarks/L-02")
    assets = root / "prepared/assets-v1"
    manifest = json.loads((assets / "manifest.json").read_text())
    rows = json.loads((root / "analysis/runs.json").read_text())
    plan = json.loads((root / "plans/main_v2.json").read_text())
    matrix_complete = all(r["status"] in ["completed", "failed"] for r in plan)
    expected = {}
    for platform in ["embodichain", "isaaclab"]:
        for shape in ["cube", "sphere"]:
            for count in [1, 100, 1000]:
                for mode in (["S"] if count == 1 else ["S", "U"]):
                    for cache in ["C", "H"]:
                        key = (platform, shape, count, mode, cache)
                        expected[key] = (
                            10 if (shape, count, mode) == ("cube", 1000, "S") else 3
                        )
    observed = Counter(
        tuple(
            r[k]
            for k in ["platform", "shape", "object_count", "asset_mode", "cache_mode"]
        )
        for r in rows
        if r["purpose"] == "main" and r["protocol_version"] == 2
    )
    assert all(k in expected and n <= expected[k] for k, n in observed.items())
    if matrix_complete:
        assert (
            dict(observed) == expected
        ), "The full planned matrix must have original records"
    source = []
    for shape in ["cube", "sphere"]:
        records = [r for r in manifest["files"] if r["shape"] == shape]
        u = [r for r in records if r["mode"] == "U"]
        assert len({r["sha256"] for r in u}) == len(u)
        assert len({r["geometry_sha256"] for r in u}) == len(u)
        assert len({tuple(r["color"]) for r in u}) == len(u)
        assert all(
            0.9 <= r["scale"] <= 1.1
            and r["triangles"] == (12 if shape == "cube" else 528)
            for r in records
        )
        for r in records:
            f = assets / r["file"]
            assert f.stat().st_size == r["bytes"]
            assert hashlib.sha256(f.read_bytes()).hexdigest() == r["sha256"]
        source.append(
            dict(
                shape=shape,
                files=len(records),
                different_U_geometries=len(u),
                different_U_colors=len(u),
                verified_hashes=len(records),
            )
        )
    checks = []
    cameras = {}
    intrinsics = {}
    for row in rows:
        if (
            row["purpose"] not in ["main", "capacity", "media"]
            or row["status"] != "completed"
        ):
            continue
        folder = root / "runs" / row["run_id"]
        cfg = json.loads((folder / "config.json").read_text())
        meta = json.loads((folder / "run.json").read_text())
        a = json.loads((folder / "model-audit.json").read_text())
        state = np.load(folder / "first_state.npy")
        m = np.load(folder / "model.npz")
        assert state.shape == (cfg["object_count"], 13) and np.isfinite(state).all()
        assert len(a["meshes"]) == cfg["object_count"] and all(
            x["triangles"] == cfg["mesh_triangles"][cfg["shape"]] for x in a["meshes"]
        )
        if cfg["platform"] == "embodichain":
            native = a["native"]
            kind = 7 if cfg["shape"] == "cube" else 3  # Installed Newton 1.6 GeoType.
            types = np.asarray(native["shape_type"])
            selected = types == kind
            assert (types == 1).sum() == 1 and selected.sum() == cfg["object_count"]
            owners = np.asarray(native["shape_body"])[selected]
            np.testing.assert_array_equal(
                np.sort(owners), np.arange(cfg["object_count"])
            )
            scales = np.asarray(native["shape_scale"])[selected][np.argsort(owners)]
            radius = np.array([0.025 * o["scale"] for o in cfg["objects"]])
            expected_scale = (
                np.repeat(radius[:, None], 3, axis=1)
                if cfg["shape"] == "cube"
                else np.column_stack([radius, np.zeros((len(radius), 2))])
            )
            np.testing.assert_allclose(
                scales, expected_scale, rtol=4 * np.finfo(np.float32).eps, atol=0
            )
        else:
            kinds = Counter(x["type"] for x in a["collisions"])
            assert kinds == {
                "Plane": 1,
                "Cube" if cfg["shape"] == "cube" else "Sphere": cfg["object_count"],
            }
        colors = np.array([x["color_linear"][:3] for x in a["materials"]])
        expect = np.array([o["color"] for o in cfg["objects"]])
        # Bound only conversion/readback roundoff against explicitly authored values.
        np.testing.assert_allclose(
            colors,
            expect,
            rtol=4 * np.finfo(np.float32).eps,
            atol=4 * np.finfo(np.float32).eps,
        )
        np.testing.assert_allclose(
            m["mass"], 0.1, rtol=4 * np.finfo(np.float32).eps, atol=0
        )
        moment = np.array(
            [
                0.1 * (0.05 * o["scale"]) ** 2 / (6 if cfg["shape"] == "cube" else 10)
                for o in cfg["objects"]
            ]
        )
        inertia = m["inertia"]
        inertia = (
            inertia if inertia.ndim == 2 else np.diagonal(inertia, axis1=-2, axis2=-1)
        )
        np.testing.assert_allclose(
            inertia,
            np.repeat(moment[:, None], 3, axis=1),
            rtol=4 * np.finfo(np.float32).eps,
            atol=0,
        )
        t = meta["stage_times_s"]
        assert (
            0
            < t["engine_initialized"]
            <= t["declared"]
            <= t["prepared"]
            <= t["ready"]
            <= t["frame"]
        )
        im = np.array(Image.open(folder / "first_frame.png"))
        assert im.shape == (480, 640, 3)
        key = (cfg["object_count"] > 1000, cfg["platform"])
        cameras.setdefault(key, np.array(a["camera_world_opengl"]))
        intrinsics.setdefault(key, np.array(a["intrinsics"]))
        fx, fy, cx, cy = cfg["camera"]["intrinsics"]
        np.testing.assert_allclose(
            a["intrinsics"],
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            rtol=4 * np.finfo(np.float32).eps,
            atol=0,
        )
        np.testing.assert_allclose(
            np.array(a["camera_world_opengl"]), cameras[key], atol=1e-6, rtol=0
        )
        checks.append(
            dict(
                run_id=row["run_id"],
                objects=len(state),
                stage_order=True,
                shape_and_material_readback=True,
                collision_check_scope=(
                    "Newton shape owners, types and scales"
                    if cfg["platform"] == "embodichain"
                    else "USD collision counts and types"
                ),
                measurement_eligible=row["timing_eligible"],
            )
        )
    pair_diff = {}
    intrinsic_diff = {}
    for capacity in [False, True]:
        if (capacity, "embodichain") in cameras and (capacity, "isaaclab") in cameras:
            value = float(
                np.max(
                    abs(
                        cameras[capacity, "embodichain"] - cameras[capacity, "isaaclab"]
                    )
                )
            )
            pair_diff["capacity" if capacity else "main"] = value
            intrinsic_diff["capacity" if capacity else "main"] = float(
                np.max(
                    abs(
                        intrinsics[capacity, "embodichain"]
                        - intrinsics[capacity, "isaaclab"]
                    )
                )
            )
            assert value < 1e-6
    result = {
        "input_checks": source,
        "main_matrix_complete": matrix_complete,
        "main_expected_groups": len(expected),
        "main_expected_attempts": sum(expected.values()),
        "run_checks": checks,
        "camera_pair_max_component_difference": pair_diff,
        "intrinsics_pair_max_component_difference_px": intrinsic_diff,
        "main_attempts": sum(r["purpose"] == "main" for r in rows),
        "main_completed": sum(
            r["purpose"] == "main" and r["status"] == "completed" for r in rows
        ),
        "main_eligible": sum(
            r["purpose"] == "main" and r["timing_eligible"] for r in rows
        ),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "note": "Readback checks use float32-roundoff bounds for matching authored inputs, not an empirical simulation quality pass/fail threshold. Native images visually inspected separately.",
    }
    (root / "analysis/validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "run_checks"})


if __name__ == "__main__":
    main()
