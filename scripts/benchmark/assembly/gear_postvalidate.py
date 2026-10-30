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
"""Independently inspect completed native gear artifacts without running physics.

Run: python -m scripts.benchmark.assembly.gear_postvalidate INPUT_DIRECTORY

Original benchmark results and MJB files remain untouched. This CPU-only pass
checks full inertia tensors, COM, geometry-matched contact properties and the
live arrays actually captured by the native runner. Missing live fields remain
unverified rather than being inferred from a host-model snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from scripts.benchmark.assembly.gear_reference import _load_manifest

__all__: list[str] = []

_CONTACT_FIELDS = (
    "geom_condim",
    "geom_priority",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_solmix",
    "geom_margin",
    "geom_gap",
)
_LIVE_FIELDS = (
    "body_mass",
    "body_inertia",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_solmix",
    "geom_margin",
    "geom_gap",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
)
_UNRECORDED_LIVE_FIELDS = (
    "body_ipos",
    "body_iquat",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
)


def _rotation(wxyz: np.ndarray) -> np.ndarray:
    """Convert the public MuJoCo principal-frame quaternion to a matrix."""
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()


def _tensor(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Reconstruct the full body-local tensor, including principal orientation."""
    rotation = _rotation(model.body_iquat[body])
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def _body_map(source: mujoco.MjModel, actual: mujoco.MjModel) -> dict[int, int]:
    """Resolve each physical gear by one unique imported body-name suffix."""
    result = {0: 0}
    for name in ("gear_a", "gear_b"):
        matches = [i for i in range(actual.nbody) if actual.body(i).name.endswith(name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one actual body ending in {name!r}: {matches}.")
        result[source.body(name).id] = matches[0]
    return result


def _colliders(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Select colliding source geometry while excluding independent visuals."""
    return np.flatnonzero(
        (model.geom_bodyid == body)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )


def _points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Read a mesh in its owning body's frame after MuJoCo mesh recentering."""
    kind = int(model.geom_type[geom])
    if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # Plane tangential display extents have no physical meaning.
        rotation = _rotation(model.geom_quat[geom])
        return np.asarray([model.geom_pos[geom], model.geom_pos[geom] + rotation[:, 2]])
    if kind != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise NotImplementedError(
            f"Gear post-validation does not support geom type {kind}."
        )
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    points = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    return points @ _rotation(model.geom_quat[geom]).T + model.geom_pos[geom]


def _geometry_pairs(
    source: mujoco.MjModel, actual: mujoco.MjModel, bodies: dict[int, int]
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Assign colliders by geometry, then measure bidirectional vertex error.

    The assignment uses body-local bounds rather than names, material values
    or importer order. Duplicate vertices introduced by native cooking do not
    change the bidirectional nearest-point comparison.
    """
    pairs = []
    rows = []
    for source_body, actual_body in bodies.items():
        source_geoms, actual_geoms = _colliders(source, source_body), _colliders(
            actual, actual_body
        )
        if len(source_geoms) != len(actual_geoms):
            raise ValueError(
                f"Collider count changed on body {source_body}: {len(source_geoms)} != {len(actual_geoms)}."
            )
        if not len(source_geoms):
            continue
        source_points = [_points(source, int(i)) for i in source_geoms]
        actual_points = [_points(actual, int(i)) for i in actual_geoms]
        signatures = lambda clouds: np.asarray(
            [
                np.concatenate(((p.min(0) + p.max(0)) / 2, np.ptp(p, axis=0)))
                for p in clouds
            ]
        )
        cost = cdist(signatures(source_points), signatures(actual_points))
        si, ai = linear_sum_assignment(cost)
        for i, j in zip(si, ai, strict=True):
            source_geom, actual_geom = int(source_geoms[i]), int(actual_geoms[j])
            if source.geom_type[source_geom] != actual.geom_type[actual_geom]:
                raise ValueError("Matched source/native collider types differ.")
            distances = cdist(source_points[i], actual_points[j])
            error = float(max(distances.min(0).max(), distances.min(1).max()))
            pairs.append((source_geom, actual_geom))
            rows.append(
                {
                    "source_geom": source_geom,
                    "actual_geom": actual_geom,
                    "geometry_error_m": error,
                }
            )
    if len(pairs) != len(
        np.flatnonzero((actual.geom_contype != 0) | (actual.geom_conaffinity != 0))
    ):
        raise ValueError(
            "Actual model contains colliders outside the matched gears/floor."
        )
    maximum = max((r["geometry_error_m"] for r in rows), default=0.0)
    return pairs, {"passed": maximum <= 1e-6, "maximum_error_m": maximum, "pairs": rows}


def _field_check(expected: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    """Compare float32-carried contact values or exact integer metadata."""
    expected, got = np.asarray(expected), np.asarray(got)
    integer = np.issubdtype(expected.dtype, np.integer)
    passed = (
        np.array_equal(expected, got)
        if integer
        else np.allclose(expected, got, atol=1e-8, rtol=1e-6)
    )
    return {
        "passed": bool(passed),
        "maximum_absolute_error": (
            float(np.max(np.abs(got - expected))) if expected.size else 0.0
        ),
    }


def _global_options(
    source: mujoco.MjModel, actual: mujoco.MjModel, live_options: dict[str, Any]
) -> dict[str, Any]:
    """Record global profile differences separately from representation checks."""
    names = (
        "solver",
        "integrator",
        "timestep",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "disableflags",
        "enableflags",
        "cone",
        "jacobian",
        "impratio",
        "ccd_tolerance",
        "ccd_iterations",
    )
    expected = {name: np.asarray(getattr(source.opt, name)).tolist() for name in names}
    host = {name: np.asarray(getattr(actual.opt, name)).tolist() for name in names}
    host_differences = {
        name: {"source": expected[name], "actual": host[name]}
        for name in names
        if not _field_check(np.asarray(expected[name]), np.asarray(host[name]))[
            "passed"
        ]
    }
    live_differences = {
        name: {"source": expected[name], "actual": live_options[name]}
        for name in names
        if name in live_options
        and not _field_check(
            np.asarray([expected[name]]), np.asarray(live_options[name]).reshape(-1)
        )["passed"]
    }
    flags = int(actual.opt.disableflags)
    source_flags = int(source.opt.disableflags)
    changed_bits = [
        name
        for name, bit in mujoco.mjtDisableBit.__members__.items()
        if name.startswith("mjDSBL_") and (flags ^ source_flags) & int(bit)
    ]
    multiccd = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    return {
        "source": expected,
        "actual_host": host,
        "host_differences": host_differences,
        "captured_live_differences": live_differences,
        "unrecorded_live_options": [name for name in names if name not in live_options],
        "changed_disable_bits": changed_bits,
        "source_multiccd_enabled": not bool(source_flags & multiccd),
        "actual_host_multiccd_enabled": not bool(flags & multiccd),
        "source_global_options_equivalent": bool(
            not host_differences and not live_differences
        ),
        "interpretation": "Separate native solver/contact profile; diagnostic behavior passes do not establish identical global options or dynamic-trajectory equivalence. CPU midphase disabling and explicit CG solver selection are retained differences, not hidden substitutions. MultiCCD and solver tolerance differences require disclosure; this check does not establish a causal explanation or completed parameter calibration.",
    }


def _compare_models(source: mujoco.MjModel, actual: mujoco.MjModel) -> dict[str, Any]:
    """Check full body physics and properties of geometrically matched colliders."""
    bodies = _body_map(source, actual)
    pairs, geometry = _geometry_pairs(source, actual, bodies)
    sd, ad = mujoco.MjData(source), mujoco.MjData(actual)
    mujoco.mj_kinematics(source, sd)
    mujoco.mj_kinematics(actual, ad)
    rows = []
    joint_pairs = []
    for source_body, actual_body in bodies.items():
        if source_body == 0:
            continue
        expected, got = _tensor(source, source_body), _tensor(actual, actual_body)
        inertia_error = float(
            np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-30)
        )
        com_error = float(
            np.linalg.norm(
                actual.body_ipos[actual_body] - source.body_ipos[source_body]
            )
        )
        world_com_error = float(
            np.linalg.norm(ad.xipos[actual_body] - sd.xipos[source_body])
        )
        mass_error = float(
            abs(actual.body_mass[actual_body] / source.body_mass[source_body] - 1)
        )
        sj = np.flatnonzero(source.jnt_bodyid == source_body)
        aj = np.flatnonzero(actual.jnt_bodyid == actual_body)
        if len(sj) != 1 or len(aj) != 1:
            raise ValueError("Gear comparison requires one hinge per moving body.")
        sj, aj = int(sj[0]), int(aj[0])
        source_dof, actual_dof = int(source.jnt_dofadr[sj]), int(actual.jnt_dofadr[aj])
        joint_pairs.append((source_dof, actual_dof))
        joint_errors = {
            name: float(
                abs(
                    getattr(actual, name)[actual_dof]
                    - getattr(source, name)[source_dof]
                )
            )
            for name in ("dof_damping", "dof_armature", "dof_frictionloss")
        }
        joint_layout = bool(
            source.jnt_type[sj] == actual.jnt_type[aj]
            and source.jnt_limited[sj] == actual.jnt_limited[aj]
            and np.allclose(source.jnt_axis[sj], actual.jnt_axis[aj], atol=1e-7, rtol=0)
            and np.allclose(source.jnt_pos[sj], actual.jnt_pos[aj], atol=1e-6, rtol=0)
        )
        rows.append(
            {
                "body": source.body(source_body).name,
                "source_body": source_body,
                "actual_body": actual_body,
                "mass_relative_error": mass_error,
                "full_inertia_relative_error": inertia_error,
                "source_body_inertia_tensor": expected.tolist(),
                "actual_body_inertia_tensor": got.tolist(),
                "com_error_m": com_error,
                "world_com_error_m": world_com_error,
                "joint_errors": joint_errors,
                "joint_layout_passed": joint_layout,
                "passed": bool(
                    mass_error <= 1e-5
                    and inertia_error <= 1e-5
                    and com_error <= 1e-6
                    and world_com_error <= 1e-6
                    and joint_layout
                    and joint_errors["dof_damping"] <= 1e-8
                    and joint_errors["dof_armature"] <= 1e-10
                    and joint_errors["dof_frictionloss"] <= 1e-8
                ),
            }
        )
    source_geoms, actual_geoms = np.asarray(pairs, dtype=int).T
    contacts = {
        name: _field_check(
            getattr(source, name)[source_geoms], getattr(actual, name)[actual_geoms]
        )
        for name in _CONTACT_FIELDS
    }
    eligibility = []
    for model, indices in ((source, source_geoms), (actual, actual_geoms)):
        ct, ca = model.geom_contype[indices], model.geom_conaffinity[indices]
        mask = ((ct[:, None] & ca[None, :]) != 0) | ((ca[:, None] & ct[None, :]) != 0)
        owners = model.geom_bodyid[indices]
        eligibility.append(mask & (owners[:, None] != owners[None, :]))
    filters = {
        "passed": bool(np.array_equal(*eligibility)),
        "source_eligible_pairs": int(np.count_nonzero(np.triu(eligibility[0], 1))),
        "actual_eligible_pairs": int(np.count_nonzero(np.triu(eligibility[1], 1))),
        "scope": "Collision-mask eligibility between different bodies; arena bit identities may differ.",
        "source_contype": source.geom_contype[source_geoms].tolist(),
        "actual_contype": actual.geom_contype[actual_geoms].tolist(),
        "source_conaffinity": source.geom_conaffinity[source_geoms].tolist(),
        "actual_conaffinity": actual.geom_conaffinity[actual_geoms].tolist(),
    }
    return {
        "passed": bool(
            geometry["passed"]
            and all(r["passed"] for r in rows)
            and all(r["passed"] for r in contacts.values())
            and filters["passed"]
        ),
        "body_mapping": bodies,
        "joint_pairs": joint_pairs,
        "geometry": geometry,
        "bodies": rows,
        "contacts": contacts,
        "collision_filtering": filters,
    }


def _live_check(
    source: mujoco.MjModel,
    actual: mujoco.MjModel,
    host: dict[str, Any],
    snapshot: dict[str, Any],
    spec: dict[str, Any],
    *,
    before_first_step: bool = False,
) -> dict[str, Any]:
    """Validate captured live values using host topology, without inventing fields."""
    checks = {}
    missing = []
    properties = snapshot.get("properties", {})
    pairs = host["geometry"]["pairs"]
    for name in _LIVE_FIELDS:
        if name not in properties:
            missing.append(name)
            continue
        array = np.asarray(properties[name])
        shape = getattr(actual, name).shape
        if array.shape == (1, *shape):
            array = array[0]
        if array.shape != shape:
            checks[name] = {
                "passed": False,
                "reason": f"Unexpected live shape {array.shape}, expected {shape}.",
            }
            continue
        if name.startswith("geom_"):
            expected = getattr(source, name)[[p["source_geom"] for p in pairs]]
            got = array[[p["actual_geom"] for p in pairs]]
        elif name.startswith("dof_"):
            expected = getattr(source, name)[[p[0] for p in host["joint_pairs"]]]
            got = array[[p[1] for p in host["joint_pairs"]]]
        else:
            # Full orientation/COM are absent from current live snapshots.
            actual_ids = [row["actual_body"] for row in host["bodies"]]
            expected, got = getattr(actual, name)[actual_ids], array[actual_ids]
            if name == "body_inertia":
                # Host MJB recompilation can reorder principal axes. A live
                # orientation was not captured, so only the live spectrum is
                # directly comparable. Host full tensors remain gated above.
                expected, got = np.sort(expected, axis=-1), np.sort(got, axis=-1)
        checks[name] = _field_check(expected, got)
    options = snapshot.get("options", {})
    for name, expected in (
        ("solver", 0 if spec["use_mujoco_cpu"] else 1),
        ("integrator", int(source.opt.integrator)),
        ("timestep", spec["timestep"]),
        ("iterations", spec["iterations"]),
    ):
        if name == "timestep" and before_first_step:
            continue
        if name not in options:
            missing.append(f"options.{name}")
        else:
            checks[f"options.{name}"] = _field_check(
                np.asarray([expected]), np.asarray(options[name]).reshape(-1)
            )
    return {
        "observed_checks_passed": bool(
            not missing and all(c["passed"] for c in checks.values())
        ),
        "checks": checks,
        "missing_captured_fields": missing,
        "unrecorded_live_fields": [
            name for name in _UNRECORDED_LIVE_FIELDS if name not in properties
        ],
        "full_live_physics_status": "unverified",
        "orientation_scope": "Live inertia spectrum checked; live principal orientation and COM were not recorded. Full host tensors checked separately.",
        "timestep_scope": (
            "Before first physical update; default initial timestep is recorded but not gated as a dynamics setting."
            if before_first_step
            else "Requested timestep checked after physical integration."
        ),
        "recorded_options": options,
    }


def _validate_cell(path: Path) -> dict[str, Any]:
    """Read a completed cell and produce independent additional evidence."""
    start = time.perf_counter()
    raw = path.read_bytes()
    result = json.loads(raw)
    row = {
        "cell": path.parent.name,
        "profile": result.get("profile", "unknown"),
        "original_passed": bool(result.get("passed")),
        "original_result_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if "compiled_models" not in result:
        return {
            **row,
            "observed_checks_passed": False,
            "combined_diagnostic_passed": False,
            "reason": "Original cell did not produce compiled-model snapshots.",
        }
    asset, digest = _load_manifest(Path(result["manifest"]))
    if digest != result["asset_sha256"]:
        raise ValueError("Result and manifest hashes differ.")
    source = mujoco.MjModel.from_xml_string(asset["source_mjcf"])
    stages = []
    for item in result["compiled_models"]:
        model_path = path.parent / item["file"]
        actual = mujoco.MjModel.from_binary_path(str(model_path))
        host = _compare_models(source, actual)
        stage = "initial" if item["step"] == 0 else "final"
        live = _live_check(
            source,
            actual,
            host,
            result[f"{stage}_active_properties"],
            result,
            before_first_step=item["step"] == 0,
        )
        stages.append(
            {
                "step": item["step"],
                "mjb": item["file"],
                "mjb_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "host": host,
                "live": live,
                "global_options": _global_options(
                    source, actual, live["recorded_options"]
                ),
            }
        )
    passed = bool(
        len(stages) == 2
        and all(
            s["host"]["passed"] and s["live"]["observed_checks_passed"] for s in stages
        )
    )
    return {
        **row,
        "asset_sha256": digest,
        "stages": stages,
        "observed_checks_passed": passed,
        "combined_diagnostic_passed": bool(row["original_passed"] and passed),
        "formal_certification": False,
        "full_live_physics_status": "unverified",
        "postprocess_time_ms": (time.perf_counter() - start) * 1000,
    }


def _report(
    directory: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    """Write raw evidence and exactly three benchmark-style Markdown tables."""
    (directory / "validation.json").write_text(
        json.dumps({"metadata": metadata, "cells": rows}, indent=2, allow_nan=False)
        + "\n"
    )
    lines = [
        "# Independent native gear property validation",
        "",
        json.dumps(metadata),
        "",
        "## Time & Memory",
        "",
        "| cell | profile | CPU postprocess ms | CPU peak MB | GPU memory |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['cell']} | {r['profile']} | {r.get('postprocess_time_ms', 'n/a')} | unmeasured | not used |"
        for r in rows
    ]
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| cell | original pass | observed property checks | combined diagnostic pass | full live physics |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['cell']} | {r['original_passed']} | {r['observed_checks_passed']} | {r['combined_diagnostic_passed']} | unverified |"
        for r in rows
    ]
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | profile | combined diagnostic success rate | completed cells |",
        "| --- | --- | --- | --- |",
    ]
    totals = []
    for profile in sorted({r["profile"] for r in rows}):
        selected = [r for r in rows if r["profile"] == profile]
        totals.append(
            (
                profile,
                sum(r["combined_diagnostic_passed"] for r in selected) / len(selected),
                len(selected),
            )
        )
    lines += [
        f"| {i} | {p} | {rate:.6f} | {n} |"
        for i, (p, rate, n) in enumerate(sorted(totals, key=lambda t: -t[1]), 1)
    ]
    lines += [
        "",
        "This is CPU artifact inspection, not a dynamics or throughput measurement. Both initial and final actual MJB snapshots are checked against the independently compiled source. Colliders are matched geometrically within each body before contact fields are compared. Full inertia uses the authored principal orientation, not sorted eigenvalues. COM, passive dynamics, collision-mask eligibility, condim, priority, raw contacts and collider geometry are gated. Captured live fields are checked separately. Live COM, principal orientation, masks, condim and priority were not recorded, so full live physics remains unverified. Original result files and their diagnostic statuses are unchanged. Combined success additionally requires all observed representation checks. Global solver/contact settings are recorded separately per stage in raw data. Their differences can include MultiCCD, CPU midphase disabling, explicit Warp CG selection and effective solver tolerance; inspect each profile rather than assuming the same flags across datasets. Representation checks do not retroactively fail valid engineering behavior thresholds, but identical-global-options and dynamic-trajectory equivalence are not established. No completed parameter-calibration claim or causal attribution is made. Pending cells are not passes. No formal certification is claimed.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Inspect all currently completed cells, safely alongside an active run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-cells", type=int, default=30)
    args = parser.parse_args()
    directory = args.input.resolve()
    output = (
        args.output or directory.with_name(directory.name + "_post_validation")
    ).resolve()
    if output == directory:
        parser.error(
            "Post-validation output must be separate from the original benchmark."
        )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    for path in sorted(directory.glob("cell_*/result.json")):
        try:
            row = _validate_cell(path)
        except json.JSONDecodeError:
            pending.append(path.parent.name)
            continue
        except Exception as error:
            spec_path = path.parent / "spec.json"
            spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
            original = json.loads(path.read_text())
            row = {
                "cell": path.parent.name,
                "profile": spec.get("profile", "unknown"),
                "original_passed": bool(original.get("passed")),
                "observed_checks_passed": False,
                "combined_diagnostic_passed": False,
                "reason": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(directory),
        "expected_cells": args.expected_cells,
        "completed_cells": len(rows),
        "pending_cells": max(args.expected_cells - len(rows), len(pending)),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formal_certification": False,
    }
    _report(output, rows, metadata)
    print(
        json.dumps(
            {
                **metadata,
                "observed_property_passes": sum(
                    r["observed_checks_passed"] for r in rows
                ),
                "combined_diagnostic_passes": sum(
                    r["combined_diagnostic_passed"] for r in rows
                ),
                "report": str(output / "report.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
