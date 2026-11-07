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

"""Standalone historical physics checks without importing EmbodiChain or SDK."""

from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from ._official_evaluation_data import _hash

__all__: list[str] = []

OFFICIAL_COMMIT = "370e89d2b8ce68aa3578aa7812859c681b7502bc"


def _positive(name: str, value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _physical_arrays(model: Any) -> dict[str, Any]:
    """Read the same physical fields recorded by original-teacher collection."""
    import numpy as np

    result = {}
    for name in (
        "body_mass",
        "body_inertia",
        "body_com",
        "joint_armature",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_axis_mode",
        "joint_target_ke",
        "joint_target_kd",
    ):
        value = getattr(model, name, None)
        if value is not None:
            result[name] = (
                value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            )
    for field in ("mu", "ke", "kd", "kf"):
        value = getattr(model.shape_materials, field, None)
        if value is not None:
            result["shape_material_" + field] = value.numpy()
    result["shape_thickness"] = model.shape_geo.thickness.numpy()
    result["shape_body"] = model.shape_body.numpy()
    result["ground_plane"] = model.ground_plane.numpy()
    result["gravity"] = np.asarray(model.gravity, dtype=np.float64)
    return result


def _canonical_physics(
    arrays: Mapping[str, Any], batch: int, bodies: int
) -> dict[str, Any]:
    """Remove replica indexing while proving the recorded replicas homogeneous."""
    import numpy as np

    result = {}
    shape_body = np.asarray(arrays["shape_body"])
    # Original Warp always appends one shared ground-plane shape.
    if (len(shape_body) - 1) % batch:
        raise ValueError("Historical shape replication is not homogeneous")
    shapes = (len(shape_body) - 1) // batch
    for key, values in arrays.items():
        value = np.asarray(values)
        if key in {"ground_plane", "gravity"}:
            result[key] = value.tolist()
            continue
        if key.startswith("shape_"):
            if len(value) != batch * shapes + 1:
                raise ValueError(f"Unexpected historical shape field {key}")
            rows = value[:-1].reshape(batch, shapes, *value.shape[1:]).copy()
            if key == "shape_body":
                rows = np.where(
                    rows >= 0, rows - np.arange(batch)[:, None] * bodies, rows
                )
            tail = value[-1:].tolist()
        else:
            if len(value) % batch:
                raise ValueError(f"Unexpected historical replication for {key}")
            rows = value.reshape(batch, len(value) // batch, *value.shape[1:])
            tail = None
        if not np.allclose(rows, rows[:1], rtol=1e-6, atol=1e-7, equal_nan=False):
            raise ValueError(f"Historical physical field {key} differs across replicas")
        result[key] = {"per_environment": rows[0].tolist(), "shared": tail}
    return result


def _verify_historical_physics(
    environment: Any,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    warp_version: str,
) -> dict[str, Any]:
    """Require actual historical physical data, never relabel a Newton manifest."""
    import numpy as np

    recorded = manifest.get("teacher_origin", {})
    if (
        recorded.get("kind") != "official_warp"
        or recorded.get("source_commit") != OFFICIAL_COMMIT
        or recorded.get("warp_version") != warp_version
    ):
        raise ValueError(
            "Model does not record the required original-Warp teacher identity"
        )
    if not isinstance(manifest.get("source_physical_arrays"), Mapping):
        raise ValueError(
            "Historical source_physical_arrays are required; do not relabel a Newton model"
        )
    if (
        recorded.get("up_axis") != int(environment.model.up_axis)
        or recorded.get("num_substeps") != environment.sim_substeps_gt
    ):
        raise ValueError("Original teacher axis/substeps differ from model provenance")
    if recorded.get("integrator") != str(environment.integrator_type_gt):
        raise ValueError("Original teacher integrator differs from model provenance")
    if "gravity" in recorded and not np.allclose(
        np.asarray(recorded["gravity"]),
        np.asarray(environment.model.gravity, dtype=np.float64),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Original teacher gravity differs from recorded provenance")
    if not math.isclose(
        float(manifest.get("dt", 0)),
        float(environment.frame_dt),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Original teacher timestep differs from model provenance")
    robot = manifest["codec"]["robot"]
    expected_continuous = [1] if robot == "cartpole" else []
    if (
        list(manifest["codec"].get("continuous_q_indices", expected_continuous))
        != expected_continuous
    ):
        raise ValueError(
            "Historical codec continuous coordinates differ from original runtime"
        )
    if not np.allclose(
        manifest["codec"].get("base_com", [0.0, 0.0, 0.0]),
        environment.model.body_com.numpy()[0],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Historical codec base COM differs from actual source model")
    asset = (
        Path(source["root"])
        / "envs"
        / "warp_sim_envs"
        / "assets"
        / ("ant.xml" if robot == "ant" else "cartpole_single.urdf")
    )
    if manifest.get("asset_sha256") != _hash(asset):
        raise ValueError("Historical model asset checksum differs")
    batch = manifest.get("sampling", {}).get("batch_size")
    split_batch = (
        manifest.get("source_split_identities", {}).get("train", {}).get("num_envs")
    )
    if batch is not None and split_batch is not None and batch != split_batch:
        raise ValueError("Conflicting source sampling batch sizes")
    batch = _positive(
        "source sampling batch_size", batch if batch is not None else split_batch
    )
    bodies = _positive("source bodies_per_env", recorded.get("bodies_per_env"))
    if bodies != environment.bodies_per_env:
        raise ValueError("Historical body count differs")
    actual = _canonical_physics(
        _physical_arrays(environment.model), environment.num_envs, bodies
    )
    expected = _canonical_physics(manifest["source_physical_arrays"], batch, bodies)
    missing_fields = sorted(set(actual) - set(expected))
    if set(expected) - set(actual) or set(missing_fields) - {"gravity"}:
        raise ValueError("Historical physical field set differs")
    for key in expected:
        if key in {"ground_plane", "gravity"}:
            pairs = [(actual[key], expected[key])]
        else:
            pairs = [
                (actual[key][name], expected[key][name])
                for name in ("per_environment", "shared")
                if actual[key][name] is not None or expected[key][name] is not None
            ]
        for left, right in pairs:
            if (
                left is None
                or right is None
                or np.shape(left) != np.shape(right)
                or not np.allclose(left, right, rtol=1e-6, atol=1e-7)
            ):
                raise ValueError(f"Historical physical mismatch in {key}")
    identity = hashlib.sha256(
        json.dumps(
            actual, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "schema": "official-warp-physical-arrays-v2",
        "sha256": identity,
        "matched_fields": sorted(expected),
        "missing_recorded_fields": missing_fields,
        "matched_recorded_fields": True,
        "complete": not missing_fields,
        "status": (
            "matched" if not missing_fields else "pending_missing_recorded_physics"
        ),
        "gravity_live": np.asarray(
            environment.model.gravity, dtype=np.float64
        ).tolist(),
        "gravity_magnitude_in_training_metadata": "gravity" in expected,
    }
