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

"""Strict JSON persistence and rigid-transform validation for assembly jobs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ["read_json", "write_json", "pose_matrix"]


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"JSON numbers must be finite: {token}")
    return value


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON object, rejecting nonfinite numeric values.

    Args:
        path: Input file, including configuration or a model response.

    Returns:
        Parsed object. Non-object roots and invalid numbers raise ValueError.
    """
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=_finite_float,
        parse_constant=_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a UTF-8 JSON object without exposing partial output.

    Args:
        path: Destination; missing parent directories are created.
        value: JSON-compatible object containing only finite numbers.
    """
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    # Serialize first so invalid values cannot damage an existing checkpoint.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pose_matrix(value: object) -> np.ndarray:
    """Validate a finite SE(3) transform mapping assemble coordinates into base.

    Args:
        value: Numeric 4 x 4 matrix; translation is in meters.

    Returns:
        Independent float64 matrix with a proper orthonormal rotation.
    """
    # Keep configuration loading and CLI help free of numerical dependencies.
    import numpy as np

    try:
        matrix = np.asarray(value)
        if matrix.dtype.kind not in "iuf":
            raise ValueError("Matrix entries must be numbers")
        matrix = matrix.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Pose must be a finite numeric 4 x 4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Pose must be a finite numeric 4 x 4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Pose bottom row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Pose rotation must be orthonormal with determinant +1")
    return matrix
