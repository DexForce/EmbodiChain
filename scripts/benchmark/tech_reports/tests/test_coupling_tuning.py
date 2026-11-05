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
"""CPU guards for material specialization; GPU equivalence runs in diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.coupling_tuning import (
    specialize_uniform_materials,
)

SOURCE = """
def kernel(tet_materials, i):
    if tet_materials[i, 0] > 0 or tet_materials[i, 1] > 0:
        return tet_materials[i, 0] + tet_materials[i, 2]
"""


@pytest.mark.parametrize(
    "kind", ["nonuniform", "inactive", "nan", "empty", "wrong_dtype", "wrong_shape"]
)
def test_unsupported_materials_never_silently_drop_work(kind: str) -> None:
    values = np.array([[6000, 15000, 0.001]] * 2, dtype=np.float32)
    if kind == "nonuniform":
        values[1, 0] += 1
    elif kind == "inactive":
        values[:, :2] = 0
    elif kind == "nan":
        values[0, 2] = np.nan
    elif kind == "empty":
        values = values[:0]
    elif kind == "wrong_dtype":
        values = values.astype(np.float64)
    else:
        values = values[:, :2]
    with pytest.raises(ValueError):
        specialize_uniform_materials(SOURCE, values)


def test_constant_embedding_retains_float32_material_value() -> None:
    materials = np.array([[6000, 15000, 0.001]], dtype=np.float32)
    source = specialize_uniform_materials(SOURCE, materials)
    assert repr(float(materials[0, 2])) in source
    assert "wp.static(True)" in source


def test_changed_kernel_contract_fails_before_capture() -> None:
    with pytest.raises(ValueError, match="single"):
        specialize_uniform_materials(
            "def unrelated(): return 1",
            np.array([[6000, 15000, 0.001]], dtype=np.float32),
        )
