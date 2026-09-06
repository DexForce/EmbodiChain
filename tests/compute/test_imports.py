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

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import warp as wp


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("embodichain.compute", ["warp", "torch", "embodichain.lab"]),
        (
            "embodichain.compute.geometry._warp.convex_query",
            [
                "embodichain.lab",
                "embodichain.compute.kinematics",
                "embodichain.compute.trajectory",
            ],
        ),
        (
            "embodichain.compute.trajectory",
            ["embodichain.lab", "embodichain.utils.warp", "dexsim", "open3d", "cv2"],
        ),
        ("embodichain.utils.warp", ["embodichain.lab", "dexsim"]),
    ],
)
def test_computation_imports_do_not_load_unrelated_domains(
    module: str, forbidden: list[str]
) -> None:
    code = f"""
import importlib
import importlib.abc
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in {forbidden!r}):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
importlib.import_module({module!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("legacy", "current"),
    [
        ("kinematics.opw_solver", "kinematics._warp.opw"),
        ("kinematics.srs_solver", "kinematics._warp.srs"),
        ("kinematics.ur_solver", "kinematics._warp.ur"),
        ("kinematics.interpolate", "trajectory._warp.resampling"),
        ("kinematics.warp_trajectory", "trajectory._warp.warping"),
        ("collision.convex_query", "geometry._warp.convex_query"),
    ],
)
def test_legacy_exports_share_kernel_and_struct_identity(
    legacy: str, current: str
) -> None:
    old = importlib.import_module("embodichain.utils.warp." + legacy)
    new = importlib.import_module("embodichain.compute." + current)
    for name in old.__all__:
        target = (
            "compute_offset_key_poses_kernel"
            if name == "get_offset_qpos_kernel"
            else name
        )
        assert getattr(old, name) is getattr(new, target)


@pytest.mark.parametrize("algorithm", ["opw", "srs", "ur"])
def test_analytical_kinematics_kernels_compile_on_cpu(algorithm: str) -> None:
    module = importlib.import_module(
        "embodichain.compute.kinematics._warp." + algorithm
    )
    wp.init()
    wp.load_module(module=module, device="cpu")
