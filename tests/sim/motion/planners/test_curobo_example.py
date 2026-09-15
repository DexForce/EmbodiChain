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

"""Regression tests for the standalone cuRobo planner example."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import runpy
import sys
from typing import Callable, cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_PATH = PROJECT_ROOT / "examples/sim/motion/planners/curobo_planner.py"
EXPECTED_DEFAULT_SEED = 0
BOX_MESH_SIZE = (0.2, 0.4, 0.6)


@pytest.fixture(scope="module")
def example_namespace() -> dict[str, object]:
    """Load the example module without starting its simulation."""
    return runpy.run_path(
        str(EXAMPLE_PATH),
        run_name="__curobo_planner_example_test__",
    )


@pytest.fixture(scope="module")
def parse_example_args(example_namespace: dict[str, object]) -> Callable[[], Namespace]:
    """Return the example argument parser."""
    return cast(Callable[[], Namespace], example_namespace["parse_args"])


def test_standalone_parser_uses_a_concrete_seed(
    monkeypatch: pytest.MonkeyPatch,
    parse_example_args: Callable[[], Namespace],
) -> None:
    """The standalone example must not inherit the launcher's None seed."""
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE_PATH)])

    args = parse_example_args()

    assert args.seed == EXPECTED_DEFAULT_SEED


def test_temporary_box_mesh_has_requested_centered_bounds(
    example_namespace: dict[str, object],
) -> None:
    """The generated obstacle mesh must match its simulator pose convention."""
    import open3d as o3d

    create_box_mesh = cast(
        Callable[[list[float]], Path],
        example_namespace["_create_temporary_box_mesh"],
    )
    mesh_path = create_box_mesh(list(BOX_MESH_SIZE))
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        bounds = mesh.get_axis_aligned_bounding_box()

        assert bounds.get_center() == pytest.approx((0.0, 0.0, 0.0))
        assert bounds.get_extent() == pytest.approx(BOX_MESH_SIZE)
    finally:
        mesh_path.unlink(missing_ok=True)
