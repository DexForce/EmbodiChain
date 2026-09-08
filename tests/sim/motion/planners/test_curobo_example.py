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
from typing import Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_PATH = PROJECT_ROOT / "examples/sim/motion/planners/curobo_planner.py"
EXPECTED_DEFAULT_SEED = 0


@pytest.fixture(scope="module")
def parse_example_args() -> Callable[[], Namespace]:
    """Load the example parser without starting its simulation."""
    namespace = runpy.run_path(
        str(EXAMPLE_PATH),
        run_name="__curobo_planner_example_test__",
    )
    return namespace["parse_args"]


def test_standalone_parser_uses_a_concrete_seed(
    monkeypatch: pytest.MonkeyPatch,
    parse_example_args: Callable[[], Namespace],
) -> None:
    """The standalone example must not inherit the launcher's None seed."""
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE_PATH)])

    args = parse_example_args()

    assert args.seed == EXPECTED_DEFAULT_SEED
