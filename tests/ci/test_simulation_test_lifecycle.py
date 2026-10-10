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

"""Exercise pytest's real setup order without acquiring native GPU resources."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("renderer", ["hybrid", "fast-rt"])
@pytest.mark.parametrize("marker", ["requires_sim", "no_sim", "subprocess_sim"])
@pytest.mark.parametrize("skipped", [False, True])
def test_process_renderer_precedes_class_scene_setup(
    pytester: pytest.Pytester, renderer: str, marker: str, skipped: bool
) -> None:
    """Class scenes see the chosen renderer; pure, child and skipped tests do not init."""
    source = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(
        source.read_text()
        + """
native_initializations = []

def _initialize_sim_engine(renderer):
    native_initializations.append(renderer)

@pytest.fixture(autouse=True)
def wait_scene_destruction_after_test():
    # Native initialization is mocked, so no native cleanup is needed.
    yield

def pytest_sessionfinish(session, exitstatus):
    assert native_initializations == EXPECTED
""".replace(
            "EXPECTED",
            repr([renderer] if marker == "requires_sim" and not skipped else []),
        )
    )
    pytester.makeini("""[pytest]
markers =
    requires_sim: native simulation test
    no_sim: pure test
    subprocess_sim: child-owned simulation
    gpu: GPU test
    renderer: native rendering test
    requires_tasks: task discovery test
    xdist_group: resource group
""")
    expected = [renderer] if marker == "requires_sim" else []
    pytester.makepyfile(f"""
import pytest
import conftest

pytestmark = [pytest.mark.{marker}, pytest.mark.skipif({skipped!r}, reason='fixture order')]

class TestClassScopedScene:
    @classmethod
    def setup_class(cls):
        assert conftest.native_initializations == {expected!r}

    def test_scene_and_camera_can_share_the_process_renderer(self):
        assert conftest.native_initializations == {expected!r}
""")
    result = pytester.runpytest_subprocess("--renderer", renderer, "-q")
    result.assert_outcomes(**{"skipped" if skipped else "passed": 1})
