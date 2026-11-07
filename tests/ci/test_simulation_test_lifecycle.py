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

import json
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


def test_slow_cpu_groups_preserve_resource_constraints(
    pytester: pytest.Pytester,
) -> None:
    """Only independent slow CPU nodes leave their module's shared group."""
    source = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(source.read_text() + """
def _initialize_sim_engine(renderer):
    raise AssertionError('collection must not initialize a native world')

def pytest_collection_finish(session):
    groups = {}
    for item in session.items:
        groups[item.name] = {
            marker.kwargs['name'] for marker in item.iter_markers('xdist_group')
        }
    shared = groups['test_plain_cpu_a']
    assert len(shared) == 1
    assert next(iter(shared)).startswith('module-')
    assert groups['test_plain_cpu_b'] == shared
    for index in (1, 2):
        cpu = groups[f'test_slow_cpu[{index}]']
        assert len(cpu) == 1
        group = next(iter(cpu))
        assert group.startswith('slow-cpu-') and '@' not in group
        assert cpu != shared
        assert groups[f'test_slow_gpu[{index}]'] == shared
        assert groups[f'test_slow_cuda_name[{index}]'] == shared
        assert groups[f'test_slow_sim[{index}]'] == {'simulation'}
        assert groups[f'test_slow_subprocess_sim[{index}]'] == shared
        assert groups[f'test_explicit_group[{index}]'] == shared | {'explicit-resource'}
    assert groups['test_slow_cpu[1]'] != groups['test_slow_cpu[2]']
""")
    pytester.makeini("""[pytest]
markers =
    slow: independent CPU reference or resource-intensive test
    requires_sim: native simulation test
    no_sim: pure test
    subprocess_sim: child-owned simulation
    gpu: GPU test
    renderer: native rendering test
    requires_tasks: task discovery test
    xdist_group: resource group
""")
    pytester.makepyfile("""
import pytest

@pytest.mark.slow
@pytest.mark.parametrize('value', [1, 2])
def test_slow_cpu(value): pass

def test_plain_cpu_a(): pass
def test_plain_cpu_b(): pass

@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize('value', [1, 2])
def test_slow_gpu(value): pass

@pytest.mark.slow
@pytest.mark.parametrize('value', [1, 2])
def test_slow_cuda_name(value): pass

@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.parametrize('value', [1, 2])
def test_slow_sim(value): pass

@pytest.mark.slow
@pytest.mark.subprocess_sim
@pytest.mark.parametrize('value', [1, 2])
def test_slow_subprocess_sim(value): pass

@pytest.mark.slow
@pytest.mark.xdist_group(name='explicit-resource')
@pytest.mark.parametrize('value', [1, 2])
def test_explicit_group(value): pass
""")
    result = pytester.runpytest_subprocess(
        "--collect-only", "--run-gpu", "-q", "-m", "slow or not slow"
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*14 tests collected*"])


def _make_selection_project(pytester: pytest.Pytester) -> Path:
    """Use the real collection hooks while rejecting native initialization."""
    source = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(source.read_text() + """
def _initialize_sim_engine(renderer):
    raise AssertionError('pure selection checks must not initialize a native world')
""")
    pytester.makeini("""[pytest]
markers =
    no_sim: pure Python test
    gpu: resource lane classification
    xdist_group: resource group
""")
    pytester.makepyfile(
        test_selected="""
from pathlib import Path
import pytest

@pytest.fixture(scope='class')
def shared_fixture():
    with Path('class-setup').open('a') as stream:
        stream.write('setup\\n')

@pytest.mark.no_sim
@pytest.mark.gpu
class TestSelected:
    @pytest.mark.parametrize('value', [1, 2], ids=['one', 'two'])
    def test_selected(self, value, shared_fixture):
        with Path('executed').open('a') as stream:
            stream.write(str(value) + '\\n')

@pytest.mark.no_sim
def test_unselected():
    raise AssertionError('marker filtering must be preserved')
""",
        test_unrelated="""
from pathlib import Path
Path('unrelated-imported').touch()
def test_plain(): pass
""",
    )
    manifest = pytester.path / "selection.json"
    result = pytester.runpytest_subprocess(
        "--collect-only",
        "--run-gpu",
        "-m",
        "gpu",
        "--ci-collect-manifest",
        str(manifest),
        "-q",
    )
    assert result.ret == pytest.ExitCode.OK
    assert not (pytester.path / "executed").exists()
    (pytester.path / "unrelated-imported").unlink()
    return manifest


def test_resource_selection_preserves_cases_without_importing_unrelated_files(
    pytester: pytest.Pytester,
) -> None:
    """Discovery selects exact cases; execution imports only their owning file."""
    manifest = _make_selection_project(pytester)
    selected = json.loads(manifest.read_text())
    assert selected["files"] == ["test_selected.py"]
    assert selected["nodeids"] == [
        "test_selected.py::TestSelected::test_selected[one]",
        "test_selected.py::TestSelected::test_selected[two]",
    ]
    result = pytester.runpytest_subprocess(
        *selected["nodeids"],
        "--run-gpu",
        "-m",
        "gpu",
        "--ci-expected-collection",
        str(manifest),
        "-q",
    )
    result.assert_outcomes(passed=2)
    assert (pytester.path / "executed").read_text().splitlines() == ["1", "2"]
    assert (pytester.path / "class-setup").read_text().splitlines() == ["setup"]
    assert not (pytester.path / "unrelated-imported").exists()
    empty = pytester.path / "empty.json"
    result = pytester.runpytest_subprocess(
        "--collect-only",
        "-m",
        "requires_sim and not requires_sim",
        "--ci-collect-manifest",
        str(empty),
        "-q",
    )
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    assert json.loads(empty.read_text()) == {
        "version": 1,
        "files": [],
        "nodeids": [],
        "markers": {},
    }


@pytest.mark.parametrize(
    "change", ["order", "membership", "markers", "collection-error"]
)
def test_resource_selection_rejects_changes_before_running_any_case(
    pytester: pytest.Pytester,
    change: str,
) -> None:
    """A stale discovery cannot silently omit, reorder or reclassify cases."""
    manifest = _make_selection_project(pytester)
    selected = json.loads(manifest.read_text())
    if change == "order":
        selected["nodeids"].reverse()
    elif change == "membership":
        selected["nodeids"].pop()
    elif change == "markers":
        selected["markers"][selected["nodeids"][0]].remove("gpu")
    else:
        (pytester.path / "test_selected.py").write_text(
            "raise RuntimeError('changed')\n"
        )
    manifest.write_text(json.dumps(selected))
    result = pytester.runpytest_subprocess(
        "test_selected.py",
        "--run-gpu",
        "-m",
        "gpu",
        "--ci-expected-collection",
        str(manifest),
        "--continue-on-collection-errors",
        "-q",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Resource-lane collection * after discovery*"])
    assert not (pytester.path / "executed").exists()
