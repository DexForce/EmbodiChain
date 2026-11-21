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
import sys
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


@pytest.mark.parametrize("workers", [0, 2])
def test_metrics_preserve_phase_results_across_processes(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """Setup failures, skips and passes survive independent worker outputs."""
    import os

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    pytester.makepyfile("""
import pytest
@pytest.fixture
def broken(): raise RuntimeError('setup failed')
def test_pass(): assert 2 + 2 == 4
@pytest.mark.skip(reason='explicit skip')
def test_skip(): pass
def test_setup_error(broken): pass
""")
    pytester.makepyfile(
        test_collection_skip="import pytest; pytest.skip('module unavailable', allow_module_level=True)"
    )
    directory = pytester.path / "metrics"
    result = pytester.runpytest_subprocess(
        "-p",
        "scripts.ci.pytest_metrics",
        f"--ci-metrics-dir={directory}",
        "-n",
        str(workers),
        "-q",
    )
    result.assert_outcomes(passed=1, skipped=2, errors=1)
    reports = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    assert {r["worker"] for r in reports} == (
        {"main", "gw0", "gw1"} if workers else {"main"}
    )
    assert any(
        "module unavailable" in row["reason"]
        for report in reports
        for row in report["collection_reports"]
    )
    phases = [phase for report in reports for phase in report["phases"]]
    assert len({p["nodeid"] for p in phases}) == 3
    assert any(p["phase"] == "setup" and p["outcome"] == "failed" for p in phases)
    assert any(
        p["outcome"] == "skipped" and "explicit skip" in p["reason"] for p in phases
    )
    assert sum(p["phase"] == "call" and p["outcome"] == "passed" for p in phases) == 1


def test_duration_order_keeps_each_group_order_and_selection(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    pytester.makeini("[pytest]\nmarkers = xdist_group(name): resource group\n")
    pytester.makepyfile("""
from pathlib import Path
import pytest

def record(name):
    with Path('order').open('a') as stream: stream.write(name + '\\n')

@pytest.mark.xdist_group(name='expensive')
def test_first(): record('first')
@pytest.mark.xdist_group(name='cheap')
def test_middle(): record('middle')
@pytest.mark.xdist_group(name='expensive')
def test_last(): record('last')
""")
    history = pytester.path / "history"
    history.mkdir()
    (history / "worker.json").write_text(
        json.dumps(
            {
                "version": 1,
                "phases": [
                    {"group": "expensive", "seconds": 3},
                    {"group": "cheap", "seconds": 1},
                ],
            }
        )
    )
    result = pytester.runpytest_subprocess(
        "-p", "scripts.ci.pytest_metrics", f"--ci-duration-history={history}", "-q"
    )
    result.assert_outcomes(passed=3)
    assert (pytester.path / "order").read_text().splitlines() == [
        "first",
        "last",
        "middle",
    ]


def test_metrics_keep_evidence_when_native_process_aborts(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native-style exit cannot erase selection and completed setup evidence."""
    import os

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    pytester.makepyfile("import os\ndef test_crash(): os._exit(7)\n")
    directory = pytester.path / "metrics"
    result = pytester.runpytest_subprocess(
        "-p", "scripts.ci.pytest_metrics", f"--ci-metrics-dir={directory}", "-q"
    )
    assert result.ret == 7
    selections = [
        json.loads(path.read_text())
        for path in (directory / "collection").glob("*.json")
    ]
    assert sum(len(row["collection"]) for row in selections) == 1
    phases = [
        json.loads(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(p["phase"] == "setup" and p["outcome"] == "passed" for p in phases)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="sys.last_exc is a Python 3.12 feature"
)
@pytest.mark.parametrize(
    "resource_marker", ["requires_sim", "gpu"], ids=["scene-owner", "device-owner"]
)
@pytest.mark.parametrize("freeze_imports", [False, True])
def test_failed_native_owner_is_released_before_next_case(
    pytester: pytest.Pytester, resource_marker: str, freeze_imports: bool
) -> None:
    """Traceback retention must not pin a failed test's native owners indefinitely."""
    source = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(source.read_text() + """
import gc, sys, types
for name in ('embodichain', 'embodichain.lab', 'embodichain.lab.sim'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
module = types.ModuleType('embodichain.lab.sim.sim_manager')
module.SimulationManager = types.SimpleNamespace(flush_cleanup_queue=gc.collect)
sys.modules[module.__name__] = module
def _initialize_sim_engine(renderer): pass
""")
    pytester.makeini(
        "[pytest]\nmarkers = requires_sim: native lifecycle\n    xdist_group(name): resource group\n"
    )
    program = """
import weakref
import pytest
references = []
class NativeOwner: pass
@pytest.fixture(autouse=True)
def user_scene_finalizer():
    import gc, os
    if 'EXPECTED_INITIAL_FREEZE_COUNT' in os.environ:
        assert gc.get_freeze_count() > int(os.environ['EXPECTED_INITIAL_FREEZE_COUNT'])
    yield
    import gc
    gc.collect()
    if references:
        assert references[0]() is None
@pytest.mark.requires_sim
def test_failure():
    owner = NativeOwner()
    owner.cycle = owner
    references.append(weakref.ref(owner))
    raise RuntimeError('original failure remains a failure')
@pytest.mark.requires_sim
def test_next_case():
    assert references[0]() is None
"""
    pytester.makepyfile(
        program.replace("@pytest.mark.requires_sim", "@pytest.mark." + resource_marker)
    )
    if freeze_imports:
        wrapper = Path(__file__).parents[2] / "scripts/ci/pytest_no_finalize.py"
        driver = pytester.path / "frozen_driver.py"
        driver.write_text(f"""
import gc, importlib.util, os
spec = importlib.util.spec_from_file_location('native_runner', {str(wrapper)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner._NATIVE_IMPORTS = ()  # Isolate GC ownership from optional SDK dependencies.
os.environ['EXPECTED_INITIAL_FREEZE_COUNT'] = str(gc.get_freeze_count())
runner.main(['--freeze-native-imports', '-q', '--run-gpu'])
""")
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
            result = pytester.run(sys.executable, str(driver), timeout=30)
        assert "static import objects" in result.stdout.str()
    else:
        result = pytester.runpytest_subprocess("-q", "--run-gpu")
    result.assert_outcomes(passed=1, failed=1)
