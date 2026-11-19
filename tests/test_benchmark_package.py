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

"""Build and install the real wheel before probing benchmark APIs outside git."""

from __future__ import annotations

import os
from email.parser import Parser
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.no_sim


@pytest.fixture(scope="module")
def installed_benchmark(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build from isolated sources and install without changing the Python env."""
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path_factory.mktemp("installed_benchmark")
    source = workspace / "source"
    source.mkdir()
    for name in (
        "setup.py",
        "pyproject.toml",
        "VERSION",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
    ):
        shutil.copy2(root / name, source / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for name in ("embodichain", "examples"):
        shutil.copytree(root / name, source / name, ignore=ignore)
    # Reuse staging from the retired top-level layout.
    stale = source / "build" / "lib" / "embodichain_tasks"
    stale.mkdir(parents=True)
    (stale / "__init__.py").touch()
    built = subprocess.run(
        [
            sys.executable,
            "-I",
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(workspace / "dist"),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((workspace / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.namelist()
        metadata_path = next(
            name for name in entries if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        assert "assembly" not in metadata.get_all("Provides-Extra", [])
        assert not any(
            Requirement(value).name.lower() == "mujoco"
            for value in metadata.get_all("Requires-Dist", [])
        )
        assert "embodichain/extensions/benchmarks/__init__.py" in entries
        assert "embodichain/extensions/benchmarks/motion_generation/api.py" in entries
        assert "embodichain/cli/benchmark.py" in entries
        assert (
            "embodichain/extensions/benchmarks/assembly/robot_reference.py" in entries
        )
        assert (
            "embodichain/extensions/robots/flexiv/rizon4s/rizon4s_sources.json"
            in entries
        )
        assert "embodichain/toolkits/assembly/native.py" in entries
        assert not any(
            name.startswith("embodichain/lab/sim/assembly/") for name in entries
        )
        assert not any(
            "share/embodichain/assembly_patches/" in name for name in entries
        )

        for name in ("LICENSE.worldparticle", "LICENSE.tome", "PROVENANCE.json"):
            assert (
                f"embodichain/extensions/models/dynamics/worldparticle/{name}"
                in entries
            )
        suites = {
            Path(name).name
            for name in entries
            if name.startswith(
                "embodichain/extensions/benchmarks/motion_generation/suites/"
            )
            and name.endswith(".yaml")
        }
        assert suites == {
            path.name
            for path in (
                root / "embodichain/extensions/benchmarks/motion_generation/suites"
            ).glob("*.yaml")
        }
        assert not any(name.startswith("scripts/") for name in entries)
        assert not any(name.startswith("embodichain/benchmarks/") for name in entries)
        assert not any(
            name.startswith(("extensions/", "embodichain_")) for name in entries
        )
    target = workspace / "consumer_env"
    created = subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--system-site-packages", str(target)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    # Expose this exact version as an inherited installation. The consumer
    # must install its own copy rather than reuse that distribution.
    inherited = workspace / "inherited"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(inherited)
    site_packages = (
        target
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    # A nested venv inherits the base interpreter's packages, not the calling
    # venv's dependencies. Make those installed dependency directories visible
    # without activating their editable finders or exposing checkout paths.
    dependency_paths = [
        str(Path(entry).resolve())
        for entry in sys.path
        if entry
        and (
            Path(entry).name in {"site-packages", "dist-packages"}
            or any(Path(entry).glob("*.dist-info"))
        )
    ]
    (site_packages / "test_dependencies.pth").write_text(
        "\n".join(dependency_paths) + "\n"
    )
    (site_packages / "inherited_benchmark.pth").write_text(str(inherited) + "\n")
    interpreter = target / "bin/python"
    installed = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-compile",
            str(wheel),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    return target, source


def test_installed_suites_and_cli_work_without_source_checkout(
    installed_benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    target, source = installed_benchmark
    # Prove resources are loaded from the wheel even when its build tree is gone.
    shutil.rmtree(source)
    script = r"""
import ast
import importlib.abc
from importlib.resources import files
from pathlib import Path
import sys

target, repository = map(Path, sys.argv[1:3])
sys.meta_path = [finder for finder in sys.meta_path if "__editable__" not in str(finder)]
sys.path = [entry for entry in sys.path if entry and not Path(entry).is_relative_to(repository)]

class RejectSamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"scripts", "examples"}:
            raise AssertionError("installed benchmark imported sample modules")

sys.meta_path.insert(0, RejectSamples())
from embodichain.cli.benchmark import main as benchmark_cli
benchmark_cli(["--help"])
assert not any(name == "embodichain.extensions.benchmarks" or name.startswith("embodichain.extensions.benchmarks.") for name in sys.modules)
import embodichain.extensions.benchmarks.motion_generation as benchmark
from embodichain.extensions.benchmarks.motion_generation import load_suite, run_benchmark
assert Path(benchmark.__file__).is_relative_to(target)
assert "embodichain.lab.sim" not in sys.modules
from embodichain.toolkits.assembly import native
assert Path(native.__file__).is_relative_to(target)
assert callable(native.prepare_native_configs)
from embodichain.extensions.benchmarks.assembly.precision import PrecisionRunCfg
profile = PrecisionRunCfg(backend="newton", newton_solver="dexuni", solver="cg")
assert profile.to_dict()["newton_solver"] == "dexuni"
assert "embodichain.lab.sim" not in sys.modules
for source in Path(benchmark.__file__).parent.rglob("*.py"):
    for node in ast.walk(ast.parse(source.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) else
                 [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        assert not any(name and name.split(".")[0] in {"scripts", "examples"} for name in names), source
resources = files(benchmark).joinpath("suites")
suite_names = sorted(path.name[:-5] for path in resources.iterdir() if path.name.endswith(".yaml"))
for name in suite_names:
    suite = load_suite(name)
    assert suite.name
assert load_suite().name == "motion_panda_free_space_smoke"
from embodichain.extensions.benchmarks.motion_generation import PlannerAdapter
from embodichain.extensions.benchmarks.motion_generation.scenarios.atomic_grasps import parallel_jaw_sampler
assert parallel_jaw_sampler(2)
from embodichain.extensions.benchmarks.motion_generation.scenarios._placement_geometry import build_support_object_target_pose
import torch
assert build_support_object_target_pose(torch.eye(4), torch.device("cpu")).shape == (4, 4)
from embodichain.extensions.benchmarks.motion_generation.artifacts import environment_metadata
assert environment_metadata()["git_commit"] is None
from embodichain.data import get_data_class
from embodichain.extensions.benchmarks.assembly import robot_reference
assert Path(robot_reference.__file__).is_relative_to(target)
assert get_data_class("FlexivRizon4SMenagerie").__module__ == "embodichain.extensions.robots.flexiv.rizon4s._menagerie_assets"
assert files("embodichain.extensions.robots.flexiv.rizon4s").joinpath("rizon4s_sources.json").is_file()
from embodichain.cli.main import main
try:
    main(["benchmark", "motion-generation", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("CLI did not dispatch to the installed suite parser")
print("installed benchmark suites and CLI passed")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(target / "bin/python"),
            "-I",
            "-c",
            script,
            str(target),
            str(Path(__file__).resolve().parents[1]),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--algorithms" in result.stdout
    assert "installed benchmark suites and CLI passed" in result.stdout
    cli = subprocess.run(
        [str(target / "bin/embodichain"), "benchmark", "motion-generation", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert "--suite" in cli.stdout
    robot_cli = subprocess.run(
        [
            str(target / "bin/python"),
            "-I",
            "-m",
            "embodichain.extensions.benchmarks.assembly.robot_reference",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert robot_cli.returncode == 0, robot_cli.stdout + robot_cli.stderr
    assert "--gripper-model" in robot_cli.stdout

    for example in ("drop_peg", "visual_demo"):
        help_result = subprocess.run(
            [
                str(target / "bin/embodichain"),
                "run-example",
                f"sim/assembly/{example}",
                "--help",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert help_result.returncode == 0, help_result.stdout + help_result.stderr
        assert "--newton-solver" in help_result.stdout
        assert "dexuni" in help_result.stdout


def test_installed_solver_entry_point_and_core_dependency_boundary(
    installed_benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    """Discover optional adapters without eagerly importing models or the SDK."""
    target, _ = installed_benchmark
    script = r"""
import importlib.abc
from importlib import metadata
from pathlib import Path
import sys

import embodichain.extensions.solvers.physics.nerd
import embodichain.extensions.solvers.physics.worldparticle
assert Path(embodichain.extensions.solvers.__file__).is_relative_to(Path(sys.prefix))
for name in ('torch', 'newton', 'dexsim', 'embodichain.extensions.models'):
    assert name not in sys.modules, name

class ConcreteImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        packages = ('embodichain.extensions.models', 'embodichain.extensions.solvers')
        if any(fullname == package or fullname.startswith(package + '.') for package in packages):
            raise AssertionError('Core imported a concrete implementation: ' + fullname)

blocker = ConcreteImportsForbidden()
sys.meta_path.insert(0, blocker)
from embodichain.lab.gym.envs.differentiable_env import DifferentiableEnv
from embodichain.lab.sim.diff._newton_fk import _newton_fk
sys.meta_path.remove(blocker)
assert 'embodichain.extensions.models' not in sys.modules
entries = [entry for entry in metadata.entry_points(group='dexsim.newton_solvers') if entry.name == 'nerd']
assert len(entries) == 1
assert entries[0].value == 'embodichain.extensions.solvers.physics.nerd.provider:NeRDProvider'
provider = entries[0].load()
assert provider.capabilities.supported_control_fields == ('joint_f',)
entries = [entry for entry in metadata.entry_points(group='dexsim.newton_solvers') if entry.name == 'worldparticle']
assert len(entries) == 1
assert entries[0].value == 'embodichain.extensions.solvers.physics.worldparticle.provider:WorldParticleProvider'
provider = entries[0].load()
assert provider.capabilities.supported_particle_kinds == ('fluid',)
assert not provider.capabilities.cuda_graph
assert 'embodichain.extensions.models.dynamics.worldparticle' not in sys.modules
assert 'open3d.ml.torch' not in sys.modules
assert 'flash_attn' not in sys.modules
import dexsim
assert dexsim.get_world_num() == 0
"""
    result = subprocess.run(
        [str(target / "bin/python"), "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_policy_components_work_without_source_or_simulator(
    installed_benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    """The wheel supplies the composed learning API without importing native code."""
    target, _ = installed_benchmark
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import inspect
from pathlib import Path
import sys

target, repository = map(Path, sys.argv[1:3])
sys.meta_path = [finder for finder in sys.meta_path if "__editable__" not in str(finder)]
sys.path = [entry for entry in sys.path if entry and not Path(entry).is_relative_to(repository)]
class NoSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dexsim' or fullname.startswith(('dexsim.', 'embodichain.lab')):
            raise AssertionError('Offline policy imported simulation: ' + fullname)
sys.meta_path.insert(0, NoSimulation())
import torch
from tensordict import TensorDict
from embodichain.learning.rl.policies import ActorCritic, ContactPrediction, TemporalActor, build_policy
from embodichain.compute.geometry import signed_distance_to_polygon
from embodichain.compute.kinematics import URDFGravityCompensation
from embodichain.data.resources import snapshot_urdf_resources
for cls in (ActorCritic, ContactPrediction, TemporalActor, signed_distance_to_polygon, URDFGravityCompensation, snapshot_urdf_resources):
    assert Path(inspect.getfile(cls)).is_relative_to(target)
policy = build_policy({
    'name': 'actor_critic',
    'obs_groups': {'actor': ['proprioception'], 'critic': ['privileged']},
    'actor': {'type': 'temporal', 'network_cfg': {'frame_dim': 3, 'history_length': 30, 'hidden_sizes': [8]}},
    'critic': {'type': 'mlp', 'network_cfg': {'hidden_sizes': [8]}},
    'auxiliary_tasks': {'contact': {'type': 'contact_prediction', 'contact_points': 2, 'hidden_sizes': [8]}},
}, 93, 2, torch.device('cpu'), critic_obs_space=4)
td = TensorDict({'obs': torch.zeros(2, 93), 'critic_obs': torch.ones(2, 4),
                 'contact_labels': torch.tensor([[0, 1], [1, 0]])}, batch_size=[2])
policy.get_action(td, deterministic=True)
evaluated = policy.evaluate_actions(td)
loss = policy.auxiliary_loss(td, evaluated)
assert torch.isfinite(loss) and loss > 0
loss.backward()
torch.testing.assert_close(policy.export_actor()(td['obs']), td['action'])
assert not any(name.startswith(('dexsim', 'embodichain.lab')) for name in sys.modules)
"""
    result = subprocess.run(
        [str(target / "bin/python"), "-I", "-c", script, str(target), str(repository)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_assembly_functors_and_defaults_have_task_ownership(
    installed_benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    """Installed task components resolve their packaged defaults without a checkout."""
    target, _ = installed_benchmark
    script = r"""
from pathlib import Path
from types import SimpleNamespace
import inspect,sys
sys.meta_path=[finder for finder in sys.meta_path if "__editable__" not in str(finder)]
import torch
from embodichain.extensions.tasks.manipulation.assembly.functors import AssemblyProprioception,assembly_milestone_reward
from embodichain.lab.gym.envs.managers import observations
from embodichain.lab.gym.utils.gym_utils import get_manager_modules
assert Path(inspect.getfile(AssemblyProprioception)).is_relative_to(sys.prefix)
assert not hasattr(observations,"AssemblyProprioception")
assert "embodichain.extensions.tasks.manipulation.assembly.functors" in get_manager_modules()
state=SimpleNamespace(refresh_assembly_state=lambda:None,_assembly_new_milestones=torch.tensor([[True,False,True,False],[False,False,False,False]]))
env=SimpleNamespace(assembly_state=state,device=torch.device("cpu"))
torch.testing.assert_close(assembly_milestone_reward(env,None,None,{},[1.,0.,2.,0.]),torch.tensor([3.,0.]))
"""
    result = subprocess.run(
        [str(target / "bin/python"), "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bundled_vlp16_entry_point_accepts_yaml(
    installed_benchmark: tuple[Path, Path],
) -> None:
    """Decode the nominal firing order from a real, non-editable installation."""
    target, source = installed_benchmark
    python = target / "bin/python"
    run_dir = target.parent
    completed = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import yaml; from embodichain.lab.sim.sensors import SensorCfg; "
            "from embodichain.extensions.sensors import VLP16Cfg; "
            "cfg=SensorCfg.from_dict(yaml.safe_load('sensor_type: VLP16')); "
            "assert isinstance(cfg,VLP16Cfg); assert cfg.horizontal_samples == 1800; "
            "assert tuple(cfg.vertical_angles[::2]) == tuple(range(-15,0,2)); "
            "assert tuple(cfg.vertical_angles[1::2]) == tuple(range(1,16,2))",
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
