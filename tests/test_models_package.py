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

"""Official models ship with the core and import without repository fallbacks."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_dir, get_packages

pytestmark = pytest.mark.no_sim
_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def staged_models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage actual runtime sources with the distribution's package-data rules."""
    tree = ast.parse((_ROOT / "setup.py").read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    data = ast.literal_eval(
        next(item.value for item in call.keywords if item.arg == "package_data")
    )
    distribution = Distribution(
        {
            "packages": [
                name
                for name in get_packages()
                if name.partition(".")[0] in {"embodichain", "embodichain_models"}
            ],
            "package_dir": get_package_dir(),
            "package_data": data,
            "include_package_data": False,
            "script_name": str(_ROOT / "setup.py"),
        }
    )
    output = tmp_path_factory.mktemp("staged_official_models")
    command = build_py(distribution)
    command.build_lib = str(output)
    command.ensure_finalized()
    command.run()
    return output


def test_official_models_and_third_party_license_are_staged(
    staged_models: Path,
) -> None:
    for name in (
        "embodichain_models/__init__.py",
        "embodichain_models/dynamics/__init__.py",
        "embodichain_models/dynamics/nerd/__init__.py",
        "embodichain_models/dynamics/nerd/model.py",
        "embodichain_models/dynamics/nerd/checkpoint.py",
        "embodichain_models/dynamics/nerd/codec.py",
        "embodichain_models/dynamics/nerd/sl_recipe.py",
    ):
        assert (staged_models / name).is_file(), name
    license_path = Path("embodichain_models/dynamics/nerd/THIRD_PARTY_LICENSES")
    source = _ROOT / "extensions/embodichain_models" / license_path
    assert (staged_models / license_path).read_bytes() == source.read_bytes()
    assert not (staged_models / "embodichain/data/nerd/ant.xml").exists()
    assert not (staged_models / "embodichain/data/nerd/cartpole_single.urdf").exists()


def test_staged_models_and_sl_factory_work_without_checkout_or_examples(
    staged_models: Path, tmp_path: Path
) -> None:
    script = """
import importlib.abc
from pathlib import Path
import sys

stage, repository = map(Path, sys.argv[1:])
sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in str(finder)]
sys.path = [str(stage)] + [entry for entry in sys.path if entry and not Path(entry).is_relative_to(repository)]

class NoSimulationOrExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dexsim', 'newton', 'examples'} or fullname.startswith('embodichain.lab'):
            raise AssertionError('model inference imported ' + fullname)

sys.meta_path.insert(0, NoSimulationOrExamples())
from embodichain.learning.models import MLP, EmpiricalNormalizer
import embodichain.learning.sl
assert not any(name == 'embodichain_models' or name.startswith('embodichain_models.') for name in sys.modules)
import embodichain_models
assert not any(name.startswith('embodichain_models.dynamics.') for name in sys.modules)
from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg
from embodichain_models.dynamics.nerd import model
assert Path(model.__file__).is_relative_to(stage)
import embodichain
assert Path(embodichain.__file__).is_relative_to(stage)
cfg = NeRDModelCfg(input_dims={'states_embedding': 4, 'joint_f': 2}, output_dim=4, n_embd=12, n_head=3, n_layer=1, block_size=3, history_length=3, head_hidden_dims=(12,))
network = NeRDModel(cfg)
import torch
prediction = network({'states_embedding': torch.zeros(1, 3, 4), 'joint_f': torch.zeros(1, 3, 2)})
assert prediction.shape == (1, 3, 4)
assert torch.isfinite(prediction).all()
from embodichain_models.dynamics.nerd.sl_recipe import build_supervised_trainer
assert callable(build_supervised_trainer)
from embodichain_models.dynamics.nerd.data import DynamicsShardWriter
from embodichain.learning.sl.workflow import train_from_config
import numpy as np
data = Path.cwd() / 'data'
with DynamicsShardWriter(data, {'codec': {'robot': 'cartpole'}, 'dt': 1 / 60}) as writer:
    for split in ('train', 'validation'):
        writer.add_trajectory(split, {'q': np.zeros((6, 2), np.float32), 'qd': np.zeros((6, 2), np.float32)}, {'joint_f': np.zeros((5, 2), np.float32)}, split=split)
result = train_from_config({
    'factory': 'embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer',
    'model': cfg.to_dict(),
    'data': {'path': str(data)},
    'trainer': {'total_updates': 2, 'batch_size': 1, 'device': 'cpu', 'output_dir': str(Path.cwd() / 'training'), 'validation_interval': 1},
})
assert result['num_updates'] == 2
assert Path(result['checkpoint']).is_file()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(staged_models), str(_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
