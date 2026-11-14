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

"""Provenance and input-selection contracts of the in-hand rotation examples."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import embodichain

pytestmark = pytest.mark.no_sim

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_ADAPTED_TARGET = 0.75


@pytest.fixture
def load_example(monkeypatch) -> Iterator[Callable[[str, str], ModuleType]]:
    """Import flat example modules from their directory, then forget them."""
    roots: set[Path] = set()

    def load(example: str, name: str) -> ModuleType:
        root = _EXAMPLES / "learning" / example
        roots.add(root)
        monkeypatch.syspath_prepend(str(root))
        monkeypatch.delitem(sys.modules, name, raising=False)
        return importlib.import_module(name)

    yield load
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if location and Path(location).resolve().parent in roots:
            del sys.modules[name]


def _package(root: Path) -> Path:
    path = root / "src/sharpa_rl_unilab/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return path


def test_sharpa_learner_must_be_imported_from_the_pinned_checkout(
    load_example, monkeypatch, tmp_path: Path
) -> None:
    reference = load_example("sharpa_inhand", "reference")
    pinned = tmp_path / "pinned"
    _package(pinned)
    learner = ModuleType("sharpa_rl_unilab")
    learner.__file__ = str(_package(tmp_path / "other"))
    monkeypatch.setitem(sys.modules, "sharpa_rl_unilab", learner)

    with pytest.raises(ValueError, match="not the --upstream checkout"):
        reference._verify_imported_learner(pinned)

    learner.__file__ = str(pinned / "src/sharpa_rl_unilab/__init__.py")
    assert (
        reference._verify_imported_learner(pinned)
        == (pinned / "src/sharpa_rl_unilab").resolve()
    )


def test_native_resume_reads_core_sources_from_a_wheel_install(
    tmp_path: Path,
) -> None:
    # A wheel puts examples in <prefix>/share/embodichain and core in site-packages.
    examples = tmp_path / "prefix/share/embodichain/examples"
    relocated = examples / "learning/sharpa_inhand/_native_checkpoint.py"
    relocated.parent.mkdir(parents=True)
    shutil.copy2(_EXAMPLES / "learning/sharpa_inhand/_native_checkpoint.py", relocated)
    spec = importlib.util.spec_from_file_location("_wheel_native_checkpoint", relocated)
    checkpoint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_module)

    run = tmp_path / "run"
    core = Path(embodichain.__file__).resolve().parent / "lab/sim"
    for name in checkpoint_module._ADAPTER_FILES:
        (run / "sources" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(core / name, run / "sources" / name)
    hashes = {name: "recorded" for name in checkpoint_module._PHYSICAL_FILES}
    (run / "experiment.json").write_text(
        json.dumps({"resource_profile": "reference_host", "source_hashes": hashes})
    )
    snapshot = {"native_environment": True, "native_identity": {"scales": [1.0] * 8}}

    checkpoint_module._verify_resume(
        run / "native_teacher_00100.pt", snapshot, hashes, "reference_host"
    )
    assert checkpoint_module._runtime_source(
        "examples/learning/sharpa_inhand/native_full.py"
    ) == (examples / "learning/sharpa_inhand/native_full.py")


def test_wuji_hand_usd_is_loaded_from_the_supplied_asset_root(
    load_example, tmp_path: Path
) -> None:
    physics = load_example("wuji_inhand", "_physics")
    original = tmp_path / "original/compiled/hand.usda"
    original.parent.mkdir(parents=True)
    original.write_text("#usda 1.0 (changed after the cache was copied)\n")
    copied = tmp_path / "copied/compiled/hand.usda"
    copied.parent.mkdir(parents=True)
    copied.write_text("#usda 1.0\n")
    profile = {
        "native_source": str(original),
        "native_source_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    }

    assert physics._native_hand_source(copied.parents[1], profile) == copied.resolve()
    with pytest.raises(ValueError, match="native_source_sha256"):
        physics._native_hand_source(original.parents[1], profile)


class _ScreeningPhysics:
    """CPU stand-in that holds one batch of 256 measured, valid grasps."""

    def __init__(self, assets: Path, output: Path, **_: object) -> None:
        self.device = torch.device("cpu")
        self.cache = [torch.zeros(32, 47) for _ in range(8)]
        self.target = torch.zeros(256, 20)

    def reset(self, choices: torch.Tensor) -> torch.Tensor:
        poses = torch.zeros(len(choices), 47)
        poses[:, 23] = 1.0
        poses[:, 27:] = 0.5
        self.target[:] = poses[:, 27:]
        return poses

    def step(self, target: torch.Tensor) -> None:
        pass

    def state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quat = torch.zeros(256, 4)
        quat[:, 0] = 1.0
        return torch.zeros(256, 3), quat, torch.zeros(256, 20)

    def grasp_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(256, 5), torch.ones(256, dtype=torch.bool)

    def close(self) -> None:
        pass


class _Feedback:
    """Stand-in for the preparation controller: it moves the holding target."""

    def __init__(self, physical: _ScreeningPhysics, assets: Path) -> None:
        self.physical = physical

    def update(self) -> None:
        self.physical.target[:] = _ADAPTED_TARGET


def _prepared(root: Path) -> Path:
    (root / "compiled").mkdir(parents=True)
    (root / "scene.xml").write_text("<mujoco/>\n")
    (root / "compiled/source.json").write_text("{}\n")
    for scale in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        np.save(root / f"candidates_{scale:g}.npy", np.zeros((32, 47), np.float32))
    return root


def _screen(screen: ModuleType, monkeypatch, assets: Path, output: Path) -> None:
    monkeypatch.setattr(screen, "_WujiPhysics", _ScreeningPhysics)
    monkeypatch.setattr(screen, "_GraspSettler", _Feedback)
    monkeypatch.setattr(
        screen, "SimulationManager", SimpleNamespace(flush_cleanup_queue=lambda: None)
    )
    argv = ["screen.py", "--assets", str(assets), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--steps", "2", "--adapt-steps", "2"])
    screen.main()


def test_wuji_cache_pairs_screened_states_with_the_target_that_held_them(
    load_example, monkeypatch, tmp_path: Path
) -> None:
    screen = load_example("wuji_inhand", "screen")
    cache = load_example("wuji_inhand", "build_initial_cache")
    output = tmp_path / "screen"
    _screen(screen, monkeypatch, _prepared(tmp_path / "prepared"), output)

    batches, feedback_steps = cache._screened_batches(output, tmp_path / "prepared")
    packet = torch.load(batches[0], weights_only=False)
    assert torch.all(packet["initial"][:, 27:] == 0.5)
    np.testing.assert_array_equal(
        cache._holding_targets(packet, feedback_steps),
        np.full((256, 20), _ADAPTED_TARGET, np.float32),
    )


def test_wuji_cache_accepts_unrecorded_targets_only_without_feedback(
    load_example,
) -> None:
    cache = load_example("wuji_inhand", "build_initial_cache")
    initial = torch.zeros(256, 47)
    initial[:, 27:] = 0.5

    np.testing.assert_array_equal(
        cache._holding_targets({"initial": initial}, 0), initial[:, 27:].numpy()
    )
    with pytest.raises(ValueError, match="rerun screen.py"):
        cache._holding_targets({"initial": initial}, 2)


def test_wuji_cache_reads_only_batches_of_the_screening_run_on_its_assets(
    load_example, monkeypatch, tmp_path: Path
) -> None:
    screen = load_example("wuji_inhand", "screen")
    cache = load_example("wuji_inhand", "build_initial_cache")
    prepared = _prepared(tmp_path / "prepared")
    output = tmp_path / "screen"
    _screen(screen, monkeypatch, prepared, output)
    shutil.copy2(output / "batch_0.pt", output / "batch_1.pt")

    batches, _ = cache._screened_batches(output, prepared)
    assert batches == [output / "batch_0.pt"]
    with pytest.raises(FileExistsError, match="fresh screening output"):
        _screen(screen, monkeypatch, prepared, output)
    np.save(prepared / "candidates_1.npy", np.ones((32, 47), np.float32))
    with pytest.raises(ValueError, match="different prepared assets"):
        cache._screened_batches(output, prepared)


def test_wuji_evaluation_rejects_repeated_seeds_before_writing(
    load_example, monkeypatch, tmp_path: Path
) -> None:
    evaluate = load_example("wuji_inhand", "evaluate")
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            *("--checkpoint", str(tmp_path / "checkpoint.pt")),
            *("--upstream", str(tmp_path / "upstream")),
            *("--assets", str(tmp_path / "assets")),
            *("--output", str(output)),
            *("--seeds", "91001", "91002", "91001"),
        ],
    )

    with pytest.raises(ValueError, match="distinct"):
        evaluate.main()
    assert not output.exists()
