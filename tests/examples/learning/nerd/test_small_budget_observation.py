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

"""Regression coverage for original-model in-place input normalization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture(scope="module")
def workers() -> tuple[Any, Any]:
    directory = Path(__file__).resolve().parents[4] / "examples/learning/nerd/_support"
    modules = []
    for filename in ("_official_small_budget", "_matched_small_budget"):
        spec = importlib.util.spec_from_file_location(
            "_test_" + filename, directory / (filename + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _inputs() -> tuple[Any, dict[str, torch.Tensor], torch.Tensor]:
    # Nonzero mean and nonunit variance expose accidental double normalization.
    rms = SimpleNamespace(mean=torch.tensor([3.0]), var=torch.tensor([4.0]))
    model = SimpleNamespace(input_rms={"joint_f": rms})
    return model, {"joint_f": torch.tensor([[[7.0], [11.0]]])}, torch.tensor([1.5])


def test_snapshot_retains_physical_features_after_model_mutation(
    workers: tuple[Any, Any], tmp_path: Path
) -> None:
    source, _ = workers
    _, inputs, targets = _inputs()
    data = {"joint_acts": inputs["joint_f"].clone(), "target": targets.clone()}
    source._save_first_batch(tmp_path, data)
    data["joint_acts"].sub_(3.0).div_(torch.sqrt(torch.tensor(4.0 + 1e-5)))
    stored = torch.load(tmp_path / "first-preprocessed-batch.pt", weights_only=True)
    assert torch.equal(stored["joint_acts"], inputs["joint_f"])


def test_loss_observer_saves_before_original_forward_normalizes_in_place(
    workers: tuple[Any, Any], tmp_path: Path
) -> None:
    source, _ = workers
    _, inputs, targets = _inputs()
    data = {"joint_acts": inputs["joint_f"].clone(), "target": targets}
    state = {"steps": 0, "last_loss": None}
    calls = []

    def original_loss(trainer: Any, batch: dict[str, Any], train: bool) -> Any:
        calls.append(train)
        batch["joint_acts"] = (batch["joint_acts"] - 3.0) / (4.0 + 1e-5) ** 0.5
        return torch.tensor(2.0), {}

    source._observed_loss(None, data, True, original_loss, state, tmp_path)
    stored = torch.load(tmp_path / "first-preprocessed-batch.pt", weights_only=True)
    assert torch.equal(stored["joint_acts"], inputs["joint_f"])
    assert not torch.equal(data["joint_acts"], inputs["joint_f"])
    assert state["last_loss"] == 2.0 and calls == [True]


@pytest.mark.parametrize("train,step", [(False, 0), (True, 1)])
def test_loss_observer_only_captures_the_first_training_batch(
    workers: tuple[Any, Any], tmp_path: Path, train: bool, step: int
) -> None:
    source, _ = workers
    _, inputs, targets = _inputs()
    state = {"steps": step, "last_loss": None}
    source._observed_loss(
        None,
        {"joint_acts": inputs["joint_f"], "target": targets},
        train,
        lambda *_: (torch.tensor(2.0), {}),
        state,
        tmp_path,
    )
    assert not (tmp_path / "first-preprocessed-batch.pt").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_feature_comparison_respects_observation_normalization_stage(
    workers: tuple[Any, Any], tmp_path: Path, legacy: bool
) -> None:
    source, candidate = workers
    model, inputs, targets = _inputs()
    if legacy:
        rms = model.input_rms["joint_f"]
        normalized = (inputs["joint_f"] - rms.mean) / torch.sqrt(rms.var + 1e-5)
        torch.save(
            {"joint_acts": normalized, "target": targets},
            tmp_path / "first-preprocessed-batch.pt",
        )
    else:
        source._save_first_batch(
            tmp_path, {"joint_acts": inputs["joint_f"], "target": targets}
        )
    differences, stage = candidate._first_batch_differences(
        model, inputs, targets, tmp_path
    )
    assert differences == {"joint_f": 0.0, "target": 0.0}
    assert stage == (
        "legacy_unlabeled_normalized_after_model_forward"
        if legacy
        else "physical_before_model_forward"
    )


def test_feature_comparison_still_detects_changed_inputs_and_targets(
    workers: tuple[Any, Any], tmp_path: Path
) -> None:
    source, candidate = workers
    model, inputs, targets = _inputs()
    source._save_first_batch(
        tmp_path, {"joint_acts": inputs["joint_f"], "target": targets}
    )
    differences, _ = candidate._first_batch_differences(
        model, {"joint_f": inputs["joint_f"] + 2.0}, targets + 0.25, tmp_path
    )
    assert differences["joint_f"] == pytest.approx(2.0 / (4.0 + 1e-5) ** 0.5)
    assert differences["target"] == 0.25


@pytest.mark.parametrize(
    "change",
    [
        {"feature_stage": "unknown"},
        {"format_version": 2},
        {"tensor_sha256": "0" * 64},
    ],
)
def test_feature_comparison_rejects_unknown_or_unbound_observations(
    workers: tuple[Any, Any], tmp_path: Path, change: dict[str, Any]
) -> None:
    source, candidate = workers
    model, inputs, targets = _inputs()
    source._save_first_batch(
        tmp_path, {"joint_acts": inputs["joint_f"], "target": targets}
    )
    metadata = tmp_path / "first-preprocessed-batch.json"
    value = json.loads(metadata.read_text())
    value.update(change)
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="observation metadata"):
        candidate._first_batch_differences(model, inputs, targets, tmp_path)
