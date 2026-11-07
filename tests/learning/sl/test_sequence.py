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

"""Owned causal encoding and source-aware exact recovery without a producer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from embodichain.data_pipeline.datasets import (
    DataBudgetExhaustedError,
    DataUnavailableError,
    SequenceSpecCfg,
    TensorDataSource,
)
from embodichain.learning.sl import (
    SequenceBatchSource,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)
from .test_distributed import _assert_tree_equal, _snapshot, _Regression

# The repository's tests/ tree is not a Python package. Load the existing
# simulator-free source fixture by its path without changing global sys.path.
_fixture_spec = importlib.util.spec_from_file_location(
    "_sl_causal_source_fixtures",
    Path(__file__).resolve().parents[2] / "data_pipeline/test_sources.py",
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixtures)
_data, _SCHEMA, paired_sources = (
    _fixtures._data,
    _fixtures._SCHEMA,
    _fixtures.paired_sources,
)


def _encode(context):
    return SupervisedBatch(
        context["obs"][:, :2],
        context["actions"][:, 1:],
        {"target": context["loss_mask"][:, 1:]},
        {"recipe": "explicit-history-to-actions"},
    )


def _fit_encode(context):
    return SupervisedBatch(context["obs"][:, 1], context["actions"][:, 1])


def _resident(*, schema=None, data=None, metadata=None):
    return TensorDataSource(
        _data() if data is None else data,
        schema=copy.deepcopy(_SCHEMA if schema is None else schema),
        source_id="same-manual-dataset-id",
        metadata=metadata,
    )


def _build_trainer(
    path: Path,
    *,
    source=None,
    encoder_version="toy-current-regression-v1",
    sequence=None,
    immutable=True,
):
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1)
    )
    bridge = SequenceBatchSource(
        _resident() if source is None else source,
        SequenceSpecCfg(2, 2) if sequence is None else sequence,
        encoder=_fit_encode,
        encoder_version=encoder_version,
        immutable=immutable,
    )
    return SupervisedTrainer(
        SupervisedLearner(
            model, _Regression(), torch.optim.Adam(model.parameters(), lr=0.01)
        ),
        bridge,
        SupervisedTrainerCfg(
            total_updates=4, batch_size=3, output_dir=str(path), checkpoint_interval=0
        ),
        signature={"semantics": "toy-regression-only"},
    )


@pytest.mark.parametrize("mode,generations", [("episode", None), ("segment", (2,))])
def test_live_and_resident_encoding_preserves_windows_masks_and_owned_provenance(
    paired_sources, mode, generations
):
    live, resident = paired_sources
    sequence = SequenceSpecCfg(2, 4, mode, generations)
    bridges = [
        SequenceBatchSource(
            source, sequence, encoder=_encode, encoder_version="toy-actions-v1"
        )
        for source in (live, resident)
    ]
    sequence.prediction_horizon = 1
    expected = resident.sample_context(
        32,
        SequenceSpecCfg(2, 4, mode, generations),
        generator=torch.Generator().manual_seed(91),
    )
    outputs = [
        bridge.sample(32, generator=torch.Generator().manual_seed(91))
        for bridge in bridges
    ]
    for output in outputs:
        torch.testing.assert_close(output.inputs, expected["obs"][:, :2])
        torch.testing.assert_close(output.targets, expected["actions"][:, 1:])
        torch.testing.assert_close(output.masks["target"], expected["loss_mask"][:, 1:])
        metadata = output.metadata["sequence_source"]
        assert metadata["sequence"]["prediction_horizon"] == 4
        assert metadata["schema"] == _SCHEMA
        assert "obs" not in metadata["context"] and "actions" not in metadata["context"]
        for key in expected.exclude("obs", "actions").keys():
            torch.testing.assert_close(metadata["context"][key], expected[key])
    saved = outputs[0].inputs.clone()
    bridges[0].sample(32, generator=torch.Generator().manual_seed(13)).inputs.fill_(
        -1000
    )
    torch.testing.assert_close(outputs[0].inputs, saved)
    outputs[0].inputs.fill_(-1000)
    outputs[1].targets.fill_(-1000)
    outputs[0].metadata["sequence_source"]["context"]["frame_ids"].fill_(-1000)
    assert not (live.engine.shared_buffer["obs"] == -1000).any()
    assert not (live.engine.shared_buffer["ods_frame_index"] == -1000).any()
    assert not (resident.data["actions"] == -1000).any()


@pytest.mark.parametrize(
    "error_type", [DataUnavailableError, DataBudgetExhaustedError, RuntimeError]
)
def test_source_errors_are_propagated_once_without_retry(monkeypatch, error_type):
    source = _resident()
    bridge = SequenceBatchSource(
        source, SequenceSpecCfg(2, 2), encoder=_encode, encoder_version="v1"
    )
    error = error_type("terminal delivery failure")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(source, "sample_context", fail)
    with pytest.raises(error_type) as captured:
        bridge.sample(2, generator=torch.Generator())
    assert captured.value is error and calls == [1]


@pytest.mark.parametrize("reserved", [False, True])
def test_encoding_failure_does_not_refund_delivery_quota(reserved):
    source = _resident()
    source.configure_sampling_budget(2, 2)
    calls = []

    def encode(context):
        calls.append(1)
        if reserved:
            return SupervisedBatch(
                context["obs"], context["actions"], metadata={"sequence_source": {}}
            )
        raise ValueError("encoding failed")

    bridge = SequenceBatchSource(
        source, SequenceSpecCfg(2, 2), encoder=encode, encoder_version="failing-v1"
    )
    with pytest.raises(
        ValueError, match="reserved sequence_source" if reserved else "encoding failed"
    ):
        bridge.sample(3, generator=torch.Generator().manual_seed(5))
    assert calls == [1]
    assert source.get_sampling_budget_snapshot()["total_delivered_chunks"] == 3
    with pytest.raises(DataBudgetExhaustedError):
        bridge.sample(2, generator=torch.Generator().manual_seed(5))
    assert calls == [1]
    assert source.get_sampling_budget_snapshot()["total_delivered_chunks"] == 3


@pytest.mark.parametrize(
    "changed", ["schema", "encoder", "sequence", "corpus", "language"]
)
def test_resume_identity_changes_fail_before_mutating_any_trainer_state(
    tmp_path, changed
):
    metadata = {"language": {"task": {0: "pick"}}}
    original = _build_trainer(
        tmp_path / "original", source=_resident(metadata=metadata)
    )
    original.fit(stop_after=2)
    schema, data = copy.deepcopy(_SCHEMA), _data()
    kwargs = {}
    if changed == "schema":
        schema["action"]["semantics"] = "joint_position"
    elif changed == "encoder":
        kwargs["encoder_version"] = "changed-v2"
    elif changed == "sequence":
        kwargs["sequence"] = SequenceSpecCfg(2, 3)
    elif changed == "corpus":
        data["actions"][0, 0] += 1
    else:
        metadata["language"]["task"][0] = "release"
    candidate = _build_trainer(
        tmp_path / "candidate",
        source=_resident(schema=schema, data=data, metadata=metadata),
        **kwargs,
    )
    before = _snapshot(candidate)
    run_id = candidate._artifacts.run_id
    with pytest.raises(ValueError, match="identity"):
        candidate.resume(original.output / "last.pt")
    _assert_tree_equal(_snapshot(candidate), before)
    assert not candidate.output.exists()
    assert candidate._artifacts.run_id == run_id
    assert not candidate._artifacts._claimed


def test_storage_diagnostics_can_change_without_changing_exact_recovery(tmp_path):
    def source(storage):
        return _resident(
            metadata={"storage": storage, "language": {"task": {0: "pick"}}}
        )

    original = _build_trainer(
        tmp_path / "original",
        source=source({"root": "/old", "numeric_read_batch_size": 1}),
    )
    original.fit(stop_after=2)
    candidate = _build_trainer(
        tmp_path / "candidate",
        source=source({"root": "/moved", "numeric_read_batch_size": 19}),
    )
    assert original.source.source_id == candidate.source.source_id
    candidate.resume(original.output / "last.pt")
    _assert_tree_equal(_snapshot(candidate), _snapshot(original))
    candidate.fit()
    original.resume(original.output / "last.pt")
    original.fit()
    _assert_tree_equal(_snapshot(candidate), _snapshot(original))
    diagnostic = candidate.source.sample(1, generator=torch.Generator()).metadata[
        "sequence_source"
    ]["storage"]
    assert diagnostic["root"] == "/moved"


@pytest.mark.parametrize("kind", ["live", "quota", "unverified"])
def test_live_quota_and_unverified_snapshots_reject_resume_without_mutation(
    tmp_path, paired_sources, kind
):
    live, _ = paired_sources
    source = live if kind == "live" else _resident()
    if kind == "quota":
        source.configure_sampling_budget(10, 10)
    trainer = _build_trainer(tmp_path / kind, source=source, immutable=False)
    trainer.fit(stop_after=1)
    before = _snapshot(trainer)
    files = {path.name: path.read_bytes() for path in trainer.output.iterdir()}
    quota = source.get_sampling_budget_snapshot() if kind == "quota" else None
    with pytest.raises(ValueError, match="Exact resume.*immutable"):
        trainer.resume(trainer.output / "last.pt")
    _assert_tree_equal(_snapshot(trainer), before)
    assert files == {path.name: path.read_bytes() for path in trainer.output.iterdir()}
    if quota is not None:
        assert source.get_sampling_budget_snapshot() == quota


@pytest.mark.parametrize(
    "changed", ["quota", "source_id", "schema", "language", "tensor"]
)
def test_immutable_promise_is_checked_after_construction(changed):
    source = _resident(metadata={"language": {"task": {0: "pick"}}})
    bridge = SequenceBatchSource(
        source,
        SequenceSpecCfg(2, 2),
        encoder=_fit_encode,
        encoder_version="v1",
        immutable=True,
    )
    if changed == "quota":
        source.configure_sampling_budget(2, 2)
    elif changed == "source_id":
        source.source_id = "changed"
    elif changed == "schema":
        source.schema["action"]["step_dt"] = 0.1
    elif changed == "language":
        source.metadata["language"]["task"][0] = "release"
    else:
        source.data["obs"][0, 0] += 1
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="changed|unbudgeted"):
        bridge.sample(1, generator=generator)
    torch.testing.assert_close(generator.get_state(), before)
    with pytest.raises(ValueError, match="changed|unbudgeted"):
        bridge.validate_resume()


def test_resume_rehash_detects_untracked_tensor_writes():
    source = _resident()
    bridge = SequenceBatchSource(
        source,
        SequenceSpecCfg(2, 2),
        encoder=_fit_encode,
        encoder_version="v1",
        immutable=True,
    )
    source.data["actions"].data[0, 0] += 1
    with pytest.raises(ValueError, match="content changed"):
        bridge.validate_resume()


def test_immutable_content_hash_handles_noncontiguous_large_cpu_features():
    from tensordict import TensorDict

    values = (
        torch.arange(2 * 3 * 100003, dtype=torch.float32)
        .reshape(2, 100003, 3)
        .transpose(1, 2)
    )
    assert not values.is_contiguous()
    data = TensorDict(
        {
            "obs": values,
            "actions": torch.ones(2, 3, 1),
            "valid": torch.ones(2, 3, dtype=torch.bool),
        },
        [2, 3],
    )
    schema = {"action": {"semantics": "toy", "step_dt": 0.1}}

    def wrap(payload):
        return SequenceBatchSource(
            TensorDataSource(
                payload, schema=schema, source_id="large", copy_data=False
            ),
            SequenceSpecCfg(1, 1),
            encoder=_fit_encode,
            encoder_version="v1",
            immutable=True,
        )

    first = wrap(data)
    second = wrap(data.contiguous())
    assert first.source_id == second.source_id
    first.validate_resume()
