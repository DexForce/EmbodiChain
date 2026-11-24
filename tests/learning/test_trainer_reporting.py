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
"""Behavioral coverage for shared native and custom scalar sinks."""

from __future__ import annotations

import builtins
import random
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from embodichain.learning.rl._trainer_reporting import (
    _write_scalars,
    _write_tensorboard_scalars,
)


def _records(directory: Path) -> list[tuple[int, str, bytes]]:
    from tensorboard.backend.event_processing.event_file_loader import (
        LegacyEventFileLoader,
    )

    records = []
    for path in sorted(directory.glob("events.out.tfevents.*")):
        for event in LegacyEventFileLoader(str(path)).Load():
            for value in event.summary.value:
                assert value.WhichOneof("value") == "simple_value"
                records.append((event.step, value.tag, value.SerializeToString()))
    return records


def test_native_scalar_records_match_per_tag_encoding(tmp_path: Path) -> None:
    metrics = {
        "nan": float("nan"),
        "inf": float("inf"),
        "negative_inf": -float("inf"),
        "negative_zero": -0.0,
        "rounding": 16777217.0,
        "overflow": 1e40,
        "unicode/数值": 1 / 3,
    }
    with SummaryWriter(str(tmp_path / "baseline")) as writer:
        for tag, value in metrics.items():
            writer.add_scalar(tag, value, 2**40)
    with SummaryWriter(str(tmp_path / "candidate")) as writer:
        _write_tensorboard_scalars(writer, {}, 0)
        _write_tensorboard_scalars(writer, metrics, 2**40)
    assert _records(tmp_path / "candidate") == _records(tmp_path / "baseline")


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_invalid_scalar_preserves_prefix_and_error_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_errors: bool
) -> None:
    metrics = {"first": 1.0, "invalid\ud800": 2.0, "last": 3.0}
    error_type = None
    with SummaryWriter(str(tmp_path / "baseline")) as writer:
        for tag, value in metrics.items():
            try:
                writer.add_scalar(tag, value, 7)
            except Exception as error:
                if not ignore_errors:
                    error_type = type(error)
                    break
    wandb_log = Mock()
    monkeypatch.setattr(
        "embodichain.learning.rl._trainer_reporting.wandb.log", wandb_log
    )
    with SummaryWriter(str(tmp_path / "candidate")) as writer:
        if ignore_errors:
            _write_tensorboard_scalars(writer, metrics, 7, ignore_errors=True)
        else:
            assert error_type is not None
            with pytest.raises(error_type):
                _write_scalars(metrics, step=7, writer=writer, use_wandb=True)
            wandb_log.assert_not_called()
    assert _records(tmp_path / "candidate") == _records(tmp_path / "baseline")


@pytest.mark.parametrize("writer_kind", ["custom", "subclass", "instance_override"])
def test_writer_overrides_keep_scalar_calls_without_tensorboard_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer_kind: str
) -> None:
    calls = []

    class CustomWriter:
        def add_scalar(self, tag, value, step):
            calls.append((tag, value, step))

    class Writer(SummaryWriter):
        def add_scalar(self, tag, value, step):
            calls.append((tag, value, step))

    if writer_kind == "custom":
        writer = CustomWriter()
    elif writer_kind == "subclass":
        writer = Writer(str(tmp_path))
    else:
        writer = SummaryWriter(str(tmp_path))
        writer.add_scalar = CustomWriter().add_scalar
    native_import = builtins.__import__

    def reject_tensorboard_import(name, *args, **kwargs):
        if name.startswith(("tensorboard", "torch.utils.tensorboard")):
            raise AssertionError("Custom writers do not require TensorBoard imports")
        return native_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", reject_tensorboard_import)
            _write_tensorboard_scalars(writer, {"first": 1.0, "last": 2.0}, 7)
        assert calls == [("first", 1.0, 7), ("last", 2.0, 7)]
    finally:
        if isinstance(writer, SummaryWriter):
            writer.close()


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_native_submission_is_not_replayed_after_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_errors: bool
) -> None:
    with SummaryWriter(str(tmp_path)) as writer:
        file_writer = writer._get_file_writer()
        original = file_writer.add_summary
        attempts = []

        def accepted_then_raised(summary, global_step=None, walltime=None):
            attempts.append(global_step)
            original(summary, global_step, walltime)
            raise RuntimeError("accepted then raised")

        monkeypatch.setattr(file_writer, "add_summary", accepted_then_raised)
        metrics = {"first": 1.0, "last": 2.0}
        if ignore_errors:
            _write_tensorboard_scalars(writer, metrics, 7, ignore_errors=True)
        else:
            with pytest.raises(RuntimeError, match="accepted then raised"):
                _write_tensorboard_scalars(writer, metrics, 7)
        assert attempts == [7]
    records = _records(tmp_path)
    assert [(step, tag) for step, tag, _ in records] == [(7, "first"), (7, "last")]


def test_scalar_reporting_preserves_host_rng(tmp_path: Path) -> None:
    with SummaryWriter(str(tmp_path)) as writer:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        _write_scalars(
            {"train/loss": 0.125, "charts/SPS": 1024.0},
            step=7,
            writer=writer,
            use_wandb=False,
        )
        assert random.getstate() == python_state
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        assert torch.equal(torch.get_rng_state(), torch_state)
