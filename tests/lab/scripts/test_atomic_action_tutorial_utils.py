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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodichain.lab.sim import SimulationManager
from scripts.tutorials.atomic_action.tutorial_utils import run_tutorial

pytestmark = pytest.mark.no_sim


def _patch_manager(
    monkeypatch: pytest.MonkeyPatch, sim: object
) -> tuple[MagicMock, MagicMock]:
    reset = MagicMock()
    flush_cleanup_queue = MagicMock()
    monkeypatch.setattr(
        SimulationManager,
        "is_instantiated",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        SimulationManager,
        "get_instance",
        classmethod(lambda cls: sim),
    )
    monkeypatch.setattr(
        SimulationManager,
        "reset",
        classmethod(lambda cls, instance_id=0: reset(instance_id)),
    )
    monkeypatch.setattr(
        SimulationManager,
        "flush_cleanup_queue",
        staticmethod(flush_cleanup_queue),
    )
    return reset, flush_cleanup_queue


def test_run_tutorial_releases_interrupted_locals_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sim = MagicMock(_is_constructed=True)
    sim.is_window_recording.return_value = False
    sim.wait_window_record_saves.side_effect = lambda: events.append("wait")
    sim.destroy.side_effect = lambda **_: events.append("destroy")
    _, flush_cleanup_queue = _patch_manager(monkeypatch, sim)
    flush_cleanup_queue.side_effect = lambda: events.append("flush")

    class BorrowedNativeWrapper:
        def __del__(self) -> None:
            events.append("borrower-released")

    def interrupt_with_live_wrapper() -> None:
        borrowed_wrapper = BorrowedNativeWrapper()
        assert borrowed_wrapper is not None
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as interrupted:
        run_tutorial(interrupt_with_live_wrapper)

    assert interrupted.value.code == 130
    assert events == ["borrower-released", "wait", "destroy", "flush"]


def test_run_tutorial_stops_recording_before_normal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = MagicMock(_is_constructed=True)
    sim.is_window_recording.return_value = True
    _, flush_cleanup_queue = _patch_manager(monkeypatch, sim)

    run_tutorial(lambda: None)

    sim.stop_window_record.assert_called_once_with()
    sim.wait_window_record_saves.assert_called_once_with()
    sim.destroy.assert_called_once_with(exit_process=False)
    flush_cleanup_queue.assert_called_once_with()


def test_run_tutorial_resets_partially_constructed_manager_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = SimpleNamespace(_is_constructed=False, instance_id=0)
    reset, flush_cleanup_queue = _patch_manager(monkeypatch, sim)

    def interrupt_during_construction() -> None:
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as interrupted:
        run_tutorial(interrupt_during_construction)

    assert interrupted.value.code == 130
    reset.assert_called_once_with(0)
    flush_cleanup_queue.assert_not_called()


def test_run_tutorial_preserves_non_interrupt_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = SimpleNamespace(_is_constructed=False, instance_id=0)
    reset, flush_cleanup_queue = _patch_manager(monkeypatch, sim)

    def fail_during_construction() -> None:
        raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        run_tutorial(fail_during_construction)

    reset.assert_called_once_with(0)
    flush_cleanup_queue.assert_not_called()
