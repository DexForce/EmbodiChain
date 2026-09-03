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

"""Tests for the isolated Task Program bundle subprocess boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine import _bundle_runner
from embodichain.gen_sim.task_engine._bundle_runner import _exception_metadata


def test_exception_metadata_preserves_explicit_causal_chain() -> None:
    """The report retains the physical planner error hidden by demo cleanup."""
    try:
        try:
            raise ValueError("invalid coordinated trajectory")
        except ValueError as planner_error:
            raise RuntimeError("demo safe-stop completed") from planner_error
    except RuntimeError as runtime_error:
        metadata = _exception_metadata(runtime_error)

    assert metadata == {
        "type": "RuntimeError",
        "message": "demo safe-stop completed",
        "causes": [
            {
                "type": "ValueError",
                "message": "invalid coordinated trajectory",
            }
        ],
    }


def test_exception_metadata_rejects_non_exception_values() -> None:
    """The private serializer fails closed on an invalid diagnostic value."""
    with pytest.raises(TypeError, match="BaseException"):
        _exception_metadata("failure")  # type: ignore[arg-type]


def test_module_entrypoint_flushes_protocol_before_fast_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulator worker skips native interpreter-order destruction."""
    flushes: list[str] = []
    exit_codes: list[int] = []

    monkeypatch.setattr(_bundle_runner, "main", lambda: 7)
    monkeypatch.setattr(
        _bundle_runner.sys,
        "stdout",
        SimpleNamespace(flush=lambda: flushes.append("stdout")),
    )
    monkeypatch.setattr(
        _bundle_runner.sys,
        "stderr",
        SimpleNamespace(flush=lambda: flushes.append("stderr")),
    )

    def fake_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        raise SystemExit(exit_code)

    monkeypatch.setattr(_bundle_runner.os, "_exit", fake_exit)

    with pytest.raises(SystemExit, match="7"):
        _bundle_runner._module_entrypoint()

    assert flushes == ["stdout", "stderr"]
    assert exit_codes == [7]


__all__: list[str] = []
