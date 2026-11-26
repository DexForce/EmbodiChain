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

"""Policy catalog preflight and the run receipt lifecycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_RP1 = {
    "robot": "robots/rp1@1",
    "policy": "policies/carry@1",
    "scene": "scenes/platform@1",
    "task": "tasks/carry@1",
}
_ANYMAL = {
    "robot": "robots/anymal@1",
    "policy": "policies/velocity@1",
    "scene": "scenes/ground@1",
    "task": "tasks/walk@1",
}
_SESSION = "dsws_" + "a" * 32


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.json"
    data = {
        "profiles": [
            {
                **spec,
                "entrypoint": "trusted_example:run",
                "files": {"policy": "checkpoint"},
                "world": {"backend": backend, "dt": 0.005},
            }
            for spec, backend in ((_RP1, "newton"), (_ANYMAL, "dexsim"))
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_prepare_validates_composition_without_a_world(
    catalog: Path, tmp_path: Path
) -> None:
    pytest.importorskip("mcp.server.mcpserver")
    from embodichain.mcp._policy_runs import ToolError, _PolicyRuns

    runs = _PolicyRuns(catalog, tmp_path / "runs", str(tmp_path), None)
    with runs.worker:
        with pytest.raises(ToolError, match="Missing policy resource"):
            runs.prepare(_RP1)
        (tmp_path / "checkpoint").touch()
        plan = runs.prepare(_RP1)
        assert plan == runs.prepare(_RP1)
        assert runs.prepare(_ANYMAL)["plan_id"] != plan["plan_id"]
        assert plan["world"] == {"backend": "newton", "dt": 0.005}
        with pytest.raises(ToolError, match="incompatible"):
            runs.prepare({**_RP1, "policy": _ANYMAL["policy"]})


def test_duplicate_start_recovers_the_terminal_receipt(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mcp.server.mcpserver")
    from embodichain.mcp import _policy_runs

    (tmp_path / "weights").touch()
    (tmp_path / "checkpoint").symlink_to(tmp_path / "weights")
    session = MagicMock()
    session.__enter__.return_value = session
    session.info = {"config": {"backend": "newton", "dt": 0.005}}
    session.invoke.return_value = {"passed": True, "rollout_steps": 100}
    connect = MagicMock(return_value=session)
    monkeypatch.setattr(_policy_runs.Session, "connect", connect)
    output = tmp_path / "runs"
    runs = _policy_runs._PolicyRuns(catalog, output, str(tmp_path), _SESSION)
    with runs.worker:
        plan = runs.prepare(_RP1)
        with pytest.raises(_policy_runs.ToolError, match="another DexSim Session"):
            runs.start(plan["plan_id"], "dsws_" + "b" * 32)
        first = runs.start(plan["plan_id"], _SESSION)
    assert first["status"] == "queued"

    again = _policy_runs._PolicyRuns(catalog, output, str(tmp_path), _SESSION)
    with again.worker:
        (tmp_path / "checkpoint").unlink()
        receipt = again.start(plan["plan_id"], _SESSION)
        assert asyncio.run(again.get(first["run_id"], 0)) == receipt
        assert receipt["done"] and receipt["task_passed"]
        assert receipt["status"] == "succeeded"
        assert receipt["result"]["rollout_steps"] == 100
        with pytest.raises(
            _policy_runs.ToolError, match="already has an assigned policy run"
        ):
            other = again.profiles[
                tuple(_ANYMAL[field] for field in _policy_runs._FIELDS)
            ]
            again.start(other["plan_id"], _SESSION)
    assert connect.call_count == 1
    assert session.invoke.call_count == 1
    assert session.invoke.call_args.args == (
        "embodichain.lab.sim.dexsim_session._policy_run",
    )
    assert session.invoke.call_args.kwargs["files"]["policy"] == str(
        tmp_path / "checkpoint"
    )


def test_concurrent_starts_share_one_committed_receipt(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    pytest.importorskip("mcp.server.mcpserver")
    from embodichain.mcp import _policy_runs

    (tmp_path / "checkpoint").touch()
    session = MagicMock()
    session.__enter__.return_value = session
    session.info = {"config": {"backend": "newton", "dt": 0.005}}
    session.invoke.return_value = {"passed": True}
    connect = MagicMock(return_value=session)
    monkeypatch.setattr(_policy_runs.Session, "connect", connect)

    # Both services must reach initial receipt publication before either writes.
    # With an early directory claim the second service fails to read the receipt.
    barrier = Barrier(2)
    write = _policy_runs._write_receipt

    def write_receipt(path: Path, receipt: dict) -> None:
        if receipt["status"] == "queued":
            barrier.wait(timeout=5)
        write(path, receipt)

    monkeypatch.setattr(_policy_runs, "_write_receipt", write_receipt)
    output = tmp_path / "runs"
    first = _policy_runs._PolicyRuns(catalog, output, None, _SESSION)
    second = _policy_runs._PolicyRuns(catalog, output, None, _SESSION)
    plan = first.prepare(_RP1)["plan_id"]
    with first.worker, second.worker, ThreadPoolExecutor(max_workers=2) as clients:
        a = clients.submit(first.start, plan, _SESSION)
        b = clients.submit(second.start, plan, _SESSION)
        assert a.result(timeout=10)["run_id"] == b.result(timeout=10)["run_id"]
    assert asyncio.run(second.get(a.result()["run_id"], 0))["status"] == "succeeded"
    connect.assert_called_once()
