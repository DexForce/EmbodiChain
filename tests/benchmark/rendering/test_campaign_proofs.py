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

"""Simulator-free multi-proof routing, cancellation and explicit retry contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.rendering import campaign_proofs as proofs
from scripts.benchmark.rendering import routing_diagnostic
from scripts.benchmark.core import execution
from scripts.benchmark.core.artifacts import stable_hash


def _proof_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], list[Path]]:
    root = tmp_path / "campaign"
    root.mkdir()
    definition = {
        "source_commit": "frozen",
        "proof_root": str(tmp_path / "evidence"),
        "budget": {"maximum_worker_runs": 8},
        "history_timeout_s": 3600,
    }
    state: dict[str, Any] = {"jobs": [], "reserved_worker_runs": 0}
    monkeypatch.setattr(proofs, "_load", lambda root: (definition, state))
    monkeypatch.setattr(proofs, "_verify_execution_source", lambda *args: proofs.ROOT)
    monkeypatch.setattr(
        proofs, "storage_gate", lambda *args, **kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        proofs, "rebuild_campaign_report", lambda root: root / "report.md"
    )
    requests = [
        {
            "kind": "routing",
            "run_id": f"run-{index}",
            "backend": "isaaclab",
            "experiment": "original",
            "phase": "coverage",
            "source_commit": "frozen",
            "case": {"width": 128, "height": 128, "temporal_mode": "static"},
        }
        for index in range(2)
    ]
    monkeypatch.setattr(proofs, "proof_population", lambda *args: requests)
    paths: list[Path] = []

    def prepare(source, commit, experiment, run_id, output, **kwargs):
        output.mkdir(parents=True, exist_ok=False)
        paths.append(output)
        return {"output": output, "command": ["worker"], "environment": {}}

    monkeypatch.setattr(routing_diagnostic, "prepare_routing_diagnostic", prepare)
    monkeypatch.setattr(
        execution, "execute_worker", lambda *args, **kwargs: {"status": "completed"}
    )
    return root, state, paths


def test_each_routing_proof_has_a_distinct_frozen_identity_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state, paths = _proof_campaign(tmp_path, monkeypatch)
    proofs.run_proofs(root, kind="routing")
    assert len(set(paths)) == 2
    assert all(p["status"] == "completed" for p in state["proofs"])
    for proof, output in zip(state["proofs"], paths):
        assert output.parent.name == proof["proof_id"]
        assert output.name == "attempt-000"
    assert state["reserved_worker_runs"] == 2


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_explicit_retry_preserves_failed_proof_and_creates_fresh_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    root, state, paths = _proof_campaign(tmp_path, monkeypatch)
    launches = []

    def execute(*args, **kwargs):
        launches.append(args)
        if len(launches) == 1:
            raise failure
        return {"status": "completed"}

    monkeypatch.setattr(execution, "execute_worker", execute)
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            proofs.run_proofs(root, kind="routing", run_ids=("run-0",))
    else:
        proofs.run_proofs(root, kind="routing", run_ids=("run-0",))
    previous = dict(state["proofs"][0])
    proofs.run_proofs(root, kind="routing", run_ids=("run-0",))
    assert len(launches) == 1
    proofs.run_proofs(root, kind="routing", run_ids=("run-0",), retry_failed=True)
    assert state["proofs"][0] == previous
    assert state["proofs"][1]["status"] == "completed"
    assert paths[0] != paths[1] and paths[1].name == "attempt-001"
    assert state["reserved_worker_runs"] == 2


def test_unfiltered_proof_retry_is_rejected_before_reserving_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state, _ = _proof_campaign(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit original-run selection"):
        proofs.run_proofs(root, kind="routing", retry_failed=True)
    assert state["reserved_worker_runs"] == 0


def test_pending_legacy_proof_records_the_actual_resuming_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state, _ = _proof_campaign(tmp_path, monkeypatch)
    request = proofs.proof_population(root, "routing")[0]
    state["proofs"] = [
        {
            **request,
            "proof_id": stable_hash(request),
            "evaluation_attempt": 0,
            "status": "not_run",
            "reason": "disk_headroom",
        }
    ]
    monkeypatch.setattr(proofs, "git_revision", lambda root: "corrected-driver")
    monkeypatch.setattr(proofs, "_source_files", lambda: {"driver.py": "current"})
    proofs.run_proofs(root, kind="routing")
    assert all(p["status"] == "completed" for p in state["proofs"])
    assert all(p["driver_source_commit"] == "corrected-driver" for p in state["proofs"])
    assert all(
        p["driver_source_files"] == {"driver.py": "current"} for p in state["proofs"]
    )
