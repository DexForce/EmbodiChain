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
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark.rendering import campaign_proofs as proofs
from scripts.benchmark.rendering import routing_diagnostic
from scripts.benchmark.core import execution
from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.core.artifacts import write_json


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


def test_fhd_history_admission_counts_distinct_states_and_both_precision_rungs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import evaluate
    from scripts.benchmark.rendering import reference

    descriptors = [
        {
            "metadata": {
                "scene_sha256": "scene",
                "world_from_objects": {"state": i % 100},
            },
            "canonical_camera": {"name": "camera", "intrinsics": [1, 2, 3]},
        }
        for i in range(301)
    ]
    run = {"descriptors": descriptors, "case": SimpleNamespace(width=1920, height=1080)}
    monkeypatch.setattr(evaluate, "_load_run", lambda *args: run)
    monkeypatch.setattr(
        reference, "reference_dependencies", lambda: {"versions": {"mitsuba": "frozen"}}
    )
    row = {"config": {"temporal_mode": "moving"}}
    cache = tmp_path / "cache"
    reserve = proofs._reference_reservation(tmp_path, row, cache)
    # 100 physical states, both rungs, potential cache copies plus diagnostics.
    expected = 512 * 1024**2 + 100 * 2 * 2 * (32 * 1920 * 1080 + 1024**2)
    assert reserve == expected
    assert reserve > 256 * 1024**2
    keys = []
    for descriptor in descriptors[:100]:
        for spp in (4096, 16384):
            keys.append(
                stable_hash(
                    evaluate._reference_identity(
                        run, descriptor, {"spp": spp, "threads": 8, "seed": 0}
                    )
                )
            )
    # Locally present immutable entries need no new array allocation. Evaluation
    # separately verifies completeness/checksums before consuming them.
    for key in keys:
        (tmp_path / "references" / key).mkdir(parents=True)
    assert proofs._reference_reservation(tmp_path, row, cache) == (512 + 200) * 1024**2
    for key in keys:
        (cache / "references" / key).mkdir(parents=True)
    assert proofs._reference_reservation(tmp_path, row, cache) == 512 * 1024**2


def test_shared_reference_admission_reserves_actual_possible_copy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import evaluate

    run = {"descriptors": [{}], "case": SimpleNamespace(width=128, height=128)}
    monkeypatch.setattr(evaluate, "_load_run", lambda *args: run)
    monkeypatch.setattr(
        evaluate, "_reference_identity", lambda run, descriptor, settings: settings
    )
    cache = tmp_path / "cache"
    for spp in (4096, 16384):
        key = stable_hash({"spp": spp, "threads": 8, "seed": 0})
        directory = cache / "references" / key
        directory.mkdir(parents=True)
        (directory / "arrays.npz").write_bytes(b"x" * 64)
    reserve = proofs._reference_reservation(
        tmp_path, {"config": {"temporal_mode": "static"}}, cache
    )
    assert reserve == 512 * 1024**2 + 2 * (64 + 1024**2)


def test_cpu_proof_stays_not_run_when_complete_reference_population_cannot_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state, _ = _proof_campaign(tmp_path, monkeypatch)
    request = {**proofs.proof_population(root, "routing")[0], "kind": "quality"}
    monkeypatch.setattr(proofs, "proof_population", lambda *args: [request])
    experiment = root / "original"
    experiment.mkdir()
    write_json(experiment / "runs.json", [{"run_id": "run-0"}])
    cold_bytes = 12 * 1024**3
    admitted = []
    monkeypatch.setattr(proofs, "_reference_reservation", lambda *args: cold_bytes)

    def gate(definition, **kwargs):
        admitted.append(kwargs)
        return {"passed": False, "reason": "evidence_budget"}

    monkeypatch.setattr(proofs, "storage_gate", gate)
    proofs.run_proofs(root, kind="quality")
    proof = state["proofs"][0]
    assert admitted[0]["reservation_bytes"] == cold_bytes
    assert proof["status"] == "not_run" and proof["reason"] == "evidence_budget"
    assert proof["storage_reservation_bytes"] == cold_bytes
    assert state["reserved_worker_runs"] == 0


def test_history_checks_acquisition_and_import_destinations_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state, _ = _proof_campaign(tmp_path, monkeypatch)
    request = {**proofs.proof_population(root, "routing")[0], "kind": "history"}
    request["case"]["modalities"] = ["rgb"]
    request["case"]["num_envs"] = request["case"]["cameras_per_env"] = 1
    monkeypatch.setattr(proofs, "proof_population", lambda *args: [request])
    checked = []

    def gate(definition, **kwargs):
        checked.append(kwargs["destination"])
        passed = kwargs["destination"] != root / "original"
        return {"passed": passed, "reason": None if passed else "disk_headroom"}

    monkeypatch.setattr(proofs, "storage_gate", gate)
    proofs.run_proofs(root, kind="history")
    assert len(checked) == 2 and checked[-1] == root / "original"
    assert state["proofs"][0]["reason"] == "disk_headroom"
    assert state["proofs"][0]["status"] == "not_run"
    assert state["reserved_worker_runs"] == 0


def test_history_rechecks_import_headroom_after_acquisition_and_retains_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import history_acquisition, history_binding

    root, state, _ = _proof_campaign(tmp_path, monkeypatch)
    request = {**proofs.proof_population(root, "routing")[0], "kind": "history"}
    request["case"].update(modalities=["rgb"], num_envs=1, cameras_per_env=1)
    monkeypatch.setattr(proofs, "proof_population", lambda *args: [request])
    acquisition = tmp_path / "acquired"
    acquisition.mkdir()
    artifact = acquisition / "history.json"
    write_json(artifact, {"evidence": "retained"})
    result = {"status": "completed", "root": str(acquisition), "history": str(artifact)}
    monkeypatch.setattr(
        history_acquisition, "acquire_history", lambda *args, **kwargs: result
    )
    imported = []
    monkeypatch.setattr(
        history_binding, "import_history_proof", lambda *args: imported.append(args)
    )
    gates = []

    def gate(definition, **kwargs):
        gates.append(kwargs)
        return {"passed": len(gates) < 3, "reason": "disk_headroom"}

    monkeypatch.setattr(proofs, "storage_gate", gate)
    proofs.run_proofs(root, kind="history")
    proof = state["proofs"][0]
    assert len(gates) == 3 and gates[-1]["destination"] == root / "original"
    assert (
        proof["status"] == "failed" and "history_import_storage_gate" in proof["reason"]
    )
    assert proof["acquisition"] == result and artifact.exists()
    assert not imported
