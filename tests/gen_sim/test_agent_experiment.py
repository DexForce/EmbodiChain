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

"""A disconnected authoring client can reconcile work without restoring physical state."""

from __future__ import annotations
from pathlib import Path
import json
from unittest.mock import Mock
import pytest
from embodichain.gen_sim import agent_experiment as pilot


def test_immutable_revisions_and_lost_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = pilot.ExperimentStore(tmp_path)
    initial = store.create("test", "initial")
    run, rev = initial["run_id"], initial["revision"]
    assert store.create("test", "initial")["revision"] == rev
    with pytest.raises(ValueError, match="Stale"):
        store.validate(run, "stale")
    corrected = store.revise(run, rev, 0.0, "change")["revision"]
    assert store.revise(run, rev, 0.0, "change")["revision"] == corrected
    with pytest.raises(ValueError, match="conflict"):
        store.revise(run, rev, 0.01, "change")
    assert store.validate(run, corrected)["physical_verdict"] == "unavailable"
    launch = Mock()
    monkeypatch.setattr(pilot.subprocess, "Popen", launch)
    before = store.submit(run, corrected, "execute")
    # Simulate a worker exit without a terminal receipt: its inherited OS lock is gone.
    after = pilot.ExperimentStore(tmp_path).submit(run, corrected, "execute")
    assert before["job_id"] == after["job_id"] and after["state"] == "unknown"
    launch.assert_called_once()
    with pytest.raises(ValueError, match="conflict"):
        store.submit(run, rev, "execute")
    assert store.submit(run, corrected, "new-attempt")["attempt_id"] == 2
    with pytest.raises(ValueError, match="budget"):
        store.submit(run, corrected, "over-budget")
    assert initial["goal"] == store.inspect(run)["goal"]
    assert (tmp_path / run / "bundles" / rev / "manifest.json").is_file()


def test_wall_budget_and_changed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = pilot.ExperimentStore(tmp_path)
    r = store.create("test", "initial")
    run, rev = r["run_id"], r["revision"]
    monkeypatch.setattr(pilot.time, "time", lambda: r["deadline"] + 1)
    with pytest.raises(ValueError, match="budget"):
        store.submit(run, rev, "late")
    folder = tmp_path / run / "bundles" / rev
    manifest = json.loads((folder / "manifest.json").read_text())
    (folder / manifest["entry"]).write_text("id: changed")
    with pytest.raises(ValueError, match="Changed sealed"):
        store.validate(run, rev)


@pytest.mark.gpu
def test_cancel_reconciles_at_owner_boundary(tmp_path: Path) -> None:
    import time

    store = pilot.ExperimentStore(tmp_path)
    r = store.create("cancel-test", "initial")
    rid, rev = r["run_id"], r["revision"]
    j = store.submit(rid, rev, "execution")
    assert store.submit(rid, rev, "execution")["job_id"] == j["job_id"]
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        observed = store.inspect(rid)["attempts"][-1]
        if observed.get("session_id"):
            break
        if observed["state"] not in ("running", "queued"):
            pytest.fail(str(observed))
        time.sleep(0.2)
    assert observed.get("session_id")
    store.cancel(rid, j["job_id"])
    while time.monotonic() < deadline:
        observed = store.inspect(rid)["attempts"][-1]
        if observed["state"] == "cancelled":
            break
        time.sleep(0.2)
    assert (
        observed["state"] == "cancelled"
        and observed["physical_verdict"] == "unavailable"
    )
    assert not list(Path(observed["session_directory"]).glob("dsws_*.json"))
