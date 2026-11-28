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

"""Boundary plans cover declared domains without resampling failed endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from embodichain.lab.expansion._preview_job import (
    boundary_samples,
    plan_preview,
    collect_preview,
)


def test_endpoint_and_stratified_interior_sampling_is_deterministic():
    points, roles = boundary_samples(7, 5, 17)
    assert roles[:3] == ["center", "lower", "upper"]
    np.testing.assert_array_equal(points[1], np.zeros(7))
    np.testing.assert_array_equal(points[2], np.ones(7))
    assert np.all((points[3:] > 0) & (points[3:] < 1))
    for column in points[3:].T:
        assert sorted((column * 5).astype(int)) == list(range(5))
    np.testing.assert_array_equal(points, boundary_samples(7, 5, 17)[0])
    assert boundary_samples(0, 5, 17)[0].shape == (1, 0)


def test_plan_keeps_job_bounds_and_produces_explicit_motion_draws(tmp_path):
    from embodichain.lab.expansion.configured import load_expansion_job

    root = Path(__file__).resolve().parents[3]
    original = root / "examples/expansion/engine/job.motion.yaml"
    before = original.read_bytes()
    plan = plan_preview(original, tmp_path / "preview", interior=1)
    assert len(plan["records"]) == 4
    assert (
        plan["planned_attempts"] == 7
    )  # exact-zero center is one nominal, not two copies
    low = yaml.safe_load((tmp_path / "preview" / plan["records"][1]["job"]).read_text())
    high = yaml.safe_load(
        (tmp_path / "preview" / plan["records"][2]["job"]).read_text()
    )
    assert low["mass_scale"]["cube"] == [0.98, 0.98] and high["mass_scale"]["cube"] == [
        1.02,
        1.02,
    ]
    low_motion = yaml.safe_load(Path(low["motion_profile"]).read_text())
    assert set(low_motion["trajectory"]["spatial"]["joint_samples"].values()) == {-1.0}
    assert low["collision_profile"] == load_expansion_job(original).collision_profile
    assert original.read_bytes() == before
    with pytest.raises(FileExistsError):
        plan_preview(original, tmp_path / "preview", interior=1)
    with pytest.raises(ValueError, match="Nothing executed"):
        plan_preview(original, tmp_path / "too-large", max_episodes=1)
    assert not (tmp_path / "too-large").exists()


@pytest.fixture
def sample_plan(monkeypatch, tmp_path):
    import embodichain.lab.expansion._preview_job as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "logs").mkdir()
    records = []
    for index, role in enumerate(("center", "lower", "upper")):
        path = f"configs/{index}.yaml"
        (tmp_path / path).write_text("max_wall_time_s: 10\n")
        records.append(
            dict(
                semantic_id="test",
                index=index,
                role=role,
                job=path,
                result=f"jobs/{index}",
                confirmed=0,
                attempts=0,
                planned=1,
                status="unrun",
            )
        )
    monkeypatch.setattr(module, "load_expansion_job", lambda path: None)
    monkeypatch.setattr(
        module, "preflight_expansion_job", lambda cfg: {"semantic_id": "test"}
    )
    return dict(output=str(tmp_path), collection_budget_s=10, records=records)


def test_collection_refills_worker_slots_and_keeps_result_order(
    monkeypatch, sample_plan
):
    import embodichain.lab.expansion._preview_job as module

    live = set()
    starts = []

    class SampleProcess:
        returncode = None

        def __init__(self, args, **kw):
            self.index = int(Path(args[-1]).stem)
            live.add(self.index)
            starts.append(live.copy())

        def poll(self):
            # The long first sample can finish only after a later sample used the free slot.
            if self.returncode is None and (self.index or len(starts) == 3):
                live.remove(self.index)
                self.returncode = [2, 1, 0][self.index]
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", SampleProcess)
    result = collect_preview(sample_plan, sample_workers=2)
    assert starts == [{0}, {0, 1}, {0, 2}]
    assert [r["status"] for r in result["records"]] == ["rejected", "error", "rejected"]
    assert [r["returncode"] for r in result["records"]] == [2, 1, 0]
    assert result["confirmed"] == 0 and not result["coverage_complete"]
    assert (
        json.loads((Path(sample_plan["output"]) / "preview-plan.json").read_text())
        == result
    )


def test_corrupt_receipts_preserve_first_error_and_settle_finished_peers(
    monkeypatch, sample_plan, tmp_path
):
    import embodichain.lab.expansion._preview_job as module

    for i in (0, 1):
        path = tmp_path / f"jobs/{i}/episodes/e/receipt-0.json"
        path.parent.mkdir(parents=True)
        path.write_text(f"corrupt receipt {i}")
    attempt = tmp_path / "jobs/2/attempts/0/attempt.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({"ordinal": 0, "status": "rejected"}))

    class FinishedProcess:
        returncode = 2

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: FinishedProcess())
    with pytest.raises(json.JSONDecodeError) as error:
        collect_preview(sample_plan, sample_workers=3)
    assert error.value.doc == "corrupt receipt 0"
    result = json.loads((tmp_path / "preview-plan.json").read_text())
    assert [r["status"] for r in result["records"]] == ["error", "error", "rejected"]
    assert result["records"][2]["attempts"] == 1


@pytest.mark.parametrize("interrupt", [False, True], ids=["deadline", "ctrl-c"])
def test_collection_reaps_all_workers_and_preserves_unstarted_points(
    monkeypatch, sample_plan, interrupt
):
    from types import SimpleNamespace
    import embodichain.lab.expansion._preview_job as module

    now = 0.0
    processes = []

    def sleep(delay):
        nonlocal now
        if interrupt:
            raise KeyboardInterrupt
        now = sample_plan["collection_budget_s"]

    class StalledProcess:
        returncode = None
        signalled = False
        reaped = False

        def __init__(self, *args, **kw):
            processes.append(self)

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            self.signalled = sig == module.signal.SIGINT

        def wait(self, timeout=None):
            if self.returncode is None:
                raise module.subprocess.TimeoutExpired("sample", timeout)
            self.reaped = True
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep)
    )
    monkeypatch.setattr(module.subprocess, "Popen", StalledProcess)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            collect_preview(sample_plan, sample_workers=2)
    else:
        collect_preview(sample_plan, sample_workers=2)
    result = json.loads((Path(sample_plan["output"]) / "preview-plan.json").read_text())
    stopped = "interrupted" if interrupt else "budget_exhausted"
    assert [r["status"] for r in result["records"]] == [
        stopped,
        stopped,
        "unrun" if interrupt else "budget_exhausted",
    ]
    assert len(processes) == 2
    assert all(p.signalled and p.reaped for p in processes)
    assert not result["complete"] and not result["coverage_complete"]


def test_source_change_stops_before_collecting(monkeypatch, tmp_path):
    import embodichain.lab.expansion._preview_job as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/job.yaml").write_text("max_wall_time_s: 10\n")
    plan = {
        "output": str(tmp_path),
        "collection_budget_s": 10,
        "records": [
            {
                "index": 0,
                "job": "configs/job.yaml",
                "semantic_id": "pinned",
                "status": "unrun",
            }
        ],
    }
    monkeypatch.setattr(module, "load_expansion_job", lambda path: None)
    monkeypatch.setattr(
        module, "preflight_expansion_job", lambda cfg: {"semantic_id": "changed"}
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("changed source must not execute"),
    )
    with pytest.raises(ValueError, match="changed after planning"):
        collect_preview(plan)
    assert (
        json.loads((tmp_path / "preview-plan.json").read_text())["records"][0]["status"]
        == "source_changed"
    )


def test_reused_affordance_is_not_full_recipe_coverage():
    from embodichain.lab.expansion._preview_job import _exact_recipe

    sample = {"env_ids": [3, 7], "success": [True, True], "reused": [False, True]}
    metadata = {
        "expansion": [
            {
                "env_id": 7,
                "affordance_requested": 1,
                "trajectory_requested": 1,
                "trajectory_selected": 1,
            }
        ],
        "program_segments": [
            {
                "metadata": {
                    "runtime": {
                        "calls": [
                            {
                                "plan_attempts": [
                                    {
                                        "planner_diagnostics": {
                                            "metadata": {
                                                "affordance_sample": {"grasp": sample}
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        ],
    }
    assert not _exact_recipe(metadata)
    sample["reused"][1] = False
    assert _exact_recipe(metadata)
    metadata["expansion"][0]["trajectory_selected"] = 0
    assert not _exact_recipe(metadata)
