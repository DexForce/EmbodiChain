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

import pytest
from embodichain.data_analysis.collection import candidate_parameters, StepObserver


def test_recipe_is_reproducible_and_actually_varies_conditions():
    rows = [candidate_parameters(i, 17) for i in range(12)]
    assert rows == [candidate_parameters(i, 17) for i in range(12)]
    for key in ("size", "x", "y", "material", "light"):
        assert len({r[key] for r in rows}) >= 2
    with pytest.raises(ValueError):
        candidate_parameters(-1, 17)


def test_observer_captures_after_step_and_preserves_return():
    class Env:
        unwrapped = "target"

        def step(self, action):
            events.append(action)
            return ("obs", 1, False, False, {})

    events = []
    wrapped = StepObserver(Env(), lambda: events.append("capture"))
    assert wrapped.step(2) == ("obs", 1, False, False, {})
    assert events == [2, "capture"]
    assert wrapped.unwrapped == "target"


def test_pending_writer_failure_is_partial_and_remaining_attempts_continue(
    tmp_path, monkeypatch
):
    import json
    import subprocess
    from embodichain.data_analysis.catalog import Catalog
    from embodichain.data_analysis.collection import collect_preview

    calls = []

    def timeout(command, **kwargs):
        request = command[-1]
        record = json.loads(__import__("pathlib").Path(request).read_text())
        with Catalog(tmp_path / "catalog.sqlite") as catalog:
            record["status"] = "pending_write"
            catalog.upsert(record)
        calls.append(request)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = collect_preview(tmp_path, count=2, timeout=1)
    assert len(calls) == 2
    assert result["statuses"] == {"partial_commit": 2}
