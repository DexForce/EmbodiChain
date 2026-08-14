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

from pathlib import Path

from scripts.benchmark.gen_sim.e1_e2_scene_action import run_benchmark


def test_e1_e2_contract_benchmark_is_reproducibly_executable(tmp_path: Path) -> None:
    results, report = run_benchmark(iterations=2, output_dir=tmp_path)

    assert tuple(item.scenario for item in results) == ("E1", "E2")
    assert all(item.success_rate == 1.0 for item in results)
    assert all(item.feasibility_status == "runtime_probe" for item in results)
    assert all(item.action_count >= 3 for item in results)
    markdown = report.read_text(encoding="utf-8")
    assert markdown.count("## Time & Memory") == 1
    assert markdown.count("## Success & Other Metrics") == 1
    assert markdown.count("## Leaderboard") == 1
