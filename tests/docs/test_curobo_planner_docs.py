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

"""Source-level coverage for the public cuRobo planner documentation."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANNER_DOCS = _REPO_ROOT / "docs" / "source" / "overview" / "sim" / "planners"


def test_curobo_planner_docs_are_linked_and_scoped() -> None:
    """Keep the optional V2 backend discoverable without overstating support."""
    index = (_PLANNER_DOCS / "index.rst").read_text(encoding="utf-8")
    page = (_PLANNER_DOCS / "curobo_planner.md").read_text(encoding="utf-8")

    assert "curobo_planner.md" in index
    assert "CuroboPlannerCfg" in page
    assert 'planner_type="curobo"' in page
    assert "cuRobo V2" in page
    assert "attached-object" in page
    assert "tool_frame_to_tcp" in page
    assert "sim_base_to_curobo_base" in page
    assert "multi_env=True" in page
    assert "lock_joints" in page
    assert "--record-save-path" in page
    assert "examples/sim/planners/curobo_planner.py" in page
