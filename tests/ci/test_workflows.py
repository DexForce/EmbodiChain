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

"""Contract tests for GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_workflow(name: str) -> dict:
    workflow_path = _REPOSITORY_ROOT / ".github" / "workflows" / name
    return yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_automatic_pages_deployment_requires_existing_docs_artifact() -> None:
    jobs = _load_workflow("docs-pages.yml")["jobs"]

    assert "inspect-docs-artifact" in jobs

    inspection = jobs["inspect-docs-artifact"]
    deployment = jobs["deploy-from-artifact"]
    assert "environment" not in inspection
    assert inspection["outputs"]["exists"] == "${{ steps.artifact.outputs.exists }}"
    assert deployment["needs"] == "inspect-docs-artifact"
    deploy_condition = deployment["if"]
    assert deploy_condition.startswith("always() &&")
    assert "needs.inspect-docs-artifact.outputs.exists == 'true'" in deploy_condition
    assert "inputs.artifact_run_id != ''" in deploy_condition
