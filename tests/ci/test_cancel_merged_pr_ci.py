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

"""Execute the merged-PR cleanup workflow against a mocked GitHub API."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parents[2] / (
    ".github/workflows/cancel-merged-pr-ci.yml"
)
_PR_NUMBER = 123
_SOURCE_REPO_ID = 42
_HEAD_SHA = "a" * 40
_HEAD_BRANCH = "feature/example"


def _workflow() -> dict:
    return yaml.load(_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _run(
    runs: list[dict], *, merged: bool = True, cancel_errors: dict | None = None
) -> dict:
    job = _workflow()["jobs"]["cancel"]
    fixture = {
        "condition": job["if"],
        "script": job["steps"][0]["with"]["script"],
        "runs": runs,
        "errors": cancel_errors or {},
        "pr": {
            "number": _PR_NUMBER,
            "merged": merged,
            "head": {
                "repo": {"id": _SOURCE_REPO_ID},
                "ref": _HEAD_BRANCH,
                "sha": _HEAD_SHA,
            },
        },
    }
    harness = """
const fs = require('node:fs');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
const lists = [];
const cancellations = [];
const github = {
  event: { pull_request: fixture.pr },
  rest: { actions: {
    listWorkflowRunsForRepo: Symbol('list'),
    cancelWorkflowRun: async params => {
      cancellations.push(params);
      const status = fixture.errors[params.run_id];
      if (status) throw Object.assign(new Error('API error'), { status });
    },
  } },
  paginate: async (method, params) => {
    if (method !== github.rest.actions.listWorkflowRunsForRepo) {
      throw new Error('Unexpected pagination method');
    }
    lists.push(params);
    return fixture.runs.filter(run => run.status === params.status);
  },
};
const context = {
  repo: { owner: 'DexForce', repo: 'EmbodiChain' },
  payload: { pull_request: fixture.pr },
};
const core = { info: () => {} };
(async () => {
  let error = null;
  try {
    if (new Function('github', `return (${fixture.condition})`)(github)) {
      const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
      await new AsyncFunction('github', 'context', 'core', fixture.script)(
        github, context, core,
      );
    }
  } catch (caught) { error = caught.status ?? caught.message; }
  process.stdout.write(JSON.stringify({ lists, cancellations, error }));
})();
"""
    result = subprocess.run(
        ["node", "-e", harness],
        input=json.dumps(fixture),
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def _candidate(run_id: int, **overrides: object) -> dict:
    return {
        "id": run_id,
        "event": "pull_request",
        "status": "in_progress",
        "pull_requests": [{"number": _PR_NUMBER}],
        "head_repository": {"id": _SOURCE_REPO_ID},
        "head_branch": _HEAD_BRANCH,
        "head_sha": _HEAD_SHA,
        **overrides,
    }


def test_cleanup_uses_base_branch_code_and_minimum_permissions() -> None:
    workflow = _workflow()
    assert workflow["on"] == {
        "pull_request_target": {"types": ["closed"], "branches": ["main"]}
    }
    assert workflow["permissions"] == {"actions": "write"}
    steps = workflow["jobs"]["cancel"]["steps"]
    assert len(steps) == 1
    assert steps[0]["uses"].startswith("actions/github-script@")


def test_unmerged_close_does_not_query_or_cancel_runs() -> None:
    assert _run([_candidate(1)], merged=False) == {
        "lists": [],
        "cancellations": [],
        "error": None,
    }


def test_cleanup_cancels_only_unfinished_checks_for_the_merged_pr() -> None:
    result = _run(
        [
            _candidate(1),
            _candidate(2, status="queued"),
            _candidate(3, status="completed"),
            _candidate(4, pull_requests=[{"number": _PR_NUMBER + 1}]),
            _candidate(5, event="push", head_branch="main"),
            _candidate(6, event="workflow_dispatch"),
            _candidate(7, event="schedule"),
        ]
    )
    assert result["error"] is None
    assert {call["run_id"] for call in result["cancellations"]} == {1, 2}
    for status, call in zip(["queued", "in_progress"], result["lists"], strict=True):
        assert call == {
            "owner": "DexForce",
            "repo": "EmbodiChain",
            "event": "pull_request",
            "status": status,
            "per_page": 100,
        }


def test_missing_pr_associations_require_exact_source_identity() -> None:
    result = _run(
        [
            _candidate(1, pull_requests=[]),
            _candidate(2, pull_requests=None),
            _candidate(3, pull_requests=[], head_repository={"id": 99}),
            _candidate(4, pull_requests=[], head_branch="other"),
            _candidate(5, pull_requests=[], head_sha="b" * 40),
            _candidate(6, pull_requests=[], head_repository=None),
        ]
    )
    assert result["error"] is None
    assert {call["run_id"] for call in result["cancellations"]} == {1, 2}


def test_finished_run_conflict_does_not_prevent_remaining_cancellations() -> None:
    result = _run([_candidate(1), _candidate(2)], cancel_errors={1: 409})
    assert result["error"] is None
    assert [call["run_id"] for call in result["cancellations"]] == [1, 2]


@pytest.mark.parametrize("status", [403, 500])
def test_unexpected_api_errors_fail_the_workflow(status: int) -> None:
    result = _run([_candidate(1)], cancel_errors={1: status})
    assert result["error"] == status
