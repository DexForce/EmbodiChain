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

from copy import deepcopy
import json

import pytest

from embodichain.gen_sim.task_engine import binding_evaluation as evaluation


def _response(case: dict) -> dict:
    return {
        "bindings": [
            {"reference_id": reference_id, "confidence": 1.0, **expected}
            for reference_id, expected in case["expected"].items()
        ]
    }


def test_frozen_cases_cover_positive_negative_and_decomposition_controls() -> None:
    cases = evaluation._cases()

    assert len(cases) == 11
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["expected"]) for case in cases) == 12
    assert {row["status"] for case in cases for row in case["expected"].values()} == {
        "resolved",
        "not_found",
        "ambiguous",
    }
    assert len(cases[-1]["intent"]["steps"]) == 2
    changed = deepcopy(cases)
    changed[0]["scene"][0]["category"] = "changed"
    assert evaluation._cases() == cases


def test_dry_run_writes_prompts_without_loading_or_calling_provider(tmp_path) -> None:
    report = evaluation._run(tmp_path, repeats=1, model=None)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["mode"] == "dry_run"
    assert manifest["max_provider_calls"] == 22
    assert report["completed_references"] == 0
    assert report["requested_references"] == 12
    assert report["correct_binding"]["rate"] == 0
    records = sorted((tmp_path / "calls").glob("*.json"))
    assert len(records) == 11
    record = json.loads(records[0].read_text())
    assert record["calls"][0]["prompt"]
    assert record["calls"][0]["schema"]
    assert "response" not in record["calls"][0]
    assert (
        len(
            [
                line
                for line in (tmp_path / "report.md").read_text().splitlines()
                if line.startswith("| ---")
            ]
        )
        == 3
    )


def test_correct_resolved_and_rejected_responses_pass_native_grounding() -> None:
    for case in evaluation._cases():
        record = evaluation._evaluate_case(
            case, repeat=0, model="test", caller=lambda **_: _response(case)
        )
        assert record["completed"], record
        assert record["observed"] == case["expected"]
        assert record["calls"][0]["response"] == _response(case)


def test_metrics_keep_failures_in_denominator_and_skip_incomplete_groups() -> None:
    cases = evaluation._cases()
    records = [
        {
            "case_id": case["id"],
            "repeat": 0,
            "completed": True,
            "observed": deepcopy(case["expected"]),
        }
        for case in cases
    ]
    records[0]["completed"] = False
    records[0]["observed"] = {}
    # A rejection turned into a resolved, wrong substitute is not a success.
    negative = next(
        i
        for i, case in enumerate(cases)
        if next(iter(case["expected"].values()))["status"] == "not_found"
    )
    records[negative]["observed"] = {
        key: {"status": "resolved", "uids": ["tray_001"]}
        for key in cases[negative]["expected"]
    }
    metrics = evaluation._metrics(cases, records, repeats=1)

    assert metrics["correct_binding"] == {"correct": 8, "total": 9, "rate": 8 / 9}
    assert metrics["wrong_substitution"] == {"correct": 1, "total": 12, "rate": 1 / 12}
    assert metrics["reject_accuracy"] == {"correct": 2, "total": 3, "rate": 2 / 3}
    assert metrics["paraphrase_consistency"]["total"] == 0
    assert metrics["paraphrase_groups_requested"] == 1


def test_malformed_and_transport_failures_are_not_valid_rejections() -> None:
    case = evaluation._cases()[0]

    def transport_failure(**kwargs):
        raise RuntimeError("transport unavailable")

    for caller in (transport_failure, lambda **_: {"bindings": []}):
        record = evaluation._evaluate_case(case, repeat=0, model="test", caller=caller)
        assert not record["completed"]
        assert record["observed"] == {}
        assert record["error"]
    assert len(record["calls"]) == 2


def test_rejected_claim_of_resolution_cannot_score_as_correct() -> None:
    case = evaluation._cases()[0]
    response = _response(case)
    response["bindings"][0]["match_evidence"] = {
        "match_type": "broader_term",
        "support": ["category resembles the reference"],
        "conflicts": ["requested color differs"],
        "candidate_uids": ["tray_001"],
    }

    record = evaluation._evaluate_case(
        case, repeat=0, model="test", caller=lambda **_: response
    )

    assert not record["completed"]
    assert record["observed"] == {}
    assert (
        evaluation._metrics([case], [record], repeats=1)["correct_binding"]["rate"] == 0
    )


def test_conflict_mixed_with_ambiguity_is_not_a_completed_semantic_refusal() -> None:
    case = evaluation._cases()[-1]
    response = _response(case)
    response["bindings"][0]["match_evidence"] = {
        "match_type": "contextual",
        "support": ["category resembles the reference"],
        "conflicts": ["explicit material mismatch"],
        "candidate_uids": ["tray_001"],
    }
    response["bindings"][1].update(status="ambiguous", uids=[])

    record = evaluation._evaluate_case(
        case, repeat=0, model="test", caller=lambda **_: response
    )

    assert not record["completed"]
    assert record["observed"] == {}
    assert len(record["calls"]) == 1


def test_reusing_output_and_nonpositive_repeats_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluation._run(tmp_path, repeats=0, model=None)
    evaluation._run(tmp_path, repeats=1, model=None)
    with pytest.raises(FileExistsError):
        evaluation._run(tmp_path, repeats=1, model=None)
