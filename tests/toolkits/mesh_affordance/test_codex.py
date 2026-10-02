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

"""Reject malformed model output and preserve diagnostic process failures."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from embodichain.toolkits.mesh_affordance import _codex


def _payload():
    return {
        "task_interpretation": "pour",
        "assumptions": ["unknown gripper"],
        "regions": [
            {
                "name": "handle",
                "patch_ids": [0],
                "score": 0.9,
                "confidence": 0.8,
                "reason": "control",
            },
            {
                "name": "rim",
                "patch_ids": [1],
                "score": 0.1,
                "confidence": 0.9,
                "reason": "keep free",
            },
        ],
    }


def test_region_scores_map_by_id_not_response_order():
    payload = _payload()
    payload["regions"].reverse()
    scores, confidence = _codex.validate_response(payload, 2)
    np.testing.assert_allclose(scores, [0.9, 0.1])
    np.testing.assert_allclose(confidence, [0.8, 0.9])


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "unknown",
        "nan",
        "out_of_range",
        "boolean_id",
        "boolean_score",
    ],
)
def test_malformed_model_reply_is_rejected(mutation):
    payload = deepcopy(_payload())
    region = payload["regions"][0]
    if mutation == "duplicate":
        region["patch_ids"].append(1)
    elif mutation == "missing":
        payload["regions"].pop()
    elif mutation == "unknown":
        region["patch_ids"] = [10]
    elif mutation == "nan":
        region["score"] = float("nan")
    elif mutation == "out_of_range":
        region["confidence"] = 1.2
    elif mutation == "boolean_id":
        region["patch_ids"] = [True]
    else:
        region["score"] = True
    with pytest.raises(ValueError):
        _codex.validate_response(payload, 2)


def test_codex_command_keeps_model_images_schema_and_readonly_boundary(
    tmp_path, monkeypatch
):
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "images": ["view one.png"],
                "object_description": "cup",
                "task_description": "pour",
            }
        )
    )

    def fake_process(command, directory, stem, timeout, prompt=None):
        assert command[command.index("--model") + 1] == "gpt-6-astra"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert command[command.index("--image") + 1] == str(tmp_path / "view one.png")
        assert command[-1] == "-"
        assert "untrusted data" in prompt
        schema = json.loads((directory / "response.schema.json").read_text())
        assert schema["additionalProperties"] is False
        (directory / "response.raw.txt").write_text(json.dumps(_payload()))

    monkeypatch.setattr(_codex, "run_process", fake_process)
    assert _codex.run_codex(tmp_path, "gpt-6-astra", "codex", 30) == _payload()


def test_subprocess_failure_has_logs(tmp_path):
    with pytest.raises(RuntimeError, match="stderr.log"):
        _codex.run_process(
            [
                sys.executable,
                "-c",
                "import sys; print('failed', file=sys.stderr); sys.exit(3)",
            ],
            tmp_path,
            "failure",
            10,
        )
    assert "failed" in (tmp_path / "failure.stderr.log").read_text()


def test_subprocess_timeout_is_reported(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        _codex.run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path,
            "timeout",
            0.1,
        )


def test_part_membership_and_confidence_are_validated_separately():
    payload = _payload()
    for region in payload["regions"]:
        region.update(part_score=1.0, part_confidence=0.8)
    scores, confidence, part_scores, part_confidence = _codex.validate_response(
        payload, 2, True
    )
    np.testing.assert_allclose(part_scores, [1, 1])
    payload["regions"][0]["part_score"] = float("nan")
    with pytest.raises(ValueError, match="part_score"):
        _codex.validate_response(payload, 2, True)


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```"])
def test_plain_and_single_fenced_json_responses_decode(wrapper):
    payload = _payload()
    assert _codex.decode_response(wrapper.format(json.dumps(payload))) == payload


@pytest.mark.parametrize(
    "text", ['prefix {"regions": []}', "```json\n{}\n``` trailing", "{} {}", "[]", ""]
)
def test_response_decoder_rejects_prose_multiple_objects_and_nonobjects(text):
    with pytest.raises(ValueError):
        _codex.decode_response(text)
