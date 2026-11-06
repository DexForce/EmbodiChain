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

"""Keep raw protocol decoding independent of frontends and simulator startup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import pytest


def test_offline_protocol_api_imports_without_frontends_or_numeric_runtime():
    root = Path(__file__).resolve().parents[5]
    script = """
import importlib.abc,sys
class Offline(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.startswith(('embodichain.gen_sim','embodichain.lab.sim','dexsim','open3d','torch','numpy')):
            raise AssertionError(name)
sys.meta_path.insert(0,Offline())
from embodichain.lab.task_program.integrations import evidence
for name in evidence.__all__:
    assert callable(getattr(evidence,name)),name
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("unavailable_value", "encoded_value"),
    [(2.0, 2.0), (float("nan"), None), (float("inf"), None), (-float("inf"), None)],
)
def test_shared_evidence_json_preserves_row_validity_and_owns_values(
    unavailable_value: float, encoded_value: float | None
) -> None:
    from embodichain.lab.evidence import PhysicalEvidenceBatch, PhysicalEvidenceFrame
    from embodichain._json_contract import canonical_hash

    ids = torch.tensor([4, 1], dtype=torch.long)
    batch = PhysicalEvidenceBatch(
        evidence_id="pose",
        values=torch.tensor([[1.0], [unavailable_value]]),
        valid=torch.tensor([True, False]),
        acquisition_errors=(None, "missing"),
        timestamp=1.5,
        env_ids=ids,
        observation_revision=3,
    )
    frame = PhysicalEvidenceFrame(
        timestamp=1.5, env_ids=ids, observation_revision=3, batches={"pose": batch}
    )
    values = frame.to_dict()
    assert values["env_ids"] == [4, 1]
    assert values["batches"]["pose"]["valid"] == [True, False]
    assert values["batches"]["pose"]["acquisition_errors"] == [None, "missing"]
    assert values["batches"]["pose"]["values"] == [[1.0], [encoded_value]]
    assert canonical_hash(values) == canonical_hash(
        json.loads(json.dumps(values, allow_nan=False))
    )
    values["batches"]["pose"]["values"][0][0] = 99.0
    values["env_ids"].reverse()
    assert frame.to_dict()["batches"]["pose"]["values"] == [[1.0], [encoded_value]]
    assert frame.to_dict()["env_ids"] == [4, 1]
    torch.testing.assert_close(
        frame.batches["pose"].values,
        torch.tensor([[1.0], [unavailable_value]]),
        equal_nan=True,
    )


@pytest.mark.parametrize("family", ["handover", "coordinated"])
def test_historical_milestone_boundary_must_match_verified_record(family):
    from .test_handover_evidence import _case as handover
    from .test_coordinated_evidence import _case as coordinated
    from ._evaluation import normalized
    from embodichain.gen_sim.task_spec.manipulation_evidence import (
        evaluate_manipulation_evidence,
    )

    spec, binding, evidence, program = (
        handover(1) if family == "handover" else coordinated(final=False)
    )
    evidence["boundary_frames"] = [
        evidence["record"]["measurements"]["boundary_frames"][0] + 100
    ]
    acceptance, packet = normalized(binding, evidence, program, spec)
    outcome = evaluate_manipulation_evidence(spec, acceptance, packet, program=program)[
        0
    ]
    assert outcome.status.value == "unavailable"


@pytest.mark.parametrize("family", ["handover", "coordinated"])
@pytest.mark.parametrize("frames", [None, [], [-1], [True], ["invalid"], [0, 1]])
def test_incomplete_captures_return_unavailable_without_failing_report(family, frames):
    from .test_handover_evidence import _case as handover
    from .test_coordinated_evidence import _case as coordinated
    from ._evaluation import normalized
    from embodichain.gen_sim.task_spec.manipulation_evidence import (
        evaluate_manipulation_evidence,
    )

    spec, binding, evidence, program = (
        handover(3, raw=True) if family == "handover" else coordinated(raw=True)
    )
    evidence["boundary_frames"] = frames
    acceptance, packet = normalized(binding, evidence, program, spec)
    result = evaluate_manipulation_evidence(spec, acceptance, packet, program=program)[
        0
    ]
    assert result.status.value == "unavailable"


@pytest.mark.parametrize("family", ["handover", "coordinated"])
@pytest.mark.parametrize(
    "damage", ["collection_time", "source_step", "missing_source_time"]
)
def test_held_pose_channels_need_the_selected_history_step_and_original_time(
    family, damage
):
    from .test_handover_evidence import _case as handover
    from .test_coordinated_evidence import _case as coordinated
    from ._evaluation import normalized
    from embodichain.gen_sim.task_spec.manipulation_evidence import (
        evaluate_manipulation_evidence,
    )

    spec, binding, evidence, program = (
        handover(3, raw=True) if family == "handover" else coordinated(raw=True)
    )
    poses = evidence["held_poses"]
    if damage == "collection_time":
        poses["pose_frame"]["timestamp"] += 0.04
        for batch in poses["pose_frame"]["batches"].values():
            batch["timestamp"] += 0.04
    elif damage == "source_step":
        poses["recorded_acquisition"]["frame_indices"][0] += 1
    else:
        poses["recorded_acquisition"]["timestamps"] = [None]
    acceptance, packet = normalized(binding, evidence, program, spec)
    assert (
        evaluate_manipulation_evidence(spec, acceptance, packet, program=program)[
            0
        ].status.value
        == "unavailable"
    )


def test_recorded_pose_keeps_native_time_separate_from_episode_steps():
    from .test_handover_evidence import _case
    from ._evaluation import evaluate_handover_evidence

    spec, binding, evidence, program = _case(1, raw=True)
    poses = evidence["held_poses"]
    # Native simulation time may include warmup; collection may happen later.
    poses["pose_frame"]["timestamp"] += 2.0
    for batch in poses["pose_frame"]["batches"].values():
        batch["timestamp"] += 2.0
    outcome = evaluate_handover_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status.value == "satisfied", outcome.reason
