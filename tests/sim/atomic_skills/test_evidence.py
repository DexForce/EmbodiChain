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

"""Tests for versioned Atomic Skill physical-evidence contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_skills import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
    PhysicalEvidenceRequest,
)


def _frame() -> PhysicalEvidenceFrame:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    return PhysicalEvidenceFrame(
        timestamp=1.25,
        observation_revision=4,
        env_ids=env_ids,
        batches={
            "contact": PhysicalEvidenceBatch(
                evidence_id="contact",
                values=torch.tensor([True, False]),
                valid=torch.tensor([True, True]),
                acquisition_errors=(None, None),
                timestamp=1.25,
                env_ids=env_ids,
                observation_revision=4,
            )
        },
    )


def test_evidence_frame_owns_aligned_batches() -> None:
    frame = _frame()

    batch = frame.get("contact")
    assert frame.batch_size == 2
    assert torch.equal(batch.env_ids, torch.tensor([3, 7]))
    assert batch.values.tolist() == [True, False]

    with pytest.raises(KeyError, match="missing"):
        frame.get("missing")


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("mismatch", ["env_ids", "timestamp", "observation_revision"])
def test_evidence_frame_rejects_misaligned_channels(owned, mismatch) -> None:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    create = (
        PhysicalEvidenceFrame._from_owned_batches if owned else PhysicalEvidenceFrame
    )
    with pytest.raises(ValueError, match="share frame"):
        create(
            timestamp=1.25,
            observation_revision=4,
            env_ids=env_ids,
            batches={
                "contact": PhysicalEvidenceBatch(
                    evidence_id="contact",
                    values=torch.tensor([True, False]),
                    valid=torch.ones(2, dtype=torch.bool),
                    acquisition_errors=(None, None),
                    timestamp=1.5 if mismatch == "timestamp" else 1.25,
                    env_ids=torch.tensor(
                        [3, 8] if mismatch == "env_ids" else [3, 7], dtype=torch.long
                    ),
                    observation_revision=5 if mismatch == "observation_revision" else 4,
                )
            },
        )


def test_public_frame_construction_and_snapshot_revalidate_mutable_batches():
    ids = torch.tensor([3, 7])
    values = torch.tensor([1.0, 2.0])
    batch = PhysicalEvidenceBatch(
        "force", values, torch.ones(2, dtype=torch.bool), (None, None), 1.25, ids, 4
    )
    frame = PhysicalEvidenceFrame(1.25, 4, ids, {"force": batch})
    values.zero_()
    ids[0] = 99
    batch.values.fill_(3.0)
    assert frame.get("force").values.tolist() == [1.0, 2.0]
    assert frame.env_ids.tolist() == [3, 7]
    batch.values[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PhysicalEvidenceFrame(1.25, 4, torch.tensor([3, 7]), {"force": batch})
    saved = frame.snapshot()
    frame.batches["force"].values[0] = float("nan")
    assert saved.get("force").values.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        frame.snapshot()


def test_evidence_request_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="at least one environment"):
        PhysicalEvidenceRequest(
            timestamp=0.0,
            observation_revision=0,
            env_ids=torch.empty(0, dtype=torch.long),
        )
