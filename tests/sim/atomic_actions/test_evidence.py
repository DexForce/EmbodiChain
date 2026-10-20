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

"""Tests for versioned Atomic Action physical-evidence contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
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


def test_evidence_frame_rejects_misaligned_channels() -> None:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    with pytest.raises(ValueError, match="share frame env_ids"):
        PhysicalEvidenceFrame(
            timestamp=1.25,
            observation_revision=4,
            env_ids=env_ids,
            batches={
                "contact": PhysicalEvidenceBatch(
                    evidence_id="contact",
                    values=torch.tensor([True, False]),
                    valid=torch.ones(2, dtype=torch.bool),
                    acquisition_errors=(None, None),
                    timestamp=1.25,
                    env_ids=torch.tensor([3, 8], dtype=torch.long),
                    observation_revision=4,
                )
            },
        )


def test_evidence_request_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="at least one environment"):
        PhysicalEvidenceRequest(
            timestamp=0.0,
            observation_revision=0,
            env_ids=torch.empty(0, dtype=torch.long),
        )
