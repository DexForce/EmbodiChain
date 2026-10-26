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

import numpy as np
import pytest
from embodichain.gen_sim.task_engine._task_program.support_evidence import (
    support_from_contacts,
)


def _measure(*, normals=None, center=(0, 0, 0.1), pairs=None, impulse=0.001):
    positions = np.array(
        [[-0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0]]
    )
    return support_from_contacts(
        np.array([[1, 2]] * 4) if pairs is None else pairs,
        positions,
        np.array([[0, 0, 1]] * 4) if normals is None else normals,
        np.full(4, impulse),
        np.array(center),
        object_actor=2,
        reference_actor=1,
        tolerance=0.003,
    )


def test_real_load_patch_supports_com_without_full_silhouette_containment():
    accepted, metrics = _measure()
    assert accepted is True
    assert metrics["upward_impulse"] > 0
    assert metrics["com_patch_distance_m"] == 0


@pytest.mark.parametrize(
    "change", ["side", "below", "wrong_reference", "floating", "outside_com"]
)
def test_non_support_is_not_a_success(change):
    kwargs = {
        "side": {"normals": np.array([[1, 0, 0]] * 4)},
        "below": {"normals": np.array([[0, 0, -1]] * 4)},
        "wrong_reference": {"pairs": np.array([[3, 2]] * 4)},
        "floating": {"impulse": 0},
        "outside_com": {"center": (0.04, 0, 0.1)},
    }[change]
    assert _measure(**kwargs)[0] is False


def test_sleeping_contact_witness_is_invalidated_by_either_body_moving():
    import torch
    from embodichain.gen_sim.task_engine._task_program.support_evidence import (
        ContactWitness,
    )

    obj, ref = torch.eye(4), torch.eye(4)
    sample = (
        np.array([[1, 2]]),
        np.array([[0, 0, 0]]),
        np.array([[0, 0, 1]]),
        np.array([0.001]),
    )
    witness = ContactWitness()
    assert witness.update(obj, ref, None) is None
    witness.update(obj, ref, sample)
    assert witness.update(obj, ref, None) is not None
    moved = obj.clone()
    moved[0, 3] = 0.01
    assert witness.update(moved, ref, None) is None
    assert witness.update(obj, ref, None) is None
    witness.update(obj, ref, sample)
    assert witness.update(obj, moved, None) is None
    witness.update(obj, ref, sample)
    assert witness.update(obj, ref, None, invalid=True) is None
