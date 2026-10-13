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


def test_single_support_point_is_valid_only_directly_under_the_com():
    result, _ = support_from_contacts(
        np.array([[1, 2]]),
        np.array([[0, 0, 0]]),
        np.array([[0, 0, 1]]),
        np.array([0.001]),
        np.array([0, 0, 0.1]),
        object_actor=2,
        reference_actor=1,
        tolerance=0.003,
    )
    assert result is True


def test_nonfinite_contact_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        _measure(center=(float("nan"), 0, 0.1))


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


def test_on_capture_uses_native_current_support_and_exact_past_receipt(
    tmp_path, monkeypatch
):
    import torch
    from copy import deepcopy
    from .test_task_spec_runtime import _placement_evidence, _observe_placement
    from embodichain.gen_sim.task_engine._task_program import support_evidence
    from embodichain.gen_sim.task_engine.contracts import canonical_hash

    data = _placement_evidence(tmp_path, relation="on")
    plan, contract, runtime, program, constraints, env, deployment = data
    place = next(
        n
        for n in plan.graph["nodes"]
        if n["call"].get("call_id") == "simulation.place_relative"
    )

    class Observation:
        supported = True

        def __init__(self, simulation, obj, reference, **kw):
            self.entity, self.reference = kw["entity_id"], kw["reference_id"]

        def observe(self, obj, ref):
            return torch.tensor([self.supported]), {
                "scope": "native_contact_support",
                "entity": self.entity,
                "reference": self.reference,
                "supported": [self.supported],
                "env_ids": [0],
                "object_pose": obj.tolist(),
                "reference_pose": ref.tolist(),
                "rows": [
                    {
                        "contacts_world": [[0, 0, 0.005]],
                        "center_of_mass_world": [0, 0, 0.045],
                        "com_patch_distance_m": 0.0 if self.supported else None,
                        "com_tolerance_m": 0.003,
                        "upward_impulse": 0.001 if self.supported else 0.0,
                    }
                ],
            }

    monkeypatch.setattr(support_evidence, "NativeSupportObservation", Observation)
    obj = env.sim.get_rigid_object("cube").get_local_pose()
    ref = env.sim.get_rigid_object("region").get_local_pose()
    _, measurements = Observation(
        None, None, None, entity_id="cube", reference_id="region"
    ).observe(obj, ref)
    boundary = runtime["segments"][-1]
    boundary["metadata"].update(
        program_segment_source_path=["program", "items", 2],
        post_policies=[
            {
                "kind": "wait_stable",
                "policy_index": 0,
                "source_path": ["program", "items", 2, "post", 0],
                "result_mask": [True],
                "result": {
                    "kind": "supported_placement",
                    "accepted_mask": [True],
                    "measurements": {"native_support": measurements},
                },
            }
        ],
    )
    program["program"]["items"][2]["post"] = [
        {"kind": "wait_stable", "entity": "cube", "preset": "native"}
    ]
    contract["program_digest"] = canonical_hash(program)
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is True
    # A later loss of support cannot erase the past measured support receipt.
    Observation.supported = False
    result = _observe_placement(tmp_path, data)
    assert result["goal"][0]["status"] == "failed"
    assert result["milestones"][0]["status"] == "satisfied"
    boundary["metadata"]["post_policies"][0]["result"]["measurements"][
        "native_support"
    ]["reference"] = "foreign"
    assert (
        _observe_placement(tmp_path, data)["milestones"][0]["status"] == "unavailable"
    )
