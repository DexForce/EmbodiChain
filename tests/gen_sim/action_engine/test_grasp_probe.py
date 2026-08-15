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

from typing import Any

from embodichain.gen_sim.action_engine import grasp_probe


def test_probe_targets_only_coordinated_pickment_objects(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_probe(uid: str, item: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(uid)
        return {
            "kind": "grasp_policy_probe",
            "subject": uid,
            "status": "proven",
            "reason": "found",
            "evidence": {"outcome": "grasp_policy_satisfied"},
        }

    monkeypatch.setattr(grasp_probe, "_probe_object", fake_probe)
    result = grasp_probe.probe_coordinated_grasp_policy(
        {
            "nodes": [
                {
                    "atomic_action": "CoordinatedPickment",
                    "object_uid": "basin",
                },
                {"atomic_action": "MoveHeldObject", "object_uid": "basin"},
            ]
        },
        {"objects": [{"uid": "basin"}]},
        robot_profile="dual_franka",
    )

    assert calls == ["basin"]
    assert result[0]["evidence"]["outcome"] == "grasp_policy_satisfied"
