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
"""Task-owned checks shared by the task1186 example and benchmarks."""

from __future__ import annotations


def inspect_1186(graph):
    """Check the declared fruit roles, transfer sequence and named destination."""
    calls = [n["call"] for n in graph["nodes"]]
    kinds = [c.get("call_id", c["kind"]) for c in calls]
    sequence = ["pick", "hand_over", "place"] * 2 + ["simulation.coordinated_transport"]
    fruit_ok = len(calls) == 7 and all(
        fruit in str(c.get("object", "")).lower()
        for fruit, block in (("apple", calls[:3]), ("orange", calls[3:6]))
        for c in block
    )
    assignments = [
        {"primary": "right"},
        {"source": "right", "destination": "left"},
        {"primary": "left"},
    ] * 2 + [{"left": "left", "right": "right"}]
    arm_ok = [c["resources"] for c in calls] == assignments
    order_ok = (
        kinds == sequence
        and fruit_ok
        and all(
            n["depends_on"] == ([graph["nodes"][i - 1]["id"]] if i else [])
            for i, n in enumerate(graph["nodes"])
        )
    )
    landing = calls[-1].get("arguments", {})
    plate = calls[2].get("on", calls[2].get("inside")) if len(calls) == 7 else None
    support_ok = bool(
        plate
        and plate == calls[5].get("on", calls[5].get("inside"))
        and landing.get("object") == plate
        and ("plate" in plate or "tray" in plate)
        and landing.get("target", "").lower().replace(" ", "_")
        in {
            "mat",
            "service_pad",
            "service_mat",
            "service_mat_01",
            "serv_mat",
            "serving_mat",
        }
        and kinds[-1] == sequence[-1]
    )
    return dict(
        arms=arm_ok,
        ordered_transfers=order_ok,
        shared_plate_and_final_support=support_ok,
    )
