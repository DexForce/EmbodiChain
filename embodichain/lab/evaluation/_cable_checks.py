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

"""Field names for recorded cable task acceptance checks."""

from __future__ import annotations

__all__: list[str] = []

RJ45_CHECKS = (
    "starts_on_floor",
    "insertion_without_finger_latch_press",
    "vertical_withdrawal_grasp",
    "complete",
    "pickup_and_lift",
    "seated_while_robot_home",
    "latch_locked_while_robot_home",
    "physical_latch_press_before_withdrawal",
    "latch_limits",
    "latch_returns_after_release",
    "unplugged",
    "placed_on_floor",
    "final_home",
    "fixed_end",
    "cable_attachment",
    "contacts",
    "finite_state",
)

WRAP_CHECKS = (
    "completed",
    "finite",
    "joint_limits",
    "direct_floor_pickup",
    "grasp_retained",
    "material_grasp_slip",
    "bilateral_finger_contact",
    "actual_winding",
    "post_contact_arc",
    "hold_stable",
    "rope_below_post_top",
    "post_nonpenetration",
    "robot_post_clearance",
    "anchor_points_toward_post",
    "smooth_anchor_exit",
    "anchor_fixed",
    "length_preserved",
    "contacts_not_dropped",
    "floor_nonpenetration",
)
