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

"""
This script shows how to customize the end-effectors of the DexForce W1 robot by
adding a pair of parallel grippers as the left and right hands.
"""

from __future__ import annotations

import argparse

from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)


def main() -> None:
    """Create a DexForce W1 with custom parallel-gripper end effectors."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Customize DexForce W1 end effectors.")
    )
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(args, add_default_light=False)
    try:
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "version": "v021",
                "arm_kind": "anthropomorphic",
                "with_default_eef": False,
                "control_parts": {
                    "left_eef": ["LEFT_FINGER1_JOINT", "LEFT_FINGER2_JOINT"],
                    "right_eef": ["RIGHT_FINGER1_JOINT", "RIGHT_FINGER2_JOINT"],
                },
                "urdf_cfg": {
                    "components": [
                        {
                            "component_type": "left_hand",
                            "urdf_path": "DH_PGC_140_50/DH_PGC_140_50.urdf",
                        },
                        {
                            "component_type": "right_hand",
                            "urdf_path": "DH_PGC_140_50/DH_PGC_140_50.urdf",
                        },
                    ]
                },
                "drive_pros": {
                    "max_effort": {
                        "left_eef": 10.0,
                        "right_eef": 10.0,
                    },
                },
            }
        )
        sim.add_robot(cfg=cfg)
        sim.update(step=1)
        print("DexforceW1 with user-defined end effectors added.")
        maybe_open_window(sim, args)
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
