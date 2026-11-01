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

"""Model-owned identified inertial overlays for the original Panda.

Values are published by franka_ros at ddd2fffd9de4,
robots/common/inertial.yaml (SHA256
c7d15b15e45876b44e030053bca3ca19cf471806e46031ec090b8500cb9b330b).
These identified parameters are not manufacturer-certified values.
"""

from __future__ import annotations

from embodichain.lab.sim.cfg import (
    LinkPhysicsOverrideCfg,
    RigidBodyPhysicsCfg,
    MassPropertiesCfg,
)

__all__: list[str] = []

_LINKS = {
    "arm_base_link": {
        "mass": 0.629769,
        "com_position": [-0.041018, -0.00014, 0.049974],
        "inertia": [
            [0.00315, 8.2904e-07, 0.00015],
            [8.2904e-07, 0.00388, 8.2299e-06],
            [0.00015, 8.2299e-06, 0.004285],
        ],
    },
    "Link1": {
        "mass": 4.970684,
        "com_position": [0.003875, 0.002081, -0.04762],
        "inertia": [
            [0.70337, -0.000139, 0.006772],
            [-0.000139, 0.70661, 0.019169],
            [0.006772, 0.019169, 0.009117],
        ],
    },
    "Link2": {
        "mass": 0.646926,
        "com_position": [-0.003141, -0.02872, 0.003495],
        "inertia": [
            [0.007962, -0.003925, 0.010254],
            [-0.003925, 0.02811, 0.000704],
            [0.010254, 0.000704, 0.025995],
        ],
    },
    "Link3": {
        "mass": 3.228604,
        "com_position": [0.027518, 0.039252, -0.066502],
        "inertia": [
            [0.037242, -0.004761, -0.011396],
            [-0.004761, 0.036155, -0.012805],
            [-0.011396, -0.012805, 0.01083],
        ],
    },
    "Link4": {
        "mass": 3.587895,
        "com_position": [-0.05317, 0.104419, 0.027454],
        "inertia": [
            [0.025853, 0.007796, -0.001332],
            [0.007796, 0.019552, 0.008641],
            [-0.001332, 0.008641, 0.028323],
        ],
    },
    "Link5": {
        "mass": 1.225946,
        "com_position": [-0.011953, 0.041065, -0.038437],
        "inertia": [
            [0.035549, -0.002117, -0.004037],
            [-0.002117, 0.029474, 0.000229],
            [-0.004037, 0.000229, 0.008627],
        ],
    },
    "Link6": {
        "mass": 1.666555,
        "com_position": [0.060149, -0.014117, -0.010517],
        "inertia": [
            [0.001964, 0.000109, -0.001158],
            [0.000109, 0.004354, 0.000341],
            [-0.001158, 0.000341, 0.005433],
        ],
    },
    "Link7": {
        "mass": 0.735522,
        "com_position": [0.010517, -0.004252, 0.061597],
        "inertia": [
            [0.012516, -0.000428, -0.001196],
            [-0.000428, 0.010027, -0.000741],
            [-0.001196, -0.000741, 0.004815],
        ],
    },
}


def make_inertial_overlays() -> dict[str, LinkPhysicsOverrideCfg]:
    """Return owned typed overlays from the pinned model parameters."""
    return {
        name: LinkPhysicsOverrideCfg(
            link_names_expr=[name],
            attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(**values)),
        )
        for name, values in _LINKS.items()
    }
