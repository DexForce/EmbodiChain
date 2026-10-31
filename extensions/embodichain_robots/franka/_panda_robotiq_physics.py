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

"""Model-owned inertial overlays for the original Panda / Robotiq proxy.

Panda identification values as published by franka_ros at ddd2fffd9de4,
robots/common/inertial.yaml (SHA256 c7d15b15e45876b44e030053bca3ca19cf471806e46031ec090b8500cb9b330b).
These are identified parameters, explicitly not manufacturer-certified values.
Robotiq CAD values were commented in ros-industrial-attic/robotiq at
45196f6558fe, robotiq_arg2f_85_model_macro.xacro (SHA256
bb38a58a3112017efbc273327fc65388449f4e4a54a23c3218cf35e7b89356bb).
The 5 g rubber pads are explicit simulation assumptions with box inertia.
Geometry, motion chains and source measurements are unchanged.
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
    "left_outer_knuckle": {
        "mass": 0.00853198276973456,
        "com_position": [-0.000200000000003065, 0.0199435877845359, 0.0292245259211331],
        "inertia": [
            [2.89328108496468e-06, -1.57935047237397e-19, -1.93980378593255e-19],
            [-1.57935047237397e-19, 1.86719750325683e-06, -1.21858577871576e-06],
            [-1.93980378593255e-19, -1.21858577871576e-06, 1.21905238907251e-06],
        ],
    },
    "right_outer_knuckle": {
        "mass": 0.00853198276973456,
        "com_position": [-0.000200000000003065, 0.0199435877845359, 0.0292245259211331],
        "inertia": [
            [2.89328108496468e-06, -1.57935047237397e-19, -1.93980378593255e-19],
            [-1.57935047237397e-19, 1.86719750325683e-06, -1.21858577871576e-06],
            [-1.93980378593255e-19, -1.21858577871576e-06, 1.21905238907251e-06],
        ],
    },
    "left_outer_finger": {
        "mass": 0.022614240507152,
        "com_position": [0.00030115855001899, 0.0373907951953854, -0.0208027427000385],
        "inertia": [
            [1.52518312458174e-05, 9.76583423954399e-10, -5.43838577022588e-10],
            [9.76583423954399e-10, 6.17694243867776e-06, 6.78636130740228e-06],
            [-5.43838577022588e-10, 6.78636130740228e-06, 1.16494917907219e-05],
        ],
    },
    "right_outer_finger": {
        "mass": 0.022614240507152,
        "com_position": [0.00030115855001899, 0.0373907951953854, -0.0208027427000385],
        "inertia": [
            [1.52518312458174e-05, 9.76583423954399e-10, -5.43838577022588e-10],
            [9.76583423954399e-10, 6.17694243867776e-06, 6.78636130740228e-06],
            [-5.43838577022588e-10, 6.78636130740228e-06, 1.16494917907219e-05],
        ],
    },
    "left_inner_knuckle": {
        "mass": 0.0271177346495152,
        "com_position": [0.000123011831763771, 0.0507850843201817, 0.00103968640075166],
        "inertia": [
            [2.61910379223783e-05, -2.43616858946494e-07, -6.37789906117123e-09],
            [-2.43616858946494e-07, 2.8270243746167e-06, -5.37200748039765e-07],
            [-6.37789906117123e-09, -5.37200748039765e-07, 2.83695868220296e-05],
        ],
    },
    "right_inner_knuckle": {
        "mass": 0.0271177346495152,
        "com_position": [0.000123011831763771, 0.0507850843201817, 0.00103968640075166],
        "inertia": [
            [2.61910379223783e-05, -2.43616858946494e-07, -6.37789906117123e-09],
            [-2.43616858946494e-07, 2.8270243746167e-06, -5.37200748039765e-07],
            [-6.37789906117123e-09, -5.37200748039765e-07, 2.83695868220296e-05],
        ],
    },
    "left_inner_finger": {
        "mass": 0.0104003125914103,
        "com_position": [0.000299999999999317, 0.0160078233491243, -0.0136945669206257],
        "inertia": [
            [2.71909453810972e-06, 1.35402465472579e-21, -7.1817349065269e-22],
            [1.35402465472579e-21, 7.69100314106116e-07, 6.74715432769696e-07],
            [-7.1817349065269e-22, 6.74715432769696e-07, 2.30315190420171e-06],
        ],
    },
    "right_inner_finger": {
        "mass": 0.0104003125914103,
        "com_position": [0.000299999999999317, 0.0160078233491243, -0.0136945669206257],
        "inertia": [
            [2.71909453810972e-06, 1.35402465472579e-21, -7.1817349065269e-22],
            [1.35402465472579e-21, 7.69100314106116e-07, 6.74715432769696e-07],
            [-7.1817349065269e-22, 6.74715432769696e-07, 2.30315190420171e-06],
        ],
    },
    "left_inner_finger_pad": {
        "mass": 0.005,
        "com_position": [0.0, 0.0, 0.0],
        "inertia": [
            [6.027385416666666e-07, 0.0, 0.0],
            [0.0, 7.876041666666668e-07, 0.0],
            [0.0, 0.0, 2.1846770833333335e-07],
        ],
    },
    "right_inner_finger_pad": {
        "mass": 0.005,
        "com_position": [0.0, 0.0, 0.0],
        "inertia": [
            [6.027385416666666e-07, 0.0, 0.0],
            [0.0, 7.876041666666668e-07, 0.0],
            [0.0, 0.0, 2.1846770833333335e-07],
        ],
    },
}


def make_inertial_overlays() -> dict[str, LinkPhysicsOverrideCfg]:
    """Return new typed overlays from the pinned identified/CAD parameters."""
    return {
        name: LinkPhysicsOverrideCfg(
            link_names_expr=[name],
            attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(**values)),
        )
        for name, values in _LINKS.items()
    }
