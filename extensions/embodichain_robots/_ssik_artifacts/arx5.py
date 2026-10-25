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

# Offline reference artifact emitted by SSIK commit
# 4109147f37a7ca3e2516c83853aca583612783f3.
# Generated implementation retains the following upstream license.
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Siddhartha Srinivasa
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Generated IK module for arx5.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 7c9a208118a7 (sha256/12 of the input chain).
``T_target`` is the pose of ``ee_link`` (end-effector link) in
``base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 6    BASE_LINK: "base_link"    EE_LINK: "ee_link"
Solver: ``ikgeo.general_6r`` (tier 2)
Expected median IK time: ~5.0 ms on commodity
single-thread hardware. FLOP budget: 30,000,000 per solve.

Usage:

    import arx5
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of ee_link in base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = arx5.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``arx5.T_HOME`` is the
4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
your robot's home pose, the artifact is for a different URDF.
"""

from __future__ import annotations

import math
from ssik.solvers.ikgeo._raghavan_roth import (
    eliminate_q0_q1 as _ssik_eliminate_q0_q1,
    weierstrass_eliminate_trig as _ssik_weierstrass,
    build_m_matrix as _ssik_build_m_matrix,
    solve_x2_branches as _ssik_solve_x2_branches,
    _back_substitute_inner as _ssik_back_substitute_inner,
    _fk_dh as _ssik_fk_dh,
)

import cython
import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik._solve_inputs import (
    check_joints as _check_joints,
    check_solve_inputs as _check_solve_inputs,
)
from ssik.refinement import lm_refine as _lm_refine
from ssik.refinement.polish import (
    Chain as _PolishChain,
    polish_accepted_flagged as _polish_flagged,
)
from ssik.continuum import (
    NOT_FLAGGED as _NOT_FLAGGED,
    slide_continua as _slide_continua,
)
from ssik.continuum import FREE_FROM_KERNEL as _FREE_FROM_KERNEL, has_null as _has_null
from ssik.refinement import (
    is_same_root as _is_same_root,
    same_root_floor as _same_root_floor,
)
import functools as _functools
from ssik.refinement.rescue import (
    rescue_via_T_perturbation as _rescue_via_T_perturbation,
)
from ssik.postprocess import finalize_solutions as _ps_finalize
from ssik._native import try_native_solve as _try_native_solve
from pathlib import Path as _Path
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.general_6r"
SOLVER_TIER = 2
EXPECTED_MS_MEDIAN = 5.0
FLOP_BUDGET = 30000000
DISPATCH_REASON = "No tier-0 (Pieper-class) match.\nTier-2 numeric Raghavan-Roth + Manocha-Canny pipeline with AE-3 leftvar selection. Closes the EAIK coverage gap (Kinova JACO 2 classical, Agilex Piper, custom non-Pieper 6R)."
BASE_LINK = "base_link"
EE_LINK = "ee_link"
DOF = 6
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array(
    [
        [3.576278899686258e-07, 0.0, 0.999999999999936, 0.15770000405609608],
        [0.0, 1.0, 0.0, -0.0005004054626567813],
        [-0.999999999999936, 0.0, 3.5762788996860295e-07, 0.15649999688664917],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# --- baked KinBody constants ---

_LINK_NAMES = [
    "base_link",
    "_poe_link_1",
    "_poe_link_2",
    "_poe_link_3",
    "_poe_link_4",
    "_poe_link_5",
    "ee_link",
]

_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, -0.9999999999737914, -7.2399802773282935e-06], dtype=np.float64),
    np.array([0.0, -0.9999999999737914, -7.2399802773282935e-06], dtype=np.float64),
    np.array([0.0, 7.2399802773282935e-06, -0.9999999999737914], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.060499999672174454],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.019999999552965164],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.03999999910593033],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, -0.2639999985694885],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.24500000476837158],
            [0.0, 1.0, 0.0, -4.0543890805556755e-07],
            [0.0, 0.0, 1.0, 0.0560000017285347],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0677499994635582],
            [0.0, 1.0, 0.0, -0.0005006262820172703],
            [0.0, 0.0, 1.0, 0.08649999286333565],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.028950000181794167],
            [0.0, 1.0, 0.0, 6.262582685446702e-07],
            [0.0, 0.0, 1.0, -0.08649999648332596],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
]

_JOINT_T_RIGHTS = [
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [3.576278899686258e-07, 0.0, 0.999999999999936, 0.05999999865889549],
            [0.0, 1.0, 0.0, 0.0],
            [-0.999999999999936, 0.0, 3.5762788996860295e-07, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
]

_JOINT_TYPES = [
    "revolute",
    "revolute",
    "revolute",
    "revolute",
    "revolute",
    "revolute",
]

_JOINT_LIMITS = [
    (np.float64(-10.0), np.float64(10.0)),
    (np.float64(-10.0), np.float64(10.0)),
    (np.float64(-10.0), np.float64(10.0)),
    (np.float64(-10.0), np.float64(10.0)),
    (np.float64(-10.0), np.float64(10.0)),
    (np.float64(-10.0), np.float64(10.0)),
]


def _build_kb() -> KinBody:
    """Reconstruct the baked KinBody. Run once at module import."""
    links = [Link(name=n) for n in _LINK_NAMES]
    joints = [
        Joint(
            name=_JOINT_NAMES[i],
            dof_index=i,
            parent_link=links[i],
            T_left=_JOINT_T_LEFTS[i],
            T_right=_JOINT_T_RIGHTS[i],
            axis=_JOINT_AXES[i],
            joint_type=_JOINT_TYPES[i],
            limits=_JOINT_LIMITS[i],
        )
        for i in range(len(_JOINT_NAMES))
    ]
    return KinBody(links=links, joints=joints)


_KB = _build_kb()


# Baked Raghavan-Roth elimination tensor for the native (C++) backend,
# loaded lazily from the sidecar ``.npz`` beside this module (the ~30s
# sympy derivation ran at build time). Cached; ``None`` when the sidecar
# is absent so ``native=True`` silently falls back to Python.
_RR_NATIVE_NPZ = _Path(__file__).with_name(
    _Path(__file__).stem.removesuffix("_ik") + "_rr.npz"
)
_rr_native_geometry_cache = None
_rr_native_geometry_loaded = False


def _rr_native_geometry():
    global _rr_native_geometry_cache, _rr_native_geometry_loaded
    if not _rr_native_geometry_loaded:
        _rr_native_geometry_loaded = True
        if _RR_NATIVE_NPZ.exists():
            from ssik._native import load_rr_native_geometry

            _rr_native_geometry_cache = load_rr_native_geometry(str(_RR_NATIVE_NPZ))
    return _rr_native_geometry_cache


# --- baked DH parameters (from poe_to_dh at build time) ---
_DH_ALPHA = np.array(
    [
        -1.5707963267948966,
        3.141585413609516,
        0.0,
        1.5707963267948966,
        1.5707963267948966,
        3.5762788995923763e-07,
    ],
    dtype=np.float64,
)
_DH_A = np.array(
    [
        0.019999999552965164,
        0.2639999985694885,
        0.2513185280282024,
        0.0677499994635582,
        2.0094982004304744e-22,
        2.303110154843517e-13,
    ],
    dtype=np.float64,
)
_DH_D = np.array(
    [
        0.03999999910593033,
        0.0,
        0.0,
        0.00050000002376183,
        2.775557561490148e-17,
        0.08894999884068966,
    ],
    dtype=np.float64,
)
_DH_TUPLE = (_DH_ALPHA, _DH_A, _DH_D)
_DH_THETA_OFFSET = np.array(
    [
        0.0,
        3.141592653589793,
        -2.9168814826922613,
        -0.22471117089753204,
        1.5707963267948966,
        -9.258070113489503e-19,
    ],
    dtype=np.float64,
)
_T_PRE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.060499999672174454],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_T_POST = np.array(
    [
        [7.23998027732783e-06, -0.9999999999737914, -4.0389678347315804e-28, 0.0],
        [0.9999999999737914, 7.23998027732783e-06, 0.0, 0.0],
        [2.286988957586611e-20, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_T_PRE_INV = np.linalg.inv(_T_PRE)
_T_POST_INV = np.linalg.inv(_T_POST)

_LINEARITY_JOINT = 1
_LEFT_BILINEAR = (2, 3)
_RIGHT_BILINEAR = (0, 5)
_DROP_JOINT = 4
_RR_META = {
    "linearity_joint": _LINEARITY_JOINT,
    "left_bilinear": _LEFT_BILINEAR,
    "right_bilinear": _RIGHT_BILINEAR,
    "drop_joint": _DROP_JOINT,
    "apply_so3": False,
}


# --- inlined per-arm matrix builders (sympy.cse'd) ---
# Per-arm DH constants combine with T_target here. JACO 2's
# 0.866 (cos 60-deg) and friends become explicit coefficients.
# Generic linear algebra (Q-rank, Weierstrass, eigvals,
# back-substitution) stays imported from ssik.solvers.ikgeo.


def _build_pq_matrices(T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11):
    """Build (P_sin, P_cos, P_one, Q) for this T_target. CSE-shared."""
    _pq_x0 = 6.12323399573637e-17 * T_10
    _pq_x1 = 2.18983925362196e-23 * T_9
    _pq_x2 = 3.7493994566544e-33 * T_10
    _pq_x3 = 1.34088981629768e-39 * T_9
    _pq_x4 = 0.088949998840684 * T_10
    _pq_x5 = 3.18110003972718e-8 * T_9
    _pq_x6 = 5.44661656822022e-18 * T_10
    _pq_x7 = 6.12323399573677e-17 * T_11
    _pq_x8 = 1.9478619907097e-24 * T_9
    _pq_x9 = T_0 * T_2
    _pq_x10 = T_10 * T_8
    _pq_x11 = T_1**2
    _pq_x12 = T_10**2
    _pq_x13 = T_2**2
    _pq_x14 = T_9**2
    _pq_x15 = T_0 * T_5
    _pq_x16 = T_0 * T_6
    _pq_x17 = T_1 * T_4
    _pq_x18 = 2.18983925362196e-23 * T_1
    _pq_x19 = T_2 * T_4
    _pq_x20 = 6.12323399573637e-17 * T_2
    _pq_x21 = 2.18983925362196e-23 * T_5
    _pq_x22 = 6.12323399573637e-17 * T_6
    _pq_x23 = 1.34088981629768e-39 * T_1
    _pq_x24 = T_2 * T_5
    _pq_x25 = 4.27642353614751e-50 * T_6
    _pq_x26 = 3.7493994566544e-33 * T_2
    _pq_x27 = 0.00711599974819972 * T_10
    _pq_x28 = 2.54487997489924e-9 * T_9
    _pq_x29 = T_0 * T_1
    _pq_x30 = T_1 * T_2
    _pq_x31 = 5.65917689691665e-9 * _pq_x30
    _pq_x32 = T_10 * T_9
    _pq_x33 = 5.65917689691665e-9 * _pq_x32
    _pq_x34 = T_4 * T_5
    _pq_x35 = T_4 * T_6
    _pq_x36 = T_5 * T_6
    _pq_x37 = 5.65917689691665e-9 * _pq_x36
    _pq_x38 = T_8 * T_9
    _pq_x39 = T_0**2
    _pq_x40 = 5.30431646238053e-26 * _pq_x39
    _pq_x41 = 1.01193974627523e-15 * _pq_x11
    _pq_x42 = 0.00791210229375768 * _pq_x12
    _pq_x43 = T_11**2
    _pq_x44 = 1.0 * _pq_x43
    _pq_x45 = 0.00791210229375768 * _pq_x13
    _pq_x46 = T_3**2
    _pq_x47 = 1.0 * _pq_x46
    _pq_x48 = T_4**2
    _pq_x49 = 5.30431646238053e-26 * _pq_x48
    _pq_x50 = T_5**2
    _pq_x51 = 1.01193974627523e-15 * _pq_x50
    _pq_x52 = T_6**2
    _pq_x53 = 0.00791210229375768 * _pq_x52
    _pq_x54 = T_7**2
    _pq_x55 = 1.0 * _pq_x54
    _pq_x56 = T_8**2
    _pq_x57 = 5.30431646238053e-26 * _pq_x56
    _pq_x58 = 1.01193974627523e-15 * _pq_x14
    _pq_x59 = 2.44929354354857e-18 * T_10
    _pq_x60 = 8.75935681870097e-25 * T_9
    _pq_x61 = 8.75935681870097e-25 * T_1
    _pq_x62 = 2.44929354354857e-18 * T_2
    _pq_x63 = 8.75935681870097e-25 * T_5
    _pq_x64 = 2.44929354354857e-18 * T_6
    _pq_x65 = T_0 * T_10
    _pq_x66 = T_10 * _pq_x16
    _pq_x67 = 3.11150763893057e-61 * T_7
    _pq_x68 = 2.03915764624954e-56 * T_9
    _pq_x69 = T_0 * _pq_x68
    _pq_x70 = T_9 * _pq_x16
    _pq_x71 = T_1 * T_5
    _pq_x72 = T_10 * _pq_x71
    _pq_x73 = T_1 * T_10
    _pq_x74 = 2.81165116044964e-57 * T_6
    _pq_x75 = 2.03915764624954e-56 * T_8
    _pq_x76 = T_8 * _pq_x71
    _pq_x77 = T_1 * T_6
    _pq_x78 = T_8 * _pq_x77
    _pq_x79 = T_8 * _pq_x19
    _pq_x80 = T_8 * _pq_x24
    _pq_x81 = T_6 * T_8
    _pq_x82 = T_2 * T_9
    _pq_x83 = T_2 * T_8
    _pq_x84 = 5.36355914530584e-41 * T_1
    _pq_x85 = 1.49975974913952e-34 * T_2
    _pq_x86 = T_10**3
    _pq_x87 = T_8**3
    _pq_x88 = T_9**3
    _pq_x89 = 4.01899640086059e-22 * T_11
    _pq_x90 = 2.323120046851e-48 * T_8
    _pq_x91 = 2.55696592208827e-29 * T_10
    _pq_x92 = 6.49591408297515e-42 * T_8
    _pq_x93 = T_0 * T_3
    _pq_x94 = 4.01899640086079e-22 * T_9
    _pq_x95 = 3.46524643630859e-25 * T_10
    _pq_x96 = T_1 * T_3
    _pq_x97 = 4.37967850724393e-23 * T_11
    _pq_x98 = 3.57489725197244e-23 * T_8
    _pq_x99 = 1.23926877120586e-31 * T_9
    _pq_x100 = T_2 * T_3
    _pq_x101 = 1.22464679914727e-16 * T_11
    _pq_x102 = T_5 * T_7
    _pq_x103 = T_6 * T_7
    _pq_x104 = T_11 * T_9
    _pq_x105 = T_8 * _pq_x104
    _pq_x106 = 1.15525555467237e-28 * T_8
    _pq_x107 = T_4 * T_7
    _pq_x108 = 4.93771683753601e-18 * T_10
    _pq_x109 = 1.27848296104413e-29 * _pq_x39
    _pq_x110 = 1.76586525373172e-24 * T_9
    _pq_x111 = 6.19634385602931e-32 * T_10
    _pq_x112 = 1.27848296104417e-29 * T_8
    _pq_x113 = 2.21598537869374e-38 * T_9
    _pq_x114 = 4.84476537428808e-19 * T_10
    _pq_x115 = T_10 * _pq_x14
    _pq_x116 = T_8 * _pq_x12
    _pq_x117 = T_9 * _pq_x12
    _pq_x118 = 1.7326232181543e-25 * T_9
    _pq_x119 = 1.27848296104413e-29 * _pq_x48
    _pq_x120 = T_10 * T_11
    _pq_x121 = 2.18899323845425e-39 * T_10
    _pq_x122 = 2.46092553904933e-38 * T_11
    _pq_x123 = T_8 * _pq_x29
    _pq_x124 = 1.56569006600713e-45 * T_9
    _pq_x125 = 1.56569006600713e-45 * T_10
    _pq_x126 = 7.0739000861336e-45 * T_10
    _pq_x127 = 1.05280120714078e-44 * T_11
    _pq_x128 = T_8 * _pq_x9
    _pq_x129 = 2.18899323845425e-39 * T_9
    _pq_x130 = 2.46092553904945e-38 * T_9
    _pq_x131 = T_10 * _pq_x30
    _pq_x132 = 4.07831529249908e-56 * T_10
    _pq_x133 = 4.77088590420656e-40 * T_11
    _pq_x134 = 2.68177963259537e-39 * T_11
    _pq_x135 = 2.18899323845436e-39 * T_8
    _pq_x136 = T_9 * _pq_x30
    _pq_x137 = 2.46092553904945e-38 * T_8
    _pq_x138 = 9.72346137165803e-63 * T_9
    _pq_x139 = T_10 * _pq_x104
    _pq_x140 = 8.55284707229503e-50 * T_10
    _pq_x141 = T_10 * _pq_x36
    _pq_x142 = T_9 * _pq_x10
    _pq_x143 = 7.49879891330881e-33 * T_11
    _pq_x144 = 7.0739000861336e-45 * T_8
    _pq_x145 = T_8 * _pq_x34
    _pq_x146 = T_8 * _pq_x35
    _pq_x147 = T_9 * _pq_x36
    _pq_x148 = 3.02347956009223e-34 * T_10
    _pq_x149 = 3.39907768352814e-33 * T_11
    _pq_x150 = 7.82845033003564e-46 * T_8
    _pq_x151 = T_9 * _pq_x39
    _pq_x152 = T_10 * _pq_x11
    _pq_x153 = 8.53100929578867e-47 * T_11
    _pq_x154 = 7.82845033003584e-46 * T_8
    _pq_x155 = 1.35689970048731e-54 * T_9
    _pq_x156 = 2.96656320412091e-35 * T_10
    _pq_x157 = 3.79416633485133e-48 * _pq_x50
    _pq_x158 = T_11 * _pq_x12
    _pq_x159 = 6.67018154645341e-34 * T_11
    _pq_x160 = T_11 * _pq_x14
    _pq_x161 = 6.29223404460698e-46 * T_8
    _pq_x162 = T_9 * _pq_x13
    _pq_x163 = 1.08128061535404e-40 * _pq_x48
    _pq_x164 = 1.06092573912052e-41 * _pq_x52
    _pq_x165 = T_8 * _pq_x14
    _pq_x166 = T_9 * _pq_x56
    _pq_x167 = 1.0 * T_4
    _pq_x168 = 0.999999999999936 * T_5
    _pq_x169 = 3.5762788995923e-7 * T_6
    _pq_x170 = _pq_x168 - _pq_x169
    _pq_x171 = 1.0 * T_0
    _pq_x172 = 0.999999999999936 * T_1
    _pq_x173 = 3.5762788995923e-7 * T_2
    _pq_x174 = _pq_x172 - _pq_x173
    _pq_x175 = _pq_x21 + _pq_x22
    _pq_x176 = _pq_x18 + _pq_x20
    _pq_x177 = 6.12323399573677e-17 * T_0
    _pq_x178 = 6.12323399573637e-17 * T_1
    _pq_x179 = 6.12323399573677e-17 * T_4
    _pq_x180 = 6.12323399573637e-17 * T_5
    _pq_x181 = 2.18983925362196e-23 * T_6
    _pq_x182 = 1.34088981629768e-39 * T_5
    _pq_x183 = 3.7493994566544e-33 * T_6
    _pq_x184 = 1.0 * T_8
    _pq_x185 = 3.5762788995923e-7 * T_10
    _pq_x186 = 0.999999999999936 * T_9
    _pq_x187 = 6.12323399573677e-17 * T_8
    _pq_x188 = 2.18983925362196e-23 * T_10
    _pq_x189 = (
        -2.77555756149015e-17 * T_4
        + 2.00949820043035e-22 * T_5
        - 7.1865260129682e-29 * T_6
    )
    _pq_x190 = (
        -2.00949820043047e-22 * T_4
        - 2.77555756148997e-17 * T_5
        + 9.92616794176107e-24 * T_6
    )
    _pq_x191 = (
        2.77555756149015e-17 * T_0
        - 2.00949820043035e-22 * T_1
        + 7.1865260129682e-29 * T_2
    )
    _pq_x192 = (
        2.00949820043047e-22 * T_0
        + 2.77555756148997e-17 * T_1
        - 9.92616794176107e-24 * T_2
    )
    _pq_x193 = 3.18110003972718e-8 * T_5
    _pq_x194 = 0.088949998840684 * T_6
    _pq_x195 = -2.30311015484352e-13 * T_4 + 1.0 * T_7 - _pq_x193 - _pq_x194
    _pq_x196 = 3.18110003972718e-8 * T_1
    _pq_x197 = 0.088949998840684 * T_2
    _pq_x198 = 2.30311015484352e-13 * T_0 - 1.0 * T_3 + _pq_x196 + _pq_x197
    _pq_x199 = 1.9478619907097e-24 * T_1
    _pq_x200 = 5.44661656822022e-18 * T_2
    _pq_x201 = 1.9478619907097e-24 * T_5
    _pq_x202 = 5.44661656822022e-18 * T_6
    _pq_x203 = 0.0399999991059303 * T_0
    _pq_x204 = 1.0 * T_11
    _pq_x205 = (
        -T_0 * _pq_x204
        + T_0 * _pq_x4
        + T_0 * _pq_x5
        + T_3 * _pq_x184
        - T_8 * _pq_x196
        - T_8 * _pq_x197
        + _pq_x203
        + 1.23046276952477e-38 * _pq_x73
        - 1.23046276952477e-38 * _pq_x82
    )
    _pq_x206 = 0.0399999991059278 * T_1
    _pq_x207 = 1.43051152786249e-8 * T_2
    _pq_x208 = 8.23656425020362e-20 * T_10
    _pq_x209 = 2.30311015484337e-13 * T_9
    _pq_x210 = 2.30311015484337e-13 * T_8
    _pq_x211 = 8.23656425020362e-20 * T_8
    _pq_x212 = (
        T_0 * _pq_x208
        - T_0 * _pq_x209
        + T_1 * _pq_x210
        - T_11 * _pq_x172
        + T_11 * _pq_x173
        - T_2 * _pq_x211
        - T_3 * _pq_x185
        + T_3 * _pq_x186
        + _pq_x206
        - _pq_x207
        + 0.0889499988406897 * _pq_x73
        - 0.0889499988406897 * _pq_x82
    )
    _pq_x213 = 0.0399999991059303 * T_4
    _pq_x214 = T_4 * _pq_x4
    _pq_x215 = T_10 * T_5
    _pq_x216 = 1.23046276952477e-38 * _pq_x215
    _pq_x217 = T_4 * _pq_x204
    _pq_x218 = T_4 * _pq_x5
    _pq_x219 = T_8 * _pq_x193
    _pq_x220 = T_8 * _pq_x194
    _pq_x221 = T_6 * T_9
    _pq_x222 = 1.23046276952477e-38 * _pq_x221
    _pq_x223 = T_7 * _pq_x184
    _pq_x224 = 0.0399999991059278 * T_5
    _pq_x225 = 1.43051152786249e-8 * T_6
    _pq_x226 = T_4 * _pq_x208
    _pq_x227 = 0.0889499988406897 * _pq_x215
    _pq_x228 = T_7 * _pq_x185
    _pq_x229 = T_11 * _pq_x168
    _pq_x230 = T_11 * _pq_x169
    _pq_x231 = T_4 * _pq_x209
    _pq_x232 = T_5 * _pq_x210
    _pq_x233 = T_6 * _pq_x211
    _pq_x234 = 0.0889499988406897 * _pq_x221
    _pq_x235 = T_7 * _pq_x186
    _pq_x236 = 5.77627777336186e-29 * T_10
    _pq_x237 = 2.0094982004304e-22 * T_9
    _pq_x238 = 2.0094982004304e-22 * T_8
    _pq_x239 = 5.77627777336186e-29 * T_8
    _pq_x240 = (
        T_0 * _pq_x236
        - T_0 * _pq_x237
        + T_1 * _pq_x238
        - T_11 * _pq_x18
        - T_11 * _pq_x20
        - T_2 * _pq_x239
        + T_3 * _pq_x0
        + T_3 * _pq_x1
        + _pq_x61
        + _pq_x62
    )
    _pq_x241 = T_4 * _pq_x236
    _pq_x242 = T_7 * _pq_x0
    _pq_x243 = T_11 * _pq_x21
    _pq_x244 = T_11 * _pq_x22
    _pq_x245 = T_4 * _pq_x237
    _pq_x246 = T_5 * _pq_x238
    _pq_x247 = T_6 * _pq_x239
    _pq_x248 = T_7 * _pq_x1
    _pq_x249 = 0.0199999995529652 * T_0
    _pq_x250 = 2.44929354354873e-18 * T_4
    _pq_x251 = 3.45446742203778e-77 * T_9
    _pq_x252 = 2.44929354354857e-18 * T_5
    _pq_x253 = 8.75935681870097e-25 * T_6
    _pq_x254 = 1.41024823960635e-29 * T_8
    _pq_x255 = 6.12323399573637e-17 * T_9
    _pq_x256 = 5.04344102249169e-36 * T_10
    _pq_x257 = 1.41024823960635e-29 * T_9
    _pq_x258 = 5.04344102249169e-36 * T_8
    _pq_x259 = 0.0199999995529639 * T_1 - 7.15255763931247e-9 * T_2
    _pq_x260 = 2.44929354354873e-18 * T_0
    _pq_x261 = -0.0199999995529652 * T_4
    _pq_x262 = 2.44929354354857e-18 * T_1
    _pq_x263 = 8.75935681870097e-25 * T_2
    _pq_x264 = 2.18983925362196e-23 * T_2
    _pq_x265 = -0.0199999995529639 * T_5 + 7.15255763931247e-9 * T_6
    _pq_x266 = 1.49975974913952e-34 * T_6
    _pq_x267 = 5.36355914530584e-41 * T_5
    _pq_x268 = 3.5369500430668e-45 * T_10
    _pq_x269 = 1.23046276952473e-38 * T_8
    _pq_x270 = 4.86173068582902e-63 * T_9
    _pq_x271 = 3.5369500430668e-45 * T_8
    _pq_x272 = 1.23046276952473e-38 * T_9
    _pq_x273 = 4.37967840935049e-25 * T_1 + 1.22464677177429e-18 * T_2
    _pq_x274 = 4.37967840935049e-25 * T_5
    _pq_x275 = 1.22464677177429e-18 * T_6
    _pq_x276 = T_10 * T_2
    _pq_x277 = 2.82989971213335e-73 * T_10
    _pq_x278 = 1.27243998744962e-9 * T_5
    _pq_x279 = 0.00355799987409986 * T_6
    _pq_x280 = 0.0399999991059303 * T_7
    _pq_x281 = 3.50324616081204e-46 * T_10
    _pq_x282 = 1.67047794380762e-52 * T_9
    _pq_x283 = 1.67047794380762e-52 * T_8
    _pq_x284 = 3.50324616081204e-46 * T_8
    _pq_x285 = 1.27243998744962e-9 * T_1
    _pq_x286 = 0.00355799987409986 * T_2
    _pq_x287 = T_11 * T_8
    _pq_x288 = 1.27848296104413e-29 * _pq_x56
    _pq_x289 = 2.55696592208827e-29 * _pq_x38
    _pq_x290 = 6.68191177523049e-52 * T_10
    _pq_x291 = 0.0399999991059278 * T_9
    _pq_x292 = T_6**3
    _pq_x293 = T_4**3
    _pq_x294 = T_5**3
    _pq_x295 = 1.00190545973564e-58 * T_6
    _pq_x296 = 0.0799999982118607 * T_11
    _pq_x297 = 7.82845033003514e-46 * T_6
    _pq_x298 = 1.46528476101376e-20 * T_5
    _pq_x299 = 4.09723291206597e-14 * T_6
    _pq_x300 = T_5 * T_8
    _pq_x301 = 8.80095607877014e-45 * T_7
    _pq_x302 = 2.0 * T_7
    _pq_x303 = 5.59544858130873e-52 * T_6
    _pq_x304 = 2.46092553904892e-38 * T_7
    _pq_x305 = T_1 * _pq_x15
    _pq_x306 = T_5 * _pq_x38
    _pq_x307 = 5.65917689691665e-9 * T_1
    _pq_x308 = 5.65917689691665e-9 * T_5
    _pq_x309 = T_6 * _pq_x38
    _pq_x310 = 0.0158242045875154 * T_6
    _pq_x311 = T_6 * _pq_x102
    _pq_x312 = T_5 * _pq_x120
    _pq_x313 = T_3 * _pq_x24
    _pq_x314 = 6.36220007945436e-8 * T_3
    _pq_x315 = 6.36220007945436e-8 * _pq_x104
    _pq_x316 = 0.177899997681368 * _pq_x120
    _pq_x317 = 0.177899997681368 * T_3
    _pq_x318 = (
        -6.36220007945436e-8 * T_0 * T_1 * T_7
        - 0.177899997681368 * T_0 * T_2 * T_7
        - 6.36220007945436e-8 * T_0 * T_3 * T_5
        - 0.177899997681368 * T_0 * T_3 * T_6
        - 5.65917689691665e-9 * T_1 * T_2 * T_4
        - 1.56569006600703e-45 * T_1 * T_2 * T_5
        - 2.46092553904955e-38 * T_1 * T_3 * T_6
        - 5.65917689691665e-9 * T_10 * T_4 * T_9
        - 0.00711599974819972 * T_10 * T_4
        - 1.56569006600703e-45 * T_10 * T_5 * T_9
        - 9.8437019361743e-40 * T_10 * T_5
        - 0.177899997681368 * T_10 * T_7 * T_8
        - 6.36220007945436e-8 * T_11 * T_5 * T_8
        - 0.177899997681368 * T_11 * T_6 * T_8
        - 2.46092553904955e-38 * T_11 * T_6 * T_9
        - 6.36220007945436e-8 * T_4 * T_5 * T_7
        - 0.177899997681368 * T_4 * T_6 * T_7
        - 2.54487997489924e-9 * T_4 * T_9
        - 1.01193974627523e-15 * T_4 * _pq_x11
        - 0.00791210229375768 * T_4 * _pq_x12
        - 0.00791210229375768 * T_4 * _pq_x13
        - 1.01193974627523e-15 * T_4 * _pq_x14
        + T_4 * _pq_x296
        + T_4 * _pq_x315
        + T_4 * _pq_x316
        + T_4 * _pq_x37
        + T_4 * _pq_x40
        - 1.0 * T_4 * _pq_x43
        - 1.0 * T_4 * _pq_x46
        + T_4 * _pq_x51
        + T_4 * _pq_x53
        + T_4 * _pq_x55
        + T_4 * _pq_x57
        - 2.79967217318129e-52 * T_5 * _pq_x11
        - 2.18899323845439e-39 * T_5 * _pq_x12
        - 2.18899323845439e-39 * T_5 * _pq_x13
        - 2.79967217318129e-52 * T_5 * _pq_x14
        - 2.18899323845383e-39 * T_5 * _pq_x52
        - 1.56569006600693e-45 * T_6 * _pq_x50
        - 6.36220007945436e-8 * T_7 * T_8 * T_9
        - 0.0799999982118607 * T_7 * T_8
        - 8.80095607877014e-45 * T_7 * _pq_x12
        - 8.80095607877014e-45 * T_7 * _pq_x13
        - 4.60622030968703e-13 * T_7 * _pq_x39
        - 4.60622030968703e-13 * T_7 * _pq_x48
        - 8.80095607877014e-45 * T_7 * _pq_x52
        - 4.60622030968703e-13 * T_7 * _pq_x56
        + _pq_x10 * _pq_x308
        + _pq_x10 * _pq_x310
        + _pq_x11 * _pq_x295
        + _pq_x11 * _pq_x301
        + _pq_x12 * _pq_x297
        + _pq_x13 * _pq_x297
        + _pq_x14 * _pq_x295
        + _pq_x14 * _pq_x301
        + _pq_x16 * _pq_x307
        + _pq_x17 * _pq_x314
        + _pq_x19 * _pq_x317
        + 9.8437019361743e-40 * _pq_x221
        + _pq_x287 * _pq_x302
        + 7.82845033003514e-46 * _pq_x292
        + 5.30431646238053e-26 * _pq_x293
        - 2.79967217318129e-52 * _pq_x294
        + _pq_x298 * _pq_x39
        + _pq_x298 * _pq_x48
        + _pq_x298 * _pq_x56
        + _pq_x299 * _pq_x39
        + _pq_x299 * _pq_x48
        + _pq_x299 * _pq_x56
        + _pq_x30 * _pq_x303
        + _pq_x30 * _pq_x304
        + 2.54487997489924e-9 * _pq_x300
        + _pq_x301 * _pq_x50
        + _pq_x302 * _pq_x93
        + _pq_x303 * _pq_x32
        + _pq_x304 * _pq_x32
        + 2.02387949255045e-15 * _pq_x305
        + 2.02387949255045e-15 * _pq_x306
        + _pq_x308 * _pq_x9
        + 5.65917689691665e-9 * _pq_x309
        + _pq_x310 * _pq_x9
        + 2.46092553904892e-38 * _pq_x311
        + 2.46092553904955e-38 * _pq_x312
        + 2.46092553904955e-38 * _pq_x313
        + 0.00711599974819972 * _pq_x81
    )
    _pq_x319 = T_5 * _pq_x48
    _pq_x320 = 6.36220007945395e-8 * T_7
    _pq_x321 = 1.01193974627516e-15 * T_5
    _pq_x322 = 1.84248808269188e-14 * T_9
    _pq_x323 = T_6 * _pq_x50
    _pq_x324 = 1.11549558500113e-38 * T_4
    _pq_x325 = T_5 * _pq_x52
    _pq_x326 = 1.46528476101366e-20 * T_4
    _pq_x327 = 0.0799999982118555 * T_11
    _pq_x328 = 5.65917689691665e-9 * T_6
    _pq_x329 = 1.89697144901585e-32 * T_6
    _pq_x330 = 2.86102305572499e-8 * T_10
    _pq_x331 = 1.99999999999987 * T_7
    _pq_x332 = 1.64731285004072e-19 * T_7
    _pq_x333 = 1.64731285004072e-19 * T_4
    _pq_x334 = 1.64731285004072e-19 * T_3
    _pq_x335 = 0.0158242045875143 * T_6
    _pq_x336 = 4.60622030968674e-13 * T_3
    _pq_x337 = T_5 * _pq_x287
    _pq_x338 = 1.06086329247604e-25 * T_0
    _pq_x339 = 1.06086329247604e-25 * T_4
    _pq_x340 = 4.09723291206518e-14 * T_1
    _pq_x341 = 4.09723291206518e-14 * T_4
    _pq_x342 = 4.09723291206623e-14 * T_1
    _pq_x343 = (
        -2.23099117000197e-38 * T_0 * T_1 * T_5
        - 4.60622030968674e-13 * T_0 * T_1 * T_7
        - 3.79394300823431e-32 * T_0 * T_2 * T_4
        - 4.09723291206623e-14 * T_0 * T_2 * T_5
        - 3.00926553810506e-36 * T_0 * T_2 * T_6
        - 1.64731285004072e-19 * T_0 * T_3 * T_6
        - 7.23783035983899e-22 * T_1 * T_2 * T_5
        - 0.177899997681334 * T_1 * T_2 * T_7
        - 4.60622030968674e-13 * T_1 * T_3 * T_4
        - 0.177899997681379 * T_1 * T_3 * T_6
        - 7.1525577991846e-7 * T_10 * T_11 * T_7
        - 3.79394300823431e-32 * T_10 * T_4 * T_8
        - 6.58925125288165e-21 * T_10 * T_4
        - 4.09723291206623e-14 * T_10 * T_5 * T_8
        - 7.23783035983899e-22 * T_10 * T_5 * T_9
        - 0.00711599974820018 * T_10 * T_5
        - 3.00926553810506e-36 * T_10 * T_6 * T_8
        - 4.13590306276514e-25 * T_10 * T_6
        - 0.177899997681334 * T_10 * T_7 * T_9
        - 4.60622030968674e-13 * T_11 * T_4 * T_9
        - 1.64731285004072e-19 * T_11 * T_6 * T_8
        - 0.177899997681379 * T_11 * T_6 * T_9
        - 2.86102305572499e-8 * T_11 * T_6
        - 7.1525577991846e-7 * T_2 * T_3 * T_7
        - 4.60622030968674e-13 * T_4 * T_5 * T_7
        - 1.46528476101366e-20 * T_4 * _pq_x12
        - 1.46528476101366e-20 * T_4 * _pq_x13
        + T_4 * _pq_x322
        - 1.46528476101366e-20 * T_4 * _pq_x52
        - 0.177899997681334 * T_5 * T_6 * T_7
        - 2.23099117000197e-38 * T_5 * T_8 * T_9
        - 1.84248808269188e-14 * T_5 * T_8
        - 0.0079121022937592 * T_5 * _pq_x12
        - 0.0079121022937592 * T_5 * _pq_x13
        + T_5 * _pq_x327
        - 5.30431630830579e-26 * T_5 * _pq_x39
        - 0.999999999999936 * T_5 * _pq_x43
        - 0.999999999999936 * T_5 * _pq_x46
        - 5.30431630830579e-26 * T_5 * _pq_x56
        - 2.82958844845814e-9 * T_6 * _pq_x12
        - 2.82958844845814e-9 * T_6 * _pq_x13
        - 1.89697155921846e-32 * T_6 * _pq_x48
        - 3.5762788995923e-7 * T_6 * _pq_x54
        - 4.60622030968674e-13 * T_7 * T_8 * T_9
        - 0.0799999982118555 * T_7 * T_9
        - 6.36220007945395e-8 * T_7 * _pq_x11
        - 6.36220007945395e-8 * T_7 * _pq_x14
        + T_7 * _pq_x330
        - 6.36220007945395e-8 * T_7 * _pq_x50
        + _pq_x10 * _pq_x332
        + _pq_x103 * _pq_x333
        + _pq_x104 * _pq_x331
        + _pq_x11 * _pq_x321
        + _pq_x11 * _pq_x326
        + _pq_x11 * _pq_x328
        + _pq_x12 * _pq_x320
        + _pq_x120 * _pq_x333
        + _pq_x13 * _pq_x320
        + _pq_x14 * _pq_x321
        + _pq_x14 * _pq_x326
        + _pq_x14 * _pq_x328
        + _pq_x15 * _pq_x336
        + _pq_x16 * _pq_x342
        + _pq_x168 * _pq_x54
        + _pq_x169 * _pq_x43
        + _pq_x169 * _pq_x46
        + _pq_x17 * _pq_x338
        + _pq_x19 * _pq_x334
        + _pq_x19 * _pq_x340
        + 0.00711599974820018 * _pq_x221
        - 2.82958844845814e-9 * _pq_x292
        + 1.11549558500113e-38 * _pq_x293
        + 1.01193974627516e-15 * _pq_x294
        + _pq_x30 * _pq_x335
        + 4.09723291206623e-14 * _pq_x309
        + 0.177899997681379 * _pq_x312
        + 0.177899997681379 * _pq_x313
        + 5.30431661645458e-26 * _pq_x319
        + _pq_x32 * _pq_x335
        + _pq_x32 * _pq_x341
        + _pq_x320 * _pq_x52
        + 5.65917689691592e-9 * _pq_x323
        + _pq_x324 * _pq_x39
        + _pq_x324 * _pq_x56
        + 0.00791210229375515 * _pq_x325
        + _pq_x326 * _pq_x50
        + _pq_x329 * _pq_x39
        + _pq_x329 * _pq_x56
        + _pq_x331 * _pq_x96
        + _pq_x332 * _pq_x9
        + 4.60622030968674e-13 * _pq_x337
        + _pq_x339 * _pq_x38
        + _pq_x341 * _pq_x36
        + 6.58925125288165e-21 * _pq_x81
    )
    _pq_x344 = T_2 * _pq_x96
    _pq_x345 = T_2**3
    _pq_x346 = T_0**3
    _pq_x347 = T_1**3
    _pq_x348 = 1.00190545973564e-58 * T_2
    _pq_x349 = 7.82845033003514e-46 * T_2
    _pq_x350 = 1.46528476101376e-20 * T_1
    _pq_x351 = 4.09723291206597e-14 * T_2
    _pq_x352 = T_1 * T_8
    _pq_x353 = 8.80095607877014e-45 * T_3
    _pq_x354 = T_2 * _pq_x11
    _pq_x355 = T_1 * _pq_x13
    _pq_x356 = 4.60622030968703e-13 * T_3
    _pq_x357 = 2.18899323845439e-39 * T_1
    _pq_x358 = 2.79967217318129e-52 * T_1
    _pq_x359 = 2.0 * T_3
    _pq_x360 = T_2 * _pq_x32
    _pq_x361 = 2.46092553904892e-38 * T_3
    _pq_x362 = T_5 * _pq_x17
    _pq_x363 = T_1 * _pq_x38
    _pq_x364 = 5.65917689691665e-9 * T_6
    _pq_x365 = T_2 * _pq_x38
    _pq_x366 = T_2 * _pq_x10
    _pq_x367 = 2.46092553904955e-38 * T_1
    _pq_x368 = 6.36220007945436e-8 * T_7
    _pq_x369 = 0.177899997681368 * T_7
    _pq_x370 = T_1 * _pq_x32
    _pq_x371 = T_6 * _pq_x71
    _pq_x372 = T_2 * _pq_x104
    _pq_x373 = T_7 * _pq_x24
    _pq_x374 = 6.36220007945436e-8 * _pq_x96
    _pq_x375 = T_1 * _pq_x287
    _pq_x376 = T_2 * _pq_x287
    _pq_x377 = (
        -T_0 * _pq_x27
        - T_0 * _pq_x28
        + T_0 * _pq_x296
        + T_0 * _pq_x315
        + T_0 * _pq_x316
        - T_0 * _pq_x33
        - T_0 * _pq_x374
        + T_0 * _pq_x41
        - T_0 * _pq_x42
        - T_0 * _pq_x44
        + T_0 * _pq_x45
        + T_0 * _pq_x47
        + T_0 * _pq_x49
        - T_0 * _pq_x51
        - T_0 * _pq_x53
        - T_0 * _pq_x55
        + T_0 * _pq_x57
        - T_0 * _pq_x58
        - 0.0799999982118607 * T_3 * T_8
        + _pq_x10 * _pq_x307
        - _pq_x10 * _pq_x317
        + _pq_x103 * _pq_x367
        + _pq_x107 * _pq_x359
        + _pq_x11 * _pq_x353
        + _pq_x12 * _pq_x349
        - _pq_x12 * _pq_x353
        - _pq_x12 * _pq_x357
        + _pq_x120 * _pq_x367
        - _pq_x13 * _pq_x353
        + _pq_x14 * _pq_x348
        + _pq_x14 * _pq_x353
        - _pq_x14 * _pq_x358
        - _pq_x15 * _pq_x364
        + _pq_x15 * _pq_x368
        + _pq_x16 * _pq_x369
        + _pq_x17 * _pq_x364
        - _pq_x17 * _pq_x368
        + _pq_x19 * _pq_x308
        + _pq_x19 * _pq_x310
        - _pq_x19 * _pq_x369
        + _pq_x24 * _pq_x303
        + _pq_x287 * _pq_x359
        + _pq_x307 * _pq_x9
        - _pq_x314 * _pq_x34
        - _pq_x314 * _pq_x38
        - _pq_x317 * _pq_x35
        - _pq_x317 * _pq_x9
        + _pq_x32 * _pq_x361
        + 7.82845033003514e-46 * _pq_x345
        + 5.30431646238053e-26 * _pq_x346
        - 2.79967217318129e-52 * _pq_x347
        + _pq_x348 * _pq_x50
        + _pq_x349 * _pq_x52
        + _pq_x350 * _pq_x39
        + _pq_x350 * _pq_x48
        + _pq_x350 * _pq_x56
        + _pq_x351 * _pq_x39
        + _pq_x351 * _pq_x48
        + _pq_x351 * _pq_x56
        + 2.54487997489924e-9 * _pq_x352
        + _pq_x353 * _pq_x50
        - _pq_x353 * _pq_x52
        - 1.56569006600693e-45 * _pq_x354
        - 2.18899323845383e-39 * _pq_x355
        - _pq_x356 * _pq_x39
        - _pq_x356 * _pq_x48
        - _pq_x356 * _pq_x56
        - _pq_x357 * _pq_x52
        - _pq_x358 * _pq_x50
        + _pq_x36 * _pq_x361
        + 5.59544858130873e-52 * _pq_x360
        + 2.02387949255045e-15 * _pq_x362
        + 2.02387949255045e-15 * _pq_x363
        + 5.65917689691665e-9 * _pq_x365
        + 0.0158242045875154 * _pq_x366
        - 1.56569006600703e-45 * _pq_x370
        - 1.56569006600703e-45 * _pq_x371
        - 2.46092553904955e-38 * _pq_x372
        - 2.46092553904955e-38 * _pq_x373
        - 6.36220007945436e-8 * _pq_x375
        - 0.177899997681368 * _pq_x376
        - 9.8437019361743e-40 * _pq_x73
        + 9.8437019361743e-40 * _pq_x82
        + 0.00711599974819972 * _pq_x83
    )
    _pq_x378 = T_10 * T_4
    _pq_x379 = T_1 * _pq_x39
    _pq_x380 = 6.36220007945395e-8 * T_3
    _pq_x381 = 1.01193974627516e-15 * T_1
    _pq_x382 = 1.11549558500113e-38 * T_0
    _pq_x383 = 1.46528476101366e-20 * T_0
    _pq_x384 = 5.65917689691665e-9 * T_2
    _pq_x385 = 1.89697144901585e-32 * T_2
    _pq_x386 = 5.30431630830579e-26 * T_1
    _pq_x387 = 1.46528476101366e-20 * T_0
    _pq_x388 = 0.0079121022937592 * T_1
    _pq_x389 = 6.36220007945395e-8 * T_3
    _pq_x390 = 2.82958844845814e-9 * T_2
    _pq_x391 = T_2 * _pq_x39
    _pq_x392 = 0.177899997681379 * T_1
    _pq_x393 = 1.99999999999987 * T_3
    _pq_x394 = T_0 * _pq_x120
    _pq_x395 = 4.60622030968674e-13 * T_7
    _pq_x396 = T_0 * _pq_x32
    _pq_x397 = T_6 * _pq_x15
    _pq_x398 = 0.177899997681334 * T_3
    _pq_x399 = T_0 * _pq_x10
    _pq_x400 = 7.1525577991846e-7 * T_3
    _pq_x401 = 4.60622030968674e-13 * T_0
    _pq_x402 = T_6 * _pq_x17
    _pq_x403 = (
        T_0 * _pq_x322
        + T_1 * _pq_x327
        - 2.86102305572499e-8 * T_11 * T_2
        - 0.0799999982118555 * T_3 * T_9
        + T_3 * _pq_x330
        - 3.79394300823431e-32 * T_4 * _pq_x16
        + 4.09723291206623e-14 * T_5 * _pq_x19
        - 3.00926553810506e-36 * T_6 * _pq_x19
        + _pq_x10 * _pq_x334
        - _pq_x10 * _pq_x342
        + _pq_x102 * _pq_x393
        + _pq_x103 * _pq_x392
        - _pq_x103 * _pq_x400
        + _pq_x104 * _pq_x393
        - _pq_x104 * _pq_x401
        + _pq_x11 * _pq_x383
        - _pq_x11 * _pq_x389
        + _pq_x12 * _pq_x380
        - _pq_x12 * _pq_x387
        - _pq_x12 * _pq_x388
        - _pq_x12 * _pq_x390
        + _pq_x120 * _pq_x392
        - _pq_x120 * _pq_x400
        + _pq_x13 * _pq_x380
        - _pq_x13 * _pq_x387
        + _pq_x14 * _pq_x381
        + _pq_x14 * _pq_x383
        + _pq_x14 * _pq_x384
        - _pq_x14 * _pq_x389
        + _pq_x15 * _pq_x339
        - _pq_x15 * _pq_x395
        + _pq_x16 * _pq_x332
        + _pq_x17 * _pq_x395
        - _pq_x172 * _pq_x43
        + _pq_x172 * _pq_x46
        - _pq_x172 * _pq_x54
        + _pq_x173 * _pq_x43
        - _pq_x173 * _pq_x46
        + _pq_x173 * _pq_x54
        - _pq_x19 * _pq_x332
        + _pq_x24 * _pq_x335
        - 4.13590306276514e-25 * _pq_x276
        - _pq_x32 * _pq_x398
        + _pq_x334 * _pq_x35
        + _pq_x334 * _pq_x9
        - _pq_x336 * _pq_x34
        - _pq_x336 * _pq_x38
        + _pq_x338 * _pq_x38
        + _pq_x340 * _pq_x9
        - 0.177899997681334 * _pq_x344
        - 2.82958844845814e-9 * _pq_x345
        + 1.11549558500113e-38 * _pq_x346
        + 1.01193974627516e-15 * _pq_x347
        - 1.84248808269188e-14 * _pq_x352
        + 5.65917689691592e-9 * _pq_x354
        + 0.00791210229375515 * _pq_x355
        - _pq_x36 * _pq_x398
        + 0.0158242045875143 * _pq_x360
        - 2.23099117000197e-38 * _pq_x362
        - 2.23099117000197e-38 * _pq_x363
        + 4.09723291206623e-14 * _pq_x365
        - 3.00926553810506e-36 * _pq_x366
        - 7.23783035983899e-22 * _pq_x370
        - 7.23783035983899e-22 * _pq_x371
        - 0.177899997681379 * _pq_x372
        - 0.177899997681379 * _pq_x373
        + 4.60622030968674e-13 * _pq_x375
        - 1.64731285004072e-19 * _pq_x376
        + 5.30431661645458e-26 * _pq_x379
        + _pq_x380 * _pq_x52
        + _pq_x381 * _pq_x50
        + _pq_x382 * _pq_x48
        + _pq_x382 * _pq_x56
        + _pq_x383 * _pq_x50
        + _pq_x384 * _pq_x50
        + _pq_x385 * _pq_x48
        + _pq_x385 * _pq_x56
        - _pq_x386 * _pq_x48
        - _pq_x386 * _pq_x56
        - _pq_x387 * _pq_x52
        - _pq_x388 * _pq_x52
        - _pq_x389 * _pq_x50
        - _pq_x390 * _pq_x52
        - 1.89697155921846e-32 * _pq_x391
        + 1.64731285004072e-19 * _pq_x394
        + 4.09723291206518e-14 * _pq_x396
        + 4.09723291206518e-14 * _pq_x397
        - 3.79394300823431e-32 * _pq_x399
        - _pq_x401 * _pq_x96
        - 4.09723291206623e-14 * _pq_x402
        - 6.58925125288165e-21 * _pq_x65
        - 0.00711599974820018 * _pq_x73
        + 0.00711599974820018 * _pq_x82
        + 6.58925125288165e-21 * _pq_x83
    )
    _pq_x404 = 1.09476442525376e-47 * T_9
    _pq_x405 = 1.60759852441169e-23 * T_9
    _pq_x406 = 4.89858708709714e-18 * T_11
    _pq_x407 = 1.27848296104417e-29 * T_4
    _pq_x408 = 1.75187136374019e-24 * T_11
    _pq_x409 = 1.76586525373172e-24 * T_5
    _pq_x410 = 4.84476537428808e-19 * T_6
    _pq_x411 = T_4 * _pq_x52
    _pq_x412 = 6.19634385602931e-32 * T_6
    _pq_x413 = 4.93771683753601e-18 * T_6
    _pq_x414 = T_6 * _pq_x48
    _pq_x415 = 1.7326232181543e-25 * T_5
    _pq_x416 = 2.21598537869374e-38 * T_5
    _pq_x417 = T_4 * _pq_x120
    _pq_x418 = T_1 * _pq_x19
    _pq_x419 = T_4 * _pq_x32
    _pq_x420 = 3.46524643630859e-25 * T_6
    _pq_x421 = 2.55696592208827e-29 * T_6
    _pq_x422 = 1.22464679914727e-16 * T_7
    _pq_x423 = 6.49591408297515e-42 * T_4
    _pq_x424 = T_2 * _pq_x71
    _pq_x425 = T_5 * _pq_x32
    _pq_x426 = 4.37967850724393e-23 * T_7
    _pq_x427 = 2.323120046851e-48 * T_0
    _pq_x428 = 2.323120046851e-48 * T_4
    _pq_x429 = 4.01899640086059e-22 * T_7
    _pq_x430 = T_4 * _pq_x102
    _pq_x431 = (
        -2.55696592208827e-29 * T_0 * T_1 * T_5
        - 3.57489725197226e-23 * T_0 * T_1 * T_6
        - 3.57489725197226e-23 * T_0 * T_2 * T_5
        - 1.71935485051491e-28 * T_0 * T_2 * T_7
        - 1.15525555467237e-28 * T_0 * T_3 * T_6
        - 7.79144796283832e-24 * T_1 * T_2 * T_7
        - 4.01899640086079e-22 * T_1 * T_3 * T_4
        - 4.62102211540159e-30 * T_10 * T_4
        - 3.57489725197226e-23 * T_10 * T_5 * T_8
        - 1.71935485051491e-28 * T_10 * T_7 * T_8
        - 7.79144796283832e-24 * T_10 * T_7 * T_9
        - 4.89858708709714e-18 * T_10 * T_7
        - 4.01899640086079e-22 * T_11 * T_4 * T_9
        - 1.15525555467237e-28 * T_11 * T_6 * T_8
        + 4.01899640086079e-22 * T_3 * _pq_x15
        + 1.15525555467237e-28 * T_3 * _pq_x19
        - 3.57489725197208e-23 * T_4 * T_5 * T_6
        - 1.71935485051491e-28 * T_4 * T_6 * T_7
        + T_4 * _pq_x109
        - 1.02759980248801e-29 * T_4 * _pq_x12
        - 1.02759980248801e-29 * T_4 * _pq_x13
        + T_4 * _pq_x288
        + T_4 * _pq_x405
        - 1.2784829610441e-29 * T_4 * _pq_x50
        - 7.79144796283832e-24 * T_5 * T_6 * T_7
        - 2.55696592208827e-29 * T_5 * T_8 * T_9
        - 1.60759852441169e-23 * T_5 * T_8
        + T_5 * _pq_x404
        + T_5 * _pq_x408
        - 2.18983925362196e-23 * T_5 * _pq_x43
        - 2.18983925362196e-23 * T_5 * _pq_x46
        - 3.57489725197226e-23 * T_6 * T_8 * T_9
        + T_6 * _pq_x406
        - 6.12323399573637e-17 * T_6 * _pq_x43
        - 6.12323399573637e-17 * T_6 * _pq_x46
        - 1.75187136374019e-24 * T_7 * T_9
        - 1.39321954733859e-30 * T_7 * _pq_x11
        - 1.08932331364397e-17 * T_7 * _pq_x12
        - 1.08932331364397e-17 * T_7 * _pq_x13
        - 1.39321954733859e-30 * T_7 * _pq_x14
        - 5.5511151229803e-17 * T_7 * _pq_x39
        - 5.5511151229803e-17 * T_7 * _pq_x48
        - 1.39321954733859e-30 * T_7 * _pq_x50
        - 1.08932331364397e-17 * T_7 * _pq_x52
        - 5.5511151229803e-17 * T_7 * _pq_x56
        + _pq_x10 * _pq_x421
        + _pq_x10 * _pq_x423
        + _pq_x100 * _pq_x422
        + _pq_x104 * _pq_x426
        + _pq_x11 * _pq_x407
        + _pq_x11 * _pq_x412
        + _pq_x11 * _pq_x416
        + _pq_x12 * _pq_x410
        + _pq_x12 * _pq_x415
        + _pq_x120 * _pq_x422
        + _pq_x13 * _pq_x410
        + _pq_x13 * _pq_x415
        + _pq_x14 * _pq_x407
        + _pq_x14 * _pq_x412
        + _pq_x14 * _pq_x416
        + _pq_x17 * _pq_x427
        + _pq_x21 * _pq_x54
        + _pq_x22 * _pq_x54
        + _pq_x29 * _pq_x429
        + 4.84476537428808e-19 * _pq_x292
        + 1.27848296104413e-29 * _pq_x293
        + 2.21598537869374e-38 * _pq_x294
        + _pq_x30 * _pq_x420
        + 1.76586525373172e-24 * _pq_x319
        + _pq_x32 * _pq_x420
        + 1.85890315680879e-31 * _pq_x323
        + 5.19786965446289e-25 * _pq_x325
        + 4.01899640086079e-22 * _pq_x337
        + _pq_x38 * _pq_x428
        + _pq_x38 * _pq_x429
        + _pq_x39 * _pq_x409
        + _pq_x39 * _pq_x413
        + _pq_x409 * _pq_x56
        + 1.52936611960026e-29 * _pq_x411
        + _pq_x413 * _pq_x56
        + 4.93771683753601e-18 * _pq_x414
        + 1.15525555467237e-28 * _pq_x417
        + 3.57489725197244e-23 * _pq_x418
        + 3.57489725197244e-23 * _pq_x419
        + _pq_x421 * _pq_x9
        + _pq_x423 * _pq_x9
        + 1.23926877120586e-31 * _pq_x424
        + 1.23926877120586e-31 * _pq_x425
        + _pq_x426 * _pq_x96
        + 4.01899640086059e-22 * _pq_x430
        + 4.62102211540159e-30 * _pq_x81
    )
    _pq_x432 = 1.27848296104417e-29 * T_0
    _pq_x433 = 1.76586525373172e-24 * T_1
    _pq_x434 = 4.84476537428808e-19 * T_2
    _pq_x435 = T_0 * _pq_x13
    _pq_x436 = 6.19634385602931e-32 * T_2
    _pq_x437 = 4.93771683753601e-18 * T_2
    _pq_x438 = 1.7326232181543e-25 * T_1
    _pq_x439 = 2.21598537869374e-38 * T_1
    _pq_x440 = 1.08932331364397e-17 * T_3
    _pq_x441 = 1.02759980248801e-29 * T_0
    _pq_x442 = 5.5511151229803e-17 * T_3
    _pq_x443 = T_0 * _pq_x11
    _pq_x444 = 1.39321954733859e-30 * T_3
    _pq_x445 = 1.15525555467237e-28 * T_7
    _pq_x446 = 1.22464679914727e-16 * T_3
    _pq_x447 = 4.37967850724393e-23 * T_3
    _pq_x448 = T_0 * _pq_x96
    _pq_x449 = 4.01899640086059e-22 * T_3
    _pq_x450 = 4.01899640086079e-22 * T_7
    _pq_x451 = T_1 * _pq_x9
    _pq_x452 = 1.71935485051491e-28 * T_3
    _pq_x453 = 7.79144796283832e-24 * T_3
    _pq_x454 = (
        -4.01899640086079e-22 * T_0 * _pq_x104
        + T_0 * _pq_x119
        + T_0 * _pq_x288
        + T_0 * _pq_x405
        - 3.57489725197226e-23 * T_1 * _pq_x10
        - T_1 * _pq_x289
        + T_1 * _pq_x404
        + T_1 * _pq_x408
        - 4.89858708709714e-18 * T_10 * T_3
        + T_2 * _pq_x406
        - 1.75187136374019e-24 * T_3 * T_9
        - 3.57489725197226e-23 * T_5 * _pq_x19
        - _pq_x10 * _pq_x452
        + _pq_x102 * _pq_x447
        + _pq_x103 * _pq_x446
        + _pq_x104 * _pq_x447
        - _pq_x11 * _pq_x444
        + _pq_x12 * _pq_x434
        + _pq_x12 * _pq_x438
        - _pq_x12 * _pq_x440
        - _pq_x12 * _pq_x441
        + _pq_x120 * _pq_x446
        - _pq_x13 * _pq_x440
        + _pq_x14 * _pq_x432
        + _pq_x14 * _pq_x436
        + _pq_x14 * _pq_x439
        - _pq_x14 * _pq_x444
        + _pq_x15 * _pq_x428
        - _pq_x15 * _pq_x450
        + _pq_x16 * _pq_x423
        + _pq_x16 * _pq_x445
        + _pq_x17 * _pq_x450
        - _pq_x18 * _pq_x43
        + _pq_x18 * _pq_x46
        - _pq_x18 * _pq_x54
        + _pq_x19 * _pq_x421
        - _pq_x19 * _pq_x445
        - _pq_x20 * _pq_x43
        + _pq_x20 * _pq_x46
        - _pq_x20 * _pq_x54
        + _pq_x24 * _pq_x420
        - _pq_x32 * _pq_x453
        + _pq_x34 * _pq_x449
        - 7.79144796283832e-24 * _pq_x344
        + 4.84476537428808e-19 * _pq_x345
        + 1.27848296104413e-29 * _pq_x346
        + 2.21598537869374e-38 * _pq_x347
        - _pq_x35 * _pq_x452
        - 1.60759852441169e-23 * _pq_x352
        + 1.85890315680879e-31 * _pq_x354
        + 5.19786965446289e-25 * _pq_x355
        - _pq_x36 * _pq_x453
        + 3.46524643630859e-25 * _pq_x360
        - 2.55696592208827e-29 * _pq_x362
        - 3.57489725197226e-23 * _pq_x365
        + 2.55696592208827e-29 * _pq_x366
        + 1.23926877120586e-31 * _pq_x370
        + 1.23926877120586e-31 * _pq_x371
        + 4.01899640086079e-22 * _pq_x375
        - 1.15525555467237e-28 * _pq_x376
        + 1.76586525373172e-24 * _pq_x379
        + _pq_x38 * _pq_x427
        + _pq_x38 * _pq_x449
        - _pq_x39 * _pq_x442
        + 4.93771683753601e-18 * _pq_x391
        + 1.15525555467237e-28 * _pq_x394
        + 3.57489725197244e-23 * _pq_x396
        + 3.57489725197244e-23 * _pq_x397
        + 6.49591408297515e-42 * _pq_x399
        - 3.57489725197226e-23 * _pq_x402
        + _pq_x432 * _pq_x50
        + _pq_x433 * _pq_x48
        + _pq_x433 * _pq_x56
        + _pq_x434 * _pq_x52
        + 1.52936611960026e-29 * _pq_x435
        + _pq_x436 * _pq_x50
        + _pq_x437 * _pq_x48
        + _pq_x437 * _pq_x56
        + _pq_x438 * _pq_x52
        + _pq_x439 * _pq_x50
        - _pq_x440 * _pq_x52
        - _pq_x441 * _pq_x52
        - _pq_x442 * _pq_x48
        - _pq_x442 * _pq_x56
        - 1.2784829610441e-29 * _pq_x443
        - _pq_x444 * _pq_x50
        + 4.01899640086059e-22 * _pq_x448
        - 3.57489725197208e-23 * _pq_x451
        - _pq_x452 * _pq_x9
        - 4.62102211540159e-30 * _pq_x65
        + 4.62102211540159e-30 * _pq_x83
    )
    _pq_x455 = 1.43051152786249e-8 * T_10
    _pq_x456 = 4.89858708709746e-18 * T_11
    _pq_x457 = T_0 * T_9
    _pq_x458 = T_4 * T_9
    _pq_x459 = T_0 * _pq_x104
    _pq_x460 = 1.08932331364404e-17 * T_3
    _pq_x461 = T_4 * _pq_x15
    _pq_x462 = T_4 * _pq_x16
    _pq_x463 = 3.89572398141941e-24 * T_7
    _pq_x464 = 1.08932331364404e-17 * T_7
    _pq_x465 = 1.5068822921685e-54 * T_1
    _pq_x466 = T_1 * _pq_x10
    _pq_x467 = 1.50688229216812e-54 * T_3
    _pq_x468 = 1.22464679914735e-16 * T_3
    _pq_x469 = T_5 * _pq_x19
    _pq_x470 = T_6 * _pq_x19
    _pq_x471 = T_6 * _pq_x24
    _pq_x472 = 3.89572398141941e-24 * T_3
    _pq_x473 = 4.84476537428839e-19 * T_0
    _pq_x474 = 3.24795708865946e-42 * T_0
    _pq_x475 = 6.1963438560297e-32 * T_0
    _pq_x476 = 2.82049647921288e-29 * T_3
    _pq_x477 = 1.34037178141418e-55 * T_1
    _pq_x478 = T_1 * _pq_x48
    _pq_x479 = 1.71430478277419e-68 * T_1
    _pq_x480 = T_1 * _pq_x56
    _pq_x481 = T_3 * _pq_x11
    _pq_x482 = 4.79354331948079e-62 * T_2
    _pq_x483 = 5.38903134565115e-61 * T_3
    _pq_x484 = T_2 * _pq_x48
    _pq_x485 = 6.12304350556196e-75 * T_2
    _pq_x486 = T_2 * _pq_x56
    _pq_x487 = T_3 * _pq_x13
    _pq_x488 = 1.75187136374019e-24 * T_10
    _pq_x489 = T_3 * _pq_x9
    _pq_x490 = 2.8204964792127e-29 * T_7
    _pq_x491 = T_7 * _pq_x16
    _pq_x492 = 6.49591417731851e-42 * T_0
    _pq_x493 = 1.08932331364411e-17 * T_1
    _pq_x494 = 1.00868820449834e-35 * T_3
    _pq_x495 = 1.08932331364383e-17 * T_3
    _pq_x496 = 1.00868820449834e-35 * T_7
    _pq_x497 = 2.8204964792127e-29 * T_3
    _pq_x498 = 8.97228146207388e-37 * T_0
    _pq_x499 = 6.83044048817319e-55 * T_0
    _pq_x500 = T_0 * _pq_x50
    _pq_x501 = 4.84476537428932e-19 * T_1
    _pq_x502 = 6.19634385602931e-32 * T_1
    _pq_x503 = 1.73262321815429e-25 * T_2
    _pq_x504 = 3.89572398141916e-24 * T_3
    _pq_x505 = 3.46524643630881e-25 * _pq_x50
    _pq_x506 = 3.46524643630881e-25 * _pq_x14
    _pq_x507 = 3.89572398141916e-24 * T_3
    _pq_x508 = T_1 * _pq_x16
    _pq_x509 = T_4 * _pq_x9
    _pq_x510 = 9.68953074857677e-19 * T_6
    _pq_x511 = 1.22464679914735e-16 * T_7
    _pq_x512 = 3.42666896342803e-68 * T_6
    _pq_x513 = 1.50688229216812e-54 * T_7
    _pq_x514 = T_4 * _pq_x10
    _pq_x515 = T_4 * _pq_x36
    _pq_x516 = 3.24795708865946e-42 * T_4
    _pq_x517 = T_5 * _pq_x39
    _pq_x518 = T_6 * _pq_x39
    _pq_x519 = 6.12304350556196e-75 * T_6
    _pq_x520 = 5.38903134565115e-61 * T_7
    _pq_x521 = 4.79354331948079e-62 * T_6
    _pq_x522 = T_4 * _pq_x50
    _pq_x523 = T_5 * _pq_x56
    _pq_x524 = T_6 * _pq_x56
    _pq_x525 = 9.68953074857615e-19 * T_6
    _pq_x526 = T_3 * _pq_x19
    _pq_x527 = 6.83044048817319e-55 * T_4
    _pq_x528 = T_4 * _pq_x11
    _pq_x529 = 6.19634385602931e-32 * T_5
    _pq_x530 = T_6 * _pq_x11
    _pq_x531 = 3.89572398141916e-24 * T_7
    _pq_x532 = 1.07271182906117e-40 * T_11
    _pq_x533 = 2.99951949827904e-34 * T_11
    _pq_x534 = 2.46092553904945e-38 * T_7
    _pq_x535 = 9.72346137165803e-63 * _pq_x104
    _pq_x536 = 9.72346137165803e-63 * _pq_x71
    _pq_x537 = 8.55284707229503e-50 * T_2
    _pq_x538 = 7.49879891330881e-33 * T_3
    _pq_x539 = 1.05280120714078e-44 * T_3
    _pq_x540 = 4.77088590420656e-40 * T_3
    _pq_x541 = 2.68177963259537e-39 * T_3
    _pq_x542 = 2.46092553904933e-38 * T_3
    _pq_x543 = 6.29223404460698e-46 * T_0
    _pq_x544 = 7.82845033003564e-46 * T_0
    _pq_x545 = 7.82845033003584e-46 * _pq_x14
    _pq_x546 = 3.39907768352814e-33 * T_3
    _pq_x547 = 1.06092573912052e-41 * _pq_x12
    _pq_x548 = 1.35689970048731e-54 * T_1
    _pq_x549 = 2.96656320412091e-35 * T_2
    _pq_x550 = 6.67018154645341e-34 * T_3
    _pq_x551 = 3.79416633485133e-48 * _pq_x14
    _pq_x552 = 8.53100929578867e-47 * T_3
    _pq_x553 = 2.46092553904933e-38 * T_7
    _pq_x554 = 1.56569006600713e-45 * T_6
    _pq_x555 = 2.68177963259537e-39 * T_7
    _pq_x556 = 7.49879891330881e-33 * T_7
    _pq_x557 = 7.82845033003564e-46 * T_4
    _pq_x558 = 1.35689970048731e-54 * T_5
    _pq_x559 = 2.96656320412091e-35 * T_6
    _pq_x560 = 8.75811540203011e-47 * T_4
    _pq_x561 = 5.65917689691665e-9 * T_10
    _pq_x562 = 6.36220007945436e-8 * T_11
    _pq_x563 = 2.02387949255045e-15 * T_9
    _pq_x564 = 0.0158242045875154 * T_10
    _pq_x565 = 0.177899997681368 * T_10
    _pq_x566 = 0.177899997681368 * T_11
    _pq_x567 = 2.0 * T_11
    _pq_x568 = 5.65917689691665e-9 * T_9
    _pq_x569 = 6.36220007945436e-8 * T_9
    _pq_x570 = 2.46092553904955e-38 * T_10
    _pq_x571 = 2.46092553904892e-38 * T_11
    _pq_x572 = 2.46092553904955e-38 * T_9
    _pq_x573 = 4.09723291206597e-14 * T_10
    _pq_x574 = 4.60622030968703e-13 * T_11
    _pq_x575 = 8.80095607877014e-45 * T_11
    _pq_x576 = 2.79967217318129e-52 * T_9
    _pq_x577 = 7.82845033003514e-46 * T_10
    _pq_x578 = T_10 * _pq_x50
    _pq_x579 = T_9 * _pq_x48
    _pq_x580 = T_9 * _pq_x52
    _pq_x581 = 4.09723291206623e-14 * T_10
    _pq_x582 = 4.60622030968674e-13 * T_11
    _pq_x583 = 2.23099117000197e-38 * T_9
    _pq_x584 = 3.00926553810506e-36 * T_10
    _pq_x585 = 1.64731285004072e-19 * T_10
    _pq_x586 = 1.64731285004072e-19 * T_11
    _pq_x587 = 4.09723291206623e-14 * T_9
    _pq_x588 = 4.60622030968674e-13 * T_9
    _pq_x589 = 0.177899997681379 * T_10
    _pq_x590 = 0.177899997681334 * T_11
    _pq_x591 = 1.99999999999987 * T_11
    _pq_x592 = 4.09723291206518e-14 * T_8
    _pq_x593 = 4.60622030968674e-13 * T_8
    _pq_x594 = 7.1525577991846e-7 * T_11
    _pq_x595 = 1.64731285004072e-19 * T_8
    _pq_x596 = 0.177899997681379 * T_9
    _pq_x597 = 1.89697144901585e-32 * T_10
    _pq_x598 = 1.11549558500113e-38 * T_8
    _pq_x599 = 6.36220007945395e-8 * T_11
    _pq_x600 = 1.46528476101366e-20 * T_8
    _pq_x601 = 1.01193974627516e-15 * T_9
    _pq_x602 = 2.82958844845814e-9 * T_10
    _pq_x603 = T_10 * _pq_x56
    _pq_x604 = 6.36220007945395e-8 * T_11
    _pq_x605 = 1.46528476101366e-20 * T_8
    _pq_x606 = 2.44929354354873e-18 * T_8
    _pq_x607 = 2.44929354354857e-18 * T_9
    _pq_x608 = 1.49975974913952e-34 * T_10
    _pq_x609 = 5.36355914530584e-41 * T_9
    _pq_x610 = T_10 * _pq_x29
    _pq_x611 = 1.23926877120594e-31 * T_9
    _pq_x612 = 9.68953074857677e-19 * T_10
    _pq_x613 = 1.22464679914735e-16 * T_11
    _pq_x614 = 1.50688229216812e-54 * T_11
    _pq_x615 = 3.89572398141941e-24 * T_8
    _pq_x616 = T_10 * _pq_x34
    _pq_x617 = 1.08932331364404e-17 * T_8
    _pq_x618 = 1.5068822921685e-54 * T_9
    _pq_x619 = T_10 * _pq_x39
    _pq_x620 = 3.24795708865946e-42 * T_8
    _pq_x621 = 5.38903134565115e-61 * T_11
    _pq_x622 = 4.79354331948079e-62 * T_10
    _pq_x623 = T_10 * _pq_x48
    _pq_x624 = 1.00868820449834e-35 * T_11
    _pq_x625 = 2.8204964792127e-29 * T_9
    _pq_x626 = 1.08932331364411e-17 * T_9
    _pq_x627 = 6.83044048817319e-55 * T_8
    _pq_x628 = 6.19634385602931e-32 * T_9
    _pq_x629 = 3.89572398141916e-24 * T_11

    p_sin = np.array(
        [
            [
                0.999999999973791,
                0,
                0,
                -0.999999999973791,
                0,
                0,
                0,
                0,
                4.43320933614055e-22,
            ],
            [0, 1.00000000000000, 1.00000000000000, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0.0677499994617826,
                0.0677499994617826,
                0,
                0.251318528021616,
                0,
                0,
                0,
                3.61999031060176e-9,
            ],
            [
                -0.0677499994635582,
                0,
                0,
                0.0677499994635582,
                0,
                0.251318528028202,
                0,
                0,
                0.263999998569489,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                -1.91135478281043e-6,
                -0.000500000023748726,
                -0.000500000023748726,
                1.91135478281043e-6,
                0,
                1.53888215454048e-17,
                0,
                1.81954118620233e-6,
                4.90508659908109e-7,
            ],
            [
                0.000500000023761830,
                -4.14849099926407e-18,
                -4.14849099926407e-18,
                -0.000500000023761830,
                -1.53888215458081e-17,
                0,
                -2.11758236813575e-22,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0.0111253091414719,
                -1.91135487364500e-9,
                -1.91135487364500e-9,
                -0.137447314198686,
                -1.53888222767379e-20,
                -0.0340536602772934,
                -1.81954127267358e-9,
                -0.132696182076385,
                -0.0357719995219869,
            ],
            [
                4.14831077195343e-21,
                -0.00194468428709169,
                -0.128266689347616,
                -4.14831077195343e-21,
                0.0340536602781859,
                -1.53888222771412e-20,
                -0.132696182079863,
                0,
                -1.61653384293875e-20,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    p_cos = np.array(
        [
            [0, 1.00000000000000, 1.00000000000000, 0, 0, 0, 0, 0, 0],
            [
                -0.999999999973791,
                0,
                0,
                0.999999999973791,
                0,
                0,
                0,
                0,
                -4.43320933614055e-22,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                -0.0677499994635582,
                0,
                0,
                0.0677499994635582,
                0,
                0.251318528028202,
                0,
                0,
                0.263999998569489,
            ],
            [
                0,
                -0.0677499994617826,
                -0.0677499994617826,
                0,
                -0.251318528021616,
                0,
                0,
                0,
                -3.61999031060176e-9,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0.000500000023761830,
                -4.14849099926407e-18,
                -4.14849099926407e-18,
                -0.000500000023761830,
                -1.53888215458081e-17,
                0,
                -2.11758236813575e-22,
                0,
                0,
            ],
            [
                1.91135478281043e-6,
                0.000500000023748726,
                0.000500000023748726,
                -1.91135478281043e-6,
                0,
                -1.53888215454048e-17,
                0,
                -1.81954118620233e-6,
                -4.90508659908109e-7,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                4.14831077195343e-21,
                -0.00194468428709169,
                -0.128266689347616,
                -4.14831077195343e-21,
                0.0340536602781859,
                -1.53888222771412e-20,
                -0.132696182079863,
                0,
                -1.61653384293875e-20,
            ],
            [
                -0.0111253091414719,
                1.91135487364500e-9,
                1.91135487364500e-9,
                0.137447314198686,
                1.53888222767379e-20,
                0.0340536602772934,
                1.81954127267358e-9,
                0.132696182076385,
                0.0357719995219869,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    p_one = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, _pq_x0 + _pq_x1],
            [
                7.23998027713317e-6,
                0,
                0,
                -7.23998027713317e-6,
                0,
                0,
                0,
                0,
                -_pq_x2 - _pq_x3 - 6.12323399557628e-17,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.0199999995529652],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1.0 * T_11
                - 2.30311015484352e-13 * T_8
                - _pq_x4
                - _pq_x5
                - 0.0399999991059303,
            ],
            [
                0,
                4.90508659891944e-7,
                4.90508659891944e-7,
                0,
                1.81954118620233e-6,
                0,
                0,
                0,
                1.41024823960644e-29 * T_8
                + _pq_x6
                - _pq_x7
                + _pq_x8
                - 0.000500000023748723,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1.55575381946529e-61 * _pq_x10
                + 4.86173068582902e-63 * _pq_x11
                - 4.27642353614751e-50 * _pq_x12
                - 4.27642353614751e-50 * _pq_x13
                + 4.86173068582902e-63 * _pq_x14
                - 1.55575381946529e-61 * _pq_x9,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                7.49879874569759e-35 * T_10
                - T_3 * _pq_x21
                - T_3 * _pq_x22
                + T_7 * _pq_x18
                + T_7 * _pq_x20
                + 2.68177957265292e-41 * T_9
                + 2.0094982004304e-22 * _pq_x15
                - 5.77627777336186e-29 * _pq_x16
                - 2.0094982004304e-22 * _pq_x17
                + 5.77627777336186e-29 * _pq_x19,
            ],
            [
                0.263999998562569,
                -3.61999031060176e-9,
                -3.61999031060176e-9,
                -0.263999998562569,
                0,
                1.11414764479973e-22,
                0,
                -0.251318528021616,
                3.5369500430668e-45 * T_0 * T_6
                + 1.23046276952473e-38 * T_1 * T_4
                + 4.86173068582902e-63 * T_1 * T_5
                + 1.22464677177429e-18 * T_10
                - T_2 * _pq_x25
                + 1.34088981629768e-39 * T_3 * T_5
                + 3.7493994566544e-33 * T_3 * T_6
                - T_7 * _pq_x23
                - T_7 * _pq_x26
                + 4.37967840935049e-25 * T_9
                - 1.23046276952473e-38 * _pq_x15
                - 3.5369500430668e-45 * _pq_x19
                - 2.03915764624954e-56 * _pq_x24
                - 0.0677499994617826,
            ],
            [
                -0.0357719995229244,
                0,
                0,
                0.0357719995229244,
                0,
                0.132696182079863,
                0,
                0.0340536602781859,
                4.60622030968703e-13 * T_0 * T_3
                + 6.36220007945436e-8 * T_1 * T_3
                + 0.177899997681368 * T_10 * T_11
                + 4.60622030968703e-13 * T_11 * T_8
                + 6.36220007945436e-8 * T_11 * T_9
                + 0.0799999982118607 * T_11
                + 0.177899997681368 * T_2 * T_3
                + 4.60622030968703e-13 * T_4 * T_7
                + 6.36220007945436e-8 * T_5 * T_7
                + 0.177899997681368 * T_6 * T_7
                - 1.84248808269199e-14 * T_8
                - 4.09723291206597e-14 * _pq_x10
                - _pq_x27
                - _pq_x28
                - 1.46528476101376e-20 * _pq_x29
                - _pq_x31
                - _pq_x33
                - 1.46528476101376e-20 * _pq_x34
                - 4.09723291206597e-14 * _pq_x35
                - _pq_x37
                - 1.46528476101376e-20 * _pq_x38
                - _pq_x40
                - _pq_x41
                - _pq_x42
                - _pq_x44
                - _pq_x45
                - _pq_x47
                - _pq_x49
                - _pq_x51
                - _pq_x53
                - _pq_x55
                - _pq_x57
                - _pq_x58
                - 4.09723291206597e-14 * _pq_x9
                + 0.135447314291695,
            ],
            [
                0,
                0.263999998569489,
                0.263999998569489,
                0,
                -2.35098870164458e-38,
                0,
                0.251318528028202,
                0,
                -T_11 * _pq_x0
                - T_11 * _pq_x1
                - T_3 * _pq_x18
                - T_3 * _pq_x20
                - T_7 * _pq_x21
                - T_7 * _pq_x22
                + 8.59677425257455e-29 * _pq_x10
                + 6.96609773669297e-31 * _pq_x11
                + 5.44661656821987e-18 * _pq_x12
                + 5.44661656821987e-18 * _pq_x13
                + 6.96609773669297e-31 * _pq_x14
                - 2.0094982004303e-22 * _pq_x29
                + 3.89572398141916e-24 * _pq_x30
                + 3.89572398141916e-24 * _pq_x32
                - 2.0094982004303e-22 * _pq_x34
                + 8.59677425257455e-29 * _pq_x35
                + 3.89572398141916e-24 * _pq_x36
                - 2.0094982004303e-22 * _pq_x38
                + 2.77555756149015e-17 * _pq_x39
                + 2.77555756149015e-17 * _pq_x48
                + 6.96609773669297e-31 * _pq_x50
                + 5.44661656821987e-18 * _pq_x52
                + 2.77555756149015e-17 * _pq_x56
                + _pq_x59
                + _pq_x60
                + 8.59677425257455e-29 * _pq_x9
                + 3.06161714336763e-20,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2.22647111576681e-69 * T_10 * _pq_x15
                + 9.79717395521038e-20 * T_10
                - T_11 * _pq_x59
                - T_11 * _pq_x60
                - 6.22566410024812e-63 * T_2 * _pq_x81
                - T_3 * _pq_x61
                - T_3 * _pq_x62
                - 2.11328522812393e-73 * T_4 * _pq_x65
                - T_4 * _pq_x69
                - T_7 * _pq_x63
                - T_7 * _pq_x64
                - 3.65807817365964e-75 * T_9 * _pq_x15
                - 1.00552487181309e-63 * T_9 * _pq_x24
                + 3.50374264916564e-26 * T_9
                + 3.43870962416867e-30 * _pq_x10
                + 2.78643903239542e-32 * _pq_x11
                + 2.1786465785914e-19 * _pq_x12
                + 2.1786465785914e-19 * _pq_x13
                + 2.78643903239542e-32 * _pq_x14
                - 3.83577297902333e-69 * _pq_x15
                - 1.07255979936586e-62 * _pq_x16
                + _pq_x17 * _pq_x75
                - 8.03799262205804e-24 * _pq_x29
                + 1.55828955773718e-25 * _pq_x30
                + 1.55828955773718e-25 * _pq_x32
                - 8.03799262205804e-24 * _pq_x34
                + 3.43870962416867e-30 * _pq_x35
                + 1.55828955773718e-25 * _pq_x36
                - 8.03799262205804e-24 * _pq_x38
                + 1.11022299978064e-18 * _pq_x39
                + 1.11022299978064e-18 * _pq_x48
                + 2.78643903239542e-32 * _pq_x50
                + 2.1786465785914e-19 * _pq_x52
                + 1.11022299978064e-18 * _pq_x56
                - _pq_x65 * _pq_x67
                + 6.22566410024812e-63 * _pq_x66
                + _pq_x67 * _pq_x83
                - 1.02287273346506e-68 * _pq_x70
                + 1.00552487181309e-63 * _pq_x72
                + _pq_x73 * _pq_x74
                - _pq_x74 * _pq_x82
                + 3.65807817365964e-75 * _pq_x76
                + 1.02287273346506e-68 * _pq_x78
                - 7.16614484009417e-74 * _pq_x79
                - 2.22647111576681e-69 * _pq_x80
                + 3.43870962416867e-30 * _pq_x9,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3.57489725197226e-23 * T_0 * T_1 * T_10
                + 2.55696592208827e-29 * T_0 * T_1 * T_9
                + 1.60759852441161e-23 * T_0 * T_1
                + 1.15525555467237e-28 * T_0 * T_10 * T_3
                + 1.71935485051491e-28 * T_0 * T_11 * T_2
                + 3.57489725197226e-23 * T_0 * T_2 * T_9
                + 1.41477998560392e-46 * T_0 * T_6
                + 7.79144796283832e-24 * T_1 * T_11 * T_2
                + 4.01899640086079e-22 * T_1 * T_3 * T_8
                + 1.75187136374019e-24 * T_1 * T_3
                + 4.92185096808696e-40 * T_1 * T_4
                + 1.71935485051491e-28 * T_10 * T_11 * T_8
                + 7.79144796283832e-24 * T_10 * T_11 * T_9
                + 4.89858708709714e-18 * T_10 * T_11
                + 3.57489725197226e-23 * T_10 * T_4 * T_5
                + 1.15525555467237e-28 * T_10 * T_4 * T_7
                + 3.57489725197208e-23 * T_10 * T_8 * T_9
                + 6.12323399573637e-17 * T_10 * _pq_x46
                + 6.12323399573637e-17 * T_10 * _pq_x54
                - 7.34788046640778e-20 * T_10
                + 1.71935485051491e-28 * T_11 * T_4 * T_6
                + 7.79144796283832e-24 * T_11 * T_5 * T_6
                + 1.75187136374019e-24 * T_11 * T_9
                + 1.39321954733859e-30 * T_11 * _pq_x11
                + 1.08932331364397e-17 * T_11 * _pq_x12
                + 1.08932331364397e-17 * T_11 * _pq_x13
                + 1.39321954733859e-30 * T_11 * _pq_x14
                + 5.5511151229803e-17 * T_11 * _pq_x39
                + 5.5511151229803e-17 * T_11 * _pq_x48
                + 1.39321954733859e-30 * T_11 * _pq_x50
                + 1.08932331364397e-17 * T_11 * _pq_x52
                + 5.5511151229803e-17 * T_11 * _pq_x56
                + 4.89858708709714e-18 * T_2 * T_3
                + 5.36355914530584e-41 * T_3 * T_5
                + 1.49975974913952e-34 * T_3 * T_6
                + 2.55696592208827e-29 * T_4 * T_5 * T_9
                + 1.60759852441161e-23 * T_4 * T_5
                + 3.57489725197226e-23 * T_4 * T_6 * T_9
                + 4.01899640086079e-22 * T_5 * T_7 * T_8
                + 1.75187136374019e-24 * T_5 * T_7
                + 4.89858708709714e-18 * T_6 * T_7
                - T_7 * _pq_x84
                - T_7 * _pq_x85
                + 1.60759852441161e-23 * T_8 * T_9
                - T_8 * _pq_x109
                - T_8 * _pq_x119
                + 1.02759980248801e-29 * T_8 * _pq_x13
                + 1.2784829610441e-29 * T_8 * _pq_x14
                + 1.02759980248801e-29 * T_8 * _pq_x52
                + 2.18983925362196e-23 * T_9 * _pq_x46
                + 2.18983925362196e-23 * T_9 * _pq_x54
                - 2.62780698687423e-26 * T_9
                - _pq_x0 * _pq_x43
                - _pq_x1 * _pq_x43
                - 6.87741924833734e-30 * _pq_x10
                - _pq_x100 * _pq_x101
                - _pq_x100 * _pq_x106
                - _pq_x101 * _pq_x103
                - _pq_x102 * _pq_x97
                - _pq_x103 * _pq_x106
                - 4.01899640086059e-22 * _pq_x105
                - _pq_x107 * _pq_x94
                - _pq_x108 * _pq_x39
                - _pq_x108 * _pq_x48
                - _pq_x108 * _pq_x56
                - _pq_x11 * _pq_x111
                - _pq_x11 * _pq_x112
                - _pq_x11 * _pq_x113
                - 5.57287806479084e-32 * _pq_x11
                - _pq_x110 * _pq_x39
                - _pq_x110 * _pq_x48
                - _pq_x110 * _pq_x56
                - _pq_x111 * _pq_x50
                - _pq_x112 * _pq_x50
                - _pq_x113 * _pq_x50
                - _pq_x114 * _pq_x13
                - _pq_x114 * _pq_x52
                - 1.85890315680879e-31 * _pq_x115
                - 1.52936611960026e-29 * _pq_x116
                - 5.19786965446289e-25 * _pq_x117
                - _pq_x118 * _pq_x13
                - _pq_x118 * _pq_x52
                - 4.3572931571828e-19 * _pq_x12
                - 4.3572931571828e-19 * _pq_x13
                - 5.57287806479084e-32 * _pq_x14
                - 4.92185096808696e-40 * _pq_x15
                - 1.41477998560392e-46 * _pq_x19
                - 6.37236764452981e-58 * _pq_x24
                - _pq_x29 * _pq_x89
                - _pq_x29 * _pq_x90
                - _pq_x30 * _pq_x95
                - _pq_x30 * _pq_x98
                - _pq_x30 * _pq_x99
                - 3.11657911547435e-25 * _pq_x30
                - 3.11657911547435e-25 * _pq_x32
                - _pq_x34 * _pq_x89
                - _pq_x34 * _pq_x90
                - _pq_x35 * _pq_x91
                - _pq_x35 * _pq_x92
                - 6.87741924833734e-30 * _pq_x35
                - _pq_x36 * _pq_x95
                - _pq_x36 * _pq_x98
                - _pq_x36 * _pq_x99
                - 3.11657911547435e-25 * _pq_x36
                - 2.22044599956128e-18 * _pq_x39
                - 2.22044599956128e-18 * _pq_x48
                - 5.57287806479084e-32 * _pq_x50
                - 4.3572931571828e-19 * _pq_x52
                - 2.22044599956128e-18 * _pq_x56
                - 4.84476537428808e-19 * _pq_x86
                - 1.27848296104413e-29 * _pq_x87
                - 2.21598537869374e-38 * _pq_x88
                - _pq_x9 * _pq_x91
                - _pq_x9 * _pq_x92
                - 6.87741924833734e-30 * _pq_x9
                - _pq_x93 * _pq_x94
                - _pq_x96 * _pq_x97,
            ],
            [
                8.05470187655673e-8,
                0.000264000011108816,
                0.000264000011108816,
                -9.95115843971683e-7,
                -1.11414769774810e-25,
                -2.46547828786384e-7,
                0.000251318539965192,
                -9.60717741111164e-7,
                T_10 * _pq_x157
                + 4.49927914685183e-36 * T_10
                + T_3 * _pq_x63
                + T_3 * _pq_x64
                - T_7 * _pq_x61
                - T_7 * _pq_x62
                + T_9 * _pq_x163
                + T_9 * _pq_x164
                + 1.60906770762629e-42 * T_9
                - _pq_x10 * _pq_x127
                + 4.21120473443536e-46 * _pq_x10
                - _pq_x100 * _pq_x140
                + _pq_x100 * _pq_x143
                + _pq_x100 * _pq_x144
                - 2.99951949827904e-34 * _pq_x100
                - _pq_x102 * _pq_x132
                + _pq_x102 * _pq_x134
                - _pq_x102 * _pq_x137
                + _pq_x102 * _pq_x138
                - 1.07271182906117e-40 * _pq_x102
                - _pq_x103 * _pq_x140
                + _pq_x103 * _pq_x143
                + _pq_x103 * _pq_x144
                - 2.99951949827904e-34 * _pq_x103
                - 1.07271182906117e-40 * _pq_x104
                + 2.46092553904933e-38 * _pq_x105
                - _pq_x107 * _pq_x126
                + _pq_x107 * _pq_x130
                - _pq_x11 * _pq_x153
                + _pq_x11 * _pq_x154
                + _pq_x11 * _pq_x155
                + 3.4124036420423e-48 * _pq_x11
                + 1.1382499004554e-47 * _pq_x115
                + 9.36466661546431e-46 * _pq_x116
                + 3.18277721736157e-41 * _pq_x117
                + 2.66807255894529e-35 * _pq_x12
                - 2.99951949827904e-34 * _pq_x120
                - _pq_x121 * _pq_x29
                - _pq_x121 * _pq_x34
                + _pq_x122 * _pq_x29
                + _pq_x122 * _pq_x34
                - 4.07831527827407e-56 * _pq_x123
                - _pq_x124 * _pq_x29
                - _pq_x124 * _pq_x34
                + _pq_x125 * _pq_x35
                + _pq_x125 * _pq_x9
                - _pq_x126 * _pq_x93
                - _pq_x127 * _pq_x35
                - _pq_x127 * _pq_x9
                + 3.97760019462587e-58 * _pq_x128
                - _pq_x129 * _pq_x35
                - _pq_x129 * _pq_x9
                + _pq_x13 * _pq_x156
                - _pq_x13 * _pq_x159
                - _pq_x13 * _pq_x161
                + 2.66807255894529e-35 * _pq_x13
                + _pq_x130 * _pq_x93
                + 2.12185147824104e-41 * _pq_x131
                - _pq_x132 * _pq_x96
                - _pq_x133 * _pq_x30
                - _pq_x133 * _pq_x36
                + _pq_x134 * _pq_x96
                + _pq_x135 * _pq_x30
                + _pq_x135 * _pq_x36
                + 7.58833266970266e-48 * _pq_x136
                - _pq_x137 * _pq_x96
                + _pq_x138 * _pq_x96
                - 4.77088590420656e-40 * _pq_x139
                + 3.4124036420423e-48 * _pq_x14
                + 2.12185147824104e-41 * _pq_x141
                - 2.18899323845414e-39 * _pq_x142
                - 2.03915763202453e-56 * _pq_x145
                + 3.97760019462587e-58 * _pq_x146
                + 7.58833266970266e-48 * _pq_x147
                + _pq_x148 * _pq_x39
                + _pq_x148 * _pq_x48
                + _pq_x148 * _pq_x56
                - _pq_x149 * _pq_x39
                - _pq_x149 * _pq_x48
                - _pq_x149 * _pq_x56
                - 8.03799262205844e-24 * _pq_x15
                + _pq_x150 * _pq_x39
                + _pq_x150 * _pq_x48
                + 1.08128061535404e-40 * _pq_x151
                + 3.79416633485133e-48 * _pq_x152
                - _pq_x153 * _pq_x50
                + _pq_x154 * _pq_x50
                + _pq_x155 * _pq_x50
                + _pq_x156 * _pq_x52
                - 6.67018154645341e-34 * _pq_x158
                - _pq_x159 * _pq_x52
                + 2.31051105770079e-30 * _pq_x16
                - 8.53100929578866e-47 * _pq_x160
                - _pq_x161 * _pq_x52
                + 1.06092573912052e-41 * _pq_x162
                - 7.82845033003545e-46 * _pq_x165
                + 1.08128061535404e-40 * _pq_x166
                + 8.03799262205844e-24 * _pq_x17
                - 2.31051105770079e-30 * _pq_x19
                + _pq_x2 * _pq_x43
                - _pq_x2 * _pq_x46
                - _pq_x2 * _pq_x54
                - 9.84370193617342e-40 * _pq_x29
                + _pq_x3 * _pq_x43
                - _pq_x3 * _pq_x46
                - _pq_x3 * _pq_x54
                + 1.90835431902758e-41 * _pq_x30
                + 1.90835431902758e-41 * _pq_x32
                - 9.84370193617342e-40 * _pq_x34
                + 4.21120473443536e-46 * _pq_x35
                + 1.90835431902758e-41 * _pq_x36
                - 9.84370193617342e-40 * _pq_x38
                + 1.35963104302113e-34 * _pq_x39
                + 1.35963104302113e-34 * _pq_x48
                + 3.4124036420423e-48 * _pq_x50
                + 2.66807255894529e-35 * _pq_x52
                + 1.35963104302113e-34 * _pq_x56
                - 5.47382212626882e-48 * _pq_x71
                + 2.96656320412091e-35 * _pq_x86
                + 7.82845033003564e-46 * _pq_x87
                + 1.35689970048731e-54 * _pq_x88
                + 4.21120473443536e-46 * _pq_x9
                - 1.07271182906117e-40 * _pq_x96
                - 2.58988571028006e-7,
            ],
        ],
        dtype=np.float64,
    )

    q = np.array(
        [
            [_pq_x167, _pq_x170, _pq_x171, _pq_x174, _pq_x175, _pq_x176, 0, 0],
            [
                -_pq_x177,
                2.18983925362196e-23 * T_2 - _pq_x178,
                _pq_x179,
                _pq_x180 - _pq_x181,
                -_pq_x23 - _pq_x26,
                _pq_x182 + _pq_x183,
                -_pq_x184,
                _pq_x185 - _pq_x186,
            ],
            [
                -_pq_x171,
                -_pq_x174,
                _pq_x167,
                _pq_x170,
                -_pq_x176,
                _pq_x175,
                _pq_x187,
                6.12323399573637e-17 * T_9 - _pq_x188,
            ],
            [_pq_x189, _pq_x190, -_pq_x191, -_pq_x192, _pq_x195, -_pq_x198, 0, 0],
            [
                1.69953884176407e-33 * T_0
                - 1.23046276952469e-38 * T_1
                + 4.40047803938535e-45 * T_2,
                1.23046276952477e-38 * T_0
                + 1.69953884176396e-33 * T_1
                - 6.07802489883838e-40 * T_2,
                -1.69953884176407e-33 * T_4
                + 1.23046276952469e-38 * T_5
                - 4.40047803938535e-45 * T_6,
                -1.23046276952477e-38 * T_4
                - 1.69953884176396e-33 * T_5
                + 6.07802489883838e-40 * T_6,
                1.41024823960644e-29 * T_0
                - 6.12323399573677e-17 * T_3
                + _pq_x199
                + _pq_x200,
                -1.41024823960644e-29 * T_4
                + 6.12323399573677e-17 * T_7
                - _pq_x201
                - _pq_x202,
                7.1865260129682e-29 * T_10
                + 2.77555756149015e-17 * T_8
                - 2.00949820043035e-22 * T_9,
                -9.92616794176107e-24 * T_10
                + 2.00949820043047e-22 * T_8
                + 2.77555756148997e-17 * T_9,
            ],
            [
                _pq_x191,
                _pq_x192,
                _pq_x189,
                _pq_x190,
                _pq_x198,
                _pq_x195,
                -4.40047803938535e-45 * T_10
                - 1.69953884176407e-33 * T_8
                + 1.23046276952469e-38 * T_9,
                6.07802489883838e-40 * T_10
                - 1.23046276952477e-38 * T_8
                - 1.69953884176396e-33 * T_9,
            ],
            [
                _pq_x205,
                _pq_x212,
                -_pq_x213
                - _pq_x214
                - _pq_x216
                + _pq_x217
                - _pq_x218
                + _pq_x219
                + _pq_x220
                + _pq_x222
                - _pq_x223,
                -_pq_x224
                + _pq_x225
                - _pq_x226
                - _pq_x227
                + _pq_x228
                + _pq_x229
                - _pq_x230
                + _pq_x231
                - _pq_x232
                + _pq_x233
                + _pq_x234
                - _pq_x235,
                _pq_x240,
                -_pq_x241
                - _pq_x242
                + _pq_x243
                + _pq_x244
                + _pq_x245
                - _pq_x246
                + _pq_x247
                - _pq_x248
                - _pq_x63
                - _pq_x64,
                3.45446742203778e-77 * _pq_x11 + 3.45446742203778e-77 * _pq_x14,
                -6.68191177523049e-52 * _pq_x10 - 6.68191177523049e-52 * _pq_x9,
            ],
            [
                T_4 * _pq_x6
                - T_4 * _pq_x7
                + T_4 * _pq_x8
                + T_5 * _pq_x251
                + T_7 * _pq_x187
                - T_8 * _pq_x201
                - T_8 * _pq_x202
                + 7.5344114608425e-55 * _pq_x215
                - 7.5344114608425e-55 * _pq_x221
                + _pq_x249
                + _pq_x250,
                -T_11 * _pq_x180
                + T_11 * _pq_x181
                + T_4 * _pq_x256
                - T_4 * _pq_x257
                + T_5 * _pq_x254
                - T_6 * _pq_x258
                - T_7 * _pq_x188
                + T_7 * _pq_x255
                + 5.44661656822057e-18 * _pq_x215
                - 5.44661656822057e-18 * _pq_x221
                + _pq_x252
                - _pq_x253
                + _pq_x259,
                T_0 * _pq_x6
                - T_0 * _pq_x7
                + T_0 * _pq_x8
                + T_1 * _pq_x251
                + T_3 * _pq_x187
                - T_8 * _pq_x199
                - T_8 * _pq_x200
                + _pq_x260
                + _pq_x261
                + 7.5344114608425e-55 * _pq_x73
                - 7.5344114608425e-55 * _pq_x82,
                T_0 * _pq_x256
                - T_0 * _pq_x257
                + T_1 * _pq_x254
                - T_11 * _pq_x178
                + T_11 * _pq_x264
                - T_2 * _pq_x258
                - T_3 * _pq_x188
                + T_3 * _pq_x255
                + _pq_x262
                - _pq_x263
                + _pq_x265
                + 5.44661656822057e-18 * _pq_x73
                - 5.44661656822057e-18 * _pq_x82,
                T_10 * _pq_x25
                - T_11 * _pq_x182
                - T_11 * _pq_x183
                + T_4 * _pq_x268
                - T_4 * _pq_x272
                + T_5 * _pq_x269
                - T_5 * _pq_x270
                - T_6 * _pq_x271
                + T_7 * _pq_x2
                + T_7 * _pq_x3
                + 2.03915764624954e-56 * _pq_x215
                + _pq_x266
                + _pq_x267
                + _pq_x273,
                T_0 * _pq_x268
                - T_0 * _pq_x272
                + T_1 * _pq_x269
                - T_1 * _pq_x270
                - T_11 * _pq_x23
                - T_11 * _pq_x26
                - T_2 * _pq_x271
                + T_3 * _pq_x2
                + T_3 * _pq_x3
                - _pq_x274
                - _pq_x275
                + 4.27642353614751e-50 * _pq_x276
                + 2.03915764624954e-56 * _pq_x73
                + _pq_x84
                + _pq_x85,
                3.18110003972718e-8 * T_0 * T_5
                + 0.088949998840684 * T_0 * T_6
                + 1.23046276952477e-38 * T_1 * T_6
                + 1.0 * T_3 * T_4
                - T_4 * _pq_x196
                - T_4 * _pq_x197
                - T_7 * _pq_x171
                - 1.22464677177436e-18 * T_8
                - 1.23046276952477e-38 * _pq_x24,
                8.23656425020362e-20 * T_0 * T_6
                + 2.30311015484337e-13 * T_1 * T_4
                + 0.0889499988406897 * T_1 * T_6
                + 4.37967840935049e-25 * T_10
                + 3.5762788995923e-7 * T_2 * T_7
                + 0.999999999999936 * T_3 * T_5
                - T_3 * _pq_x169
                - T_7 * _pq_x172
                - 1.22464677177429e-18 * T_9
                - 2.30311015484337e-13 * _pq_x15
                - 8.23656425020362e-20 * _pq_x19
                - 0.0889499988406897 * _pq_x24,
            ],
            [
                -1.22464677177436e-18 * T_0
                + _pq_x213
                + _pq_x214
                + _pq_x216
                - _pq_x217
                + _pq_x218
                - _pq_x219
                - _pq_x220
                - _pq_x222
                + _pq_x223,
                -1.22464677177429e-18 * T_1
                + 4.37967840935049e-25 * T_2
                + _pq_x224
                - _pq_x225
                + _pq_x226
                + _pq_x227
                - _pq_x228
                - _pq_x229
                + _pq_x230
                - _pq_x231
                + _pq_x232
                - _pq_x233
                - _pq_x234
                + _pq_x235,
                1.22464677177436e-18 * T_4 + _pq_x205,
                1.22464677177429e-18 * T_5 - 4.37967840935049e-25 * T_6 + _pq_x212,
                -2.68177957265292e-41 * T_1
                - 7.49879874569759e-35 * T_2
                + _pq_x241
                + _pq_x242
                - _pq_x243
                - _pq_x244
                - _pq_x245
                + _pq_x246
                - _pq_x247
                + _pq_x248
                + _pq_x63
                + _pq_x64,
                2.68177957265292e-41 * T_5 + 7.49879874569759e-35 * T_6 + _pq_x240,
                6.12323399573677e-17 * T_0 * T_7
                - T_0 * _pq_x201
                - T_0 * _pq_x202
                + 1.9478619907097e-24 * T_1 * T_4
                + 3.45446742203778e-77 * T_1 * T_5
                + 5.44661656822022e-18 * T_2 * T_4
                + 7.5344114608425e-55 * T_2 * T_5
                - T_3 * _pq_x179
                - 0.0199999995529652 * T_8
                - 7.5344114608425e-55 * _pq_x77,
                7.15255763931247e-9 * T_10
                - T_3 * _pq_x180
                + T_3 * _pq_x181
                + T_7 * _pq_x178
                - T_7 * _pq_x264
                - 0.0199999995529639 * T_9
                + 1.41024823960635e-29 * _pq_x15
                - 5.04344102249169e-36 * _pq_x16
                - 1.41024823960635e-29 * _pq_x17
                + 5.04344102249169e-36 * _pq_x19
                + 5.44661656822057e-18 * _pq_x24
                - 5.44661656822057e-18 * _pq_x77,
            ],
            [
                T_0 * _pq_x277
                - T_1 * _pq_x75
                + 1.11022299978064e-18 * T_4
                - 8.03799262205824e-24 * T_5
                + 2.87461034093473e-30 * T_6
                + _pq_x69,
                8.03799262205876e-24 * T_4
                + 1.11022299978057e-18 * T_5
                - 3.97046708795757e-25 * T_6
                + 4.276423536148e-50 * _pq_x73
                - 4.27642353614751e-50 * _pq_x82,
                1.11022299978064e-18 * T_0
                - 8.03799262205824e-24 * T_1
                + 2.87461034093473e-30 * T_2
                - T_4 * _pq_x277
                - T_4 * _pq_x68
                + T_5 * _pq_x75,
                8.03799262205876e-24 * T_0
                + 1.11022299978057e-18 * T_1
                - 3.97046708795757e-25 * T_2
                + T_9 * _pq_x25
                - 4.276423536148e-50 * _pq_x215,
                T_0 * _pq_x281
                + T_0 * _pq_x282
                + 1.75162308040602e-46 * T_0
                - T_1 * _pq_x283
                - T_2 * _pq_x284
                + 9.21244041345997e-15 * T_4
                + _pq_x278
                + _pq_x279
                - _pq_x280
                - 4.59177480789956e-41 * _pq_x73
                + 4.59177480789956e-41 * _pq_x82,
                9.21244041345997e-15 * T_0
                - 0.0399999991059303 * T_3
                - T_4 * _pq_x281
                - T_4 * _pq_x282
                - 1.75162308040602e-46 * T_4
                + T_5 * _pq_x283
                + T_6 * _pq_x284
                + 4.59177480789956e-41 * _pq_x215
                - 4.59177480789956e-41 * _pq_x221
                + _pq_x285
                + _pq_x286,
                5.74922068186946e-30 * T_10
                + 2.22044599956128e-18 * T_8
                - 1.60759852441165e-23 * T_9
                + 4.93771683753601e-18 * _pq_x10
                - 1.43730520259364e-28 * _pq_x100
                + 4.01899640086069e-22 * _pq_x102
                - 1.43730520259364e-28 * _pq_x103
                + 4.01899640086069e-22 * _pq_x104
                - 5.5511151229803e-17 * _pq_x107
                + _pq_x109
                - 1.27848296104413e-29 * _pq_x11
                + _pq_x119
                + 1.27848296104413e-29 * _pq_x12
                - 1.43730520259364e-28 * _pq_x120
                + 1.27848296104413e-29 * _pq_x13
                - 1.27848296104413e-29 * _pq_x14
                - 5.5511151229803e-17 * _pq_x287
                + _pq_x288
                + 1.76586525373172e-24 * _pq_x29
                - 3.57489725197226e-23 * _pq_x30
                - 3.57489725197226e-23 * _pq_x32
                + 1.76586525373172e-24 * _pq_x34
                + 4.93771683753601e-18 * _pq_x35
                - 3.57489725197226e-23 * _pq_x36
                + 1.76586525373172e-24 * _pq_x38
                - 1.27848296104413e-29 * _pq_x50
                + 1.27848296104413e-29 * _pq_x52
                + 4.93771683753601e-18 * _pq_x9
                - 5.5511151229803e-17 * _pq_x93
                + 4.01899640086069e-22 * _pq_x96,
                -7.94093417591514e-25 * T_10
                + 1.60759852441175e-23 * T_8
                + 2.22044599956114e-18 * T_9
                + 3.57489725197249e-23 * _pq_x10
                + 1.98523358835221e-23 * _pq_x100
                - 5.55111512297994e-17 * _pq_x102
                + 1.98523358835221e-23 * _pq_x103
                - 5.55111512297994e-17 * _pq_x104
                - 4.01899640086095e-22 * _pq_x107
                + 1.76586525382416e-24 * _pq_x11
                - 1.76586525382416e-24 * _pq_x12
                + 1.98523358835221e-23 * _pq_x120
                - 1.76586525382416e-24 * _pq_x13
                + 1.76586525382416e-24 * _pq_x14
                - 4.01899640086095e-22 * _pq_x287
                + _pq_x289
                + 2.55696592208827e-29 * _pq_x29
                + 4.93771683753506e-18 * _pq_x30
                + 4.93771683753506e-18 * _pq_x32
                + 2.55696592208827e-29 * _pq_x34
                + 3.57489725197249e-23 * _pq_x35
                + 4.93771683753506e-18 * _pq_x36
                + 9.2561914231024e-35 * _pq_x39
                + 9.2561914231024e-35 * _pq_x48
                + 1.76586525382416e-24 * _pq_x50
                - 1.76586525382416e-24 * _pq_x52
                + 9.2561914231024e-35 * _pq_x56
                + 3.57489725197249e-23 * _pq_x9
                - 4.01899640086095e-22 * _pq_x93
                - 5.55111512297994e-17 * _pq_x96,
            ],
            [
                _pq_x261,
                T_0 * _pq_x290 - 6.68191177523049e-52 * T_2 * T_8 + _pq_x265,
                -_pq_x249,
                -T_4 * _pq_x290 + 6.68191177523049e-52 * T_6 * T_8 - _pq_x259,
                -1.55575381946529e-61 * T_0 * T_10
                + 1.55575381946529e-61 * T_2 * T_8
                - _pq_x274
                - _pq_x275,
                1.55575381946529e-61 * T_10 * T_4
                - 1.55575381946529e-61 * T_6 * T_8
                - _pq_x273,
                1.0 * T_0 * T_3
                - T_0 * _pq_x196
                - T_0 * _pq_x197
                + 1.23046276952446e-38 * T_1 * T_2
                + 1.23046276952446e-38 * T_10 * T_9
                + 1.0 * T_11 * T_8
                + 1.0 * T_4 * T_7
                - T_4 * _pq_x193
                - T_4 * _pq_x194
                + 1.23046276952446e-38 * T_5 * T_6
                - T_8 * _pq_x4
                - T_8 * _pq_x5
                - 0.0399999991059303 * T_8
                + 4.40047803938507e-45 * _pq_x11
                - 4.40047803938507e-45 * _pq_x12
                - 4.40047803938507e-45 * _pq_x13
                + 4.40047803938507e-45 * _pq_x14
                - 2.30311015484352e-13 * _pq_x39
                - 2.30311015484352e-13 * _pq_x48
                + 4.40047803938507e-45 * _pq_x50
                - 4.40047803938507e-45 * _pq_x52
                - 2.30311015484352e-13 * _pq_x56,
                8.23656425020362e-20 * T_0 * T_2
                + 0.999999999999936 * T_1 * T_3
                + 8.23656425020362e-20 * T_10 * T_8
                + 1.43051152786249e-8 * T_10
                + 0.999999999999936 * T_11 * T_9
                - T_11 * _pq_x185
                - T_3 * _pq_x173
                + 8.23656425020362e-20 * T_4 * T_6
                + 0.999999999999936 * T_5 * T_7
                - T_7 * _pq_x169
                - 3.18110003972698e-8 * _pq_x11
                + 3.18110003972698e-8 * _pq_x12
                + 3.18110003972698e-8 * _pq_x13
                - 3.18110003972698e-8 * _pq_x14
                - 2.30311015484337e-13 * _pq_x29
                - _pq_x291
                - 0.0889499988406669 * _pq_x30
                - 0.0889499988406669 * _pq_x32
                - 2.30311015484337e-13 * _pq_x34
                - 0.0889499988406669 * _pq_x36
                - 2.30311015484337e-13 * _pq_x38
                - 3.18110003972698e-8 * _pq_x50
                + 3.18110003972698e-8 * _pq_x52,
            ],
            [
                0.00119999994635582 * T_4 - _pq_x318,
                2.67276465035125e-53 * T_0 * T_10
                + 0.00119999994635574 * T_5
                - 4.29153448766421e-10 * T_6
                - _pq_x343
                - 2.67276465035125e-53 * _pq_x83,
                0.00119999994635582 * T_0 - 2.46092553904892e-38 * _pq_x344 - _pq_x377,
                0.00119999994635574 * T_1
                - 4.29153448766421e-10 * T_2
                + 2.67276465035125e-53 * T_6 * T_8
                - 2.67276465035125e-53 * _pq_x378
                - _pq_x403,
                6.22301513876591e-63 * T_2 * T_8
                + 2.62780698687423e-26 * T_5
                + 7.34788046640778e-20 * T_6
                - _pq_x431
                - 6.22301513876591e-63 * _pq_x65,
                2.62780698687423e-26 * T_1
                + 6.22301513876591e-63 * T_10 * T_4
                + 7.34788046640778e-20 * T_2
                - _pq_x454
                - 6.22301513876591e-63 * _pq_x81,
                3.50324616081204e-46 * T_0 * T_10 * T_4
                + 6.25256900105688e-83 * T_0 * T_10 * T_5
                + 0.0399999991059303 * T_0 * T_3
                + 1.67047794380762e-52 * T_0 * T_4 * T_9
                + 1.75162308040602e-46 * T_0 * T_4
                + 4.46542745808061e-79 * T_0 * T_5 * T_9
                + 1.24862394221809e-72 * T_0 * T_6 * T_9
                - T_0 * _pq_x285
                - T_0 * _pq_x286
                + 4.92185096808589e-40 * T_1 * T_2
                + 4.92185096808589e-40 * T_10 * T_9
                - 4.59177480789956e-41 * T_10 * _pq_x17
                + 0.0399999991059303 * T_11 * T_8
                + 4.59177480789956e-41 * T_2 * T_4 * T_9
                + 2.23609305867238e-89 * T_2 * T_6 * T_8
                + 0.0399999991059303 * T_4 * T_7
                - T_4 * _pq_x278
                - T_4 * _pq_x279
                + 4.92185096808589e-40 * T_5 * T_6
                - 0.00159999992847443 * T_8
                - 0.00355799987409986 * _pq_x10
                + 1.76019117641069e-46 * _pq_x11
                - 1.76019117641069e-46 * _pq_x12
                - 1.76019117641069e-46 * _pq_x13
                + 1.76019117641069e-46 * _pq_x14
                - _pq_x17 * _pq_x283
                - _pq_x19 * _pq_x284
                - 1.27243998744962e-9 * _pq_x38
                - 9.21244041345997e-15 * _pq_x39
                - 9.21244041345997e-15 * _pq_x48
                + 1.76019117641069e-46 * _pq_x50
                - 1.76019117641069e-46 * _pq_x52
                - 9.21244041345997e-15 * _pq_x56
                - 5.03279246486433e-90 * _pq_x66
                - 4.46542745808061e-79 * _pq_x76
                - 1.24862394221809e-72 * _pq_x78
                - 6.25256900105626e-83 * _pq_x80,
                3.07783577266036e-64 * T_0 * T_10 * T_4
                + 3.50324616081224e-46 * T_0 * T_10 * T_5
                + 3.29462562644083e-21 * T_0 * T_2
                + 1.67047794380752e-52 * T_0 * T_5 * T_9
                + 1.75162308040591e-46 * T_0 * T_5
                + 6.90893484389448e-77 * T_1 * T_10 * T_4
                + 1.64214673571707e-47 * T_1 * T_10 * T_6
                + 0.0399999991059278 * T_1 * T_3
                + 5.97409502267353e-59 * T_1 * T_6 * T_8
                + 3.29462562644083e-21 * T_10 * T_8
                + 5.72204598355228e-10 * T_10
                + 0.0399999991059278 * T_11 * T_9
                - T_11 * _pq_x455
                + 4.59177480789927e-41 * T_2 * T_5 * T_9
                + 6.4146443178375e-54 * T_2 * T_6 * T_8
                + 1.3363823550461e-51 * T_2 * T_7 * T_8
                - T_3 * _pq_x207
                + 3.29462562644083e-21 * T_4 * T_6
                + 0.0399999991059278 * T_5 * T_7
                - 1.64214673571707e-47 * T_6 * _pq_x82
                - T_7 * _pq_x225
                - 1.3363823550461e-51 * T_7 * _pq_x65
                - 6.90893484389448e-77 * T_9 * _pq_x19
                - 0.00159999992847432 * T_9
                - 1.27243998744954e-9 * _pq_x11
                + 1.27243998744954e-9 * _pq_x12
                + 1.27243998744954e-9 * _pq_x13
                - 1.27243998744954e-9 * _pq_x14
                - 6.26429266249492e-53 * _pq_x16
                - 9.21244041345939e-15 * _pq_x29
                - 0.00355799987409918 * _pq_x30
                - 0.00355799987409918 * _pq_x32
                - 9.21244041345939e-15 * _pq_x34
                - 0.00355799987409918 * _pq_x36
                - 9.21244041345939e-15 * _pq_x38
                - 1.27243998744954e-9 * _pq_x50
                + 1.27243998744954e-9 * _pq_x52
                - 6.4146443178375e-54 * _pq_x66
                - 5.97409502267353e-59 * _pq_x70
                - 4.59177480789927e-41 * _pq_x72
                - 1.67047794380752e-52 * _pq_x76
                - 3.07783577266036e-64 * _pq_x79
                - 3.50324616081224e-46 * _pq_x80,
            ],
            [
                T_0 * _pq_x456
                - 1.22464674440138e-19 * T_0
                - 6.90893484407556e-77 * T_1 * _pq_x104
                - T_11 * _pq_x213
                - 4.89858708709746e-18 * T_3 * T_8
                + 0.00159999992847443 * T_4
                - 6.90893484407556e-77 * T_7 * _pq_x71
                - T_8 * _pq_x278
                - T_8 * _pq_x279
                + T_8 * _pq_x280
                - _pq_x10 * _pq_x460
                + _pq_x103 * _pq_x465
                + _pq_x107 * _pq_x468
                - _pq_x12 * _pq_x473
                - _pq_x12 * _pq_x477
                + _pq_x12 * _pq_x482
                - _pq_x12 * _pq_x483
                + _pq_x120 * _pq_x465
                - _pq_x14 * _pq_x475
                - _pq_x14 * _pq_x479
                + _pq_x14 * _pq_x483
                + _pq_x14 * _pq_x485
                + _pq_x15 * _pq_x463
                + _pq_x16 * _pq_x464
                - _pq_x17 * _pq_x463
                - _pq_x177 * _pq_x43
                + _pq_x177 * _pq_x46
                - _pq_x177 * _pq_x54
                - _pq_x19 * _pq_x464
                + 4.92185096808715e-40 * _pq_x215
                - 4.92185096808715e-40 * _pq_x221
                + _pq_x287 * _pq_x468
                + _pq_x32 * _pq_x467
                - _pq_x34 * _pq_x472
                + 1.50688229216812e-54 * _pq_x344
                + 4.79354331948079e-62 * _pq_x345
                + 3.24795708865946e-42 * _pq_x346
                - 1.71430478277419e-68 * _pq_x347
                - _pq_x35 * _pq_x460
                + 1.55828955773728e-25 * _pq_x352
                - 9.58708663896097e-62 * _pq_x354
                - 1.34037178141384e-55 * _pq_x355
                + _pq_x36 * _pq_x467
                + 3.42666896342803e-68 * _pq_x360
                + 1.23926877120594e-31 * _pq_x362
                + 1.23926877120594e-31 * _pq_x363
                + 3.46524643630881e-25 * _pq_x365
                + 9.68953074857677e-19 * _pq_x366
                - 9.58708663896158e-62 * _pq_x370
                - 9.58708663896158e-62 * _pq_x371
                - 1.5068822921685e-54 * _pq_x372
                - 1.5068822921685e-54 * _pq_x373
                - 3.89572398141941e-24 * _pq_x375
                - 1.08932331364404e-17 * _pq_x376
                + 0.00355799987409986 * _pq_x378
                + 8.97228146207446e-37 * _pq_x379
                - _pq_x38 * _pq_x472
                - _pq_x39 * _pq_x476
                + 2.50883158556139e-30 * _pq_x391
                + 1.08932331364404e-17 * _pq_x394
                - 3.46524643630881e-25 * _pq_x396
                - 3.46524643630881e-25 * _pq_x397
                + 3.45446742203778e-77 * _pq_x399
                + 3.46524643630881e-25 * _pq_x402
                + 4.84476537428839e-19 * _pq_x435
                + 6.1963438560297e-32 * _pq_x443
                - 3.89572398141941e-24 * _pq_x448
                + 3.46524643630881e-25 * _pq_x451
                - 1.55828955773728e-25 * _pq_x457
                + 1.27243998744962e-9 * _pq_x458
                + 3.89572398141941e-24 * _pq_x459
                - _pq_x460 * _pq_x9
                + 1.67047794380762e-52 * _pq_x461
                + 3.50324616081204e-46 * _pq_x462
                + 3.46524643630881e-25 * _pq_x466
                + 3.46524643630881e-25 * _pq_x469
                + 9.68953074857677e-19 * _pq_x470
                + 3.42666896342803e-68 * _pq_x471
                - _pq_x473 * _pq_x52
                + _pq_x474 * _pq_x48
                + _pq_x474 * _pq_x56
                - _pq_x475 * _pq_x50
                - _pq_x476 * _pq_x48
                - _pq_x476 * _pq_x56
                - _pq_x477 * _pq_x52
                + 8.97228146207446e-37 * _pq_x478
                - _pq_x479 * _pq_x50
                + 8.97228146207446e-37 * _pq_x480
                + 5.38903134565114e-61 * _pq_x481
                + _pq_x482 * _pq_x52
                + _pq_x483 * _pq_x50
                - _pq_x483 * _pq_x52
                + 2.50883158556139e-30 * _pq_x484
                + _pq_x485 * _pq_x50
                + 2.50883158556139e-30 * _pq_x486
                - 5.38903134565114e-61 * _pq_x487
                - 4.35729315718308e-19 * _pq_x65
                - 6.02752903394823e-56 * _pq_x73
                + 6.02752903394823e-56 * _pq_x82
                + 4.35729315718308e-19 * _pq_x83,
                T_1 * _pq_x406
                - 1.2246467444013e-19 * T_1
                - 2.06795153138257e-25 * T_10 * T_6
                - T_11 * _pq_x224
                + T_11 * _pq_x225
                - T_2 * _pq_x408
                + T_2 * _pq_x505
                + T_2 * _pq_x506
                + 4.37967831145704e-26 * T_2
                - 4.89858708709714e-18 * T_3 * T_9
                + T_3 * _pq_x488
                + 0.00159999992847432 * T_5
                - 5.72204598355228e-10 * T_6
                + T_7 * _pq_x291
                - T_7 * _pq_x455
                + _pq_x10 * _pq_x494
                + _pq_x102 * _pq_x446
                - _pq_x103 * _pq_x447
                + _pq_x103 * _pq_x493
                + _pq_x104 * _pq_x446
                - _pq_x12 * _pq_x498
                - _pq_x12 * _pq_x501
                - _pq_x12 * _pq_x503
                + _pq_x12 * _pq_x504
                - _pq_x120 * _pq_x447
                + _pq_x120 * _pq_x493
                + _pq_x14 * _pq_x498
                + _pq_x14 * _pq_x502
                - _pq_x14 * _pq_x507
                - _pq_x15 * _pq_x490
                + _pq_x17 * _pq_x490
                - _pq_x178 * _pq_x43
                + _pq_x178 * _pq_x46
                - _pq_x178 * _pq_x54
                - _pq_x19 * _pq_x496
                + 0.00355799987410009 * _pq_x215
                - 0.00355799987410009 * _pq_x221
                + _pq_x264 * _pq_x43
                - _pq_x264 * _pq_x46
                + _pq_x264 * _pq_x54
                + 9.21244041345939e-15 * _pq_x300
                - _pq_x32 * _pq_x495
                - _pq_x34 * _pq_x497
                - 1.08932331364384e-17 * _pq_x344
                - 1.73262321815429e-25 * _pq_x345
                + 6.83044048817319e-55 * _pq_x346
                + 6.19634385602931e-32 * _pq_x347
                + _pq_x35 * _pq_x494
                - 1.12819856646788e-30 * _pq_x352
                + 3.46524643630837e-25 * _pq_x354
                + 4.84476537428684e-19 * _pq_x355
                - _pq_x36 * _pq_x495
                + 9.68953074857615e-19 * _pq_x360
                - 1.36608809763446e-54 * _pq_x362
                + 1.65681706283128e-52 * _pq_x363
                + 2.50883158556155e-30 * _pq_x365
                - 1.74719536171882e-67 * _pq_x366
                - 4.43106268962308e-38 * _pq_x370
                - 4.43565446443098e-38 * _pq_x371
                - 1.08932331364411e-17 * _pq_x372
                - 1.08932331364411e-17 * _pq_x373
                + 2.8204964792127e-29 * _pq_x375
                - 1.00868820449834e-35 * _pq_x376
                + 3.29462562644083e-21 * _pq_x378
                + 3.24795718300261e-42 * _pq_x379
                + _pq_x38 * _pq_x492
                - _pq_x38 * _pq_x497
                - 1.16156007403522e-48 * _pq_x391
                + 1.00868820449834e-35 * _pq_x394
                + 2.50883158556091e-30 * _pq_x396
                + 2.50883158556091e-30 * _pq_x397
                - 2.32312008059081e-48 * _pq_x399
                - 2.50883158556155e-30 * _pq_x402
                - 8.97228146207388e-37 * _pq_x435
                + 8.97228146207388e-37 * _pq_x443
                - 2.8204964792127e-29 * _pq_x448
                + 2.50883158556091e-30 * _pq_x451
                + 1.12819856646788e-30 * _pq_x457
                - 9.21244041345939e-15 * _pq_x458
                - 2.8204964792127e-29 * _pq_x459
                + 6.49591417731851e-42 * _pq_x461
                - 2.32312008059081e-48 * _pq_x462
                - 2.50883158556155e-30 * _pq_x466
                + 2.50883158556155e-30 * _pq_x469
                - 1.74719536171882e-67 * _pq_x470
                + 9.68953074857615e-19 * _pq_x471
                - 3.2479569943159e-42 * _pq_x478
                + _pq_x48 * _pq_x499
                - 3.2479569943159e-42 * _pq_x480
                - 3.89572398141916e-24 * _pq_x481
                + 1.16156000655559e-48 * _pq_x484
                + 1.16156000655559e-48 * _pq_x486
                + 3.89572398141916e-24 * _pq_x487
                + 1.00868820449834e-35 * _pq_x489
                + 1.00868820449834e-35 * _pq_x491
                - _pq_x498 * _pq_x52
                + _pq_x499 * _pq_x56
                + _pq_x50 * _pq_x502
                - _pq_x50 * _pq_x507
                + 8.97228146207388e-37 * _pq_x500
                - _pq_x501 * _pq_x52
                - _pq_x503 * _pq_x52
                + _pq_x504 * _pq_x52
                - 4.0347527278096e-37 * _pq_x65
                - 4.35729315718336e-19 * _pq_x73
                - 3.29462562644083e-21 * _pq_x81
                + 4.35729315718336e-19 * _pq_x82
                + 4.0347527278096e-37 * _pq_x83,
                3.89572398141941e-24 * T_0 * T_1 * T_7
                + 0.00355799987409986 * T_0 * T_10
                + 1.08932331364404e-17 * T_0 * T_2 * T_7
                + 3.89572398141941e-24 * T_0 * T_3 * T_5
                + 1.08932331364404e-17 * T_0 * T_3 * T_6
                + 1.27243998744962e-9 * T_0 * T_9
                - 1.67047794380762e-52 * T_0 * _pq_x17
                + 0.00159999992847443 * T_0
                + 4.92185096808715e-40 * T_1 * T_10
                + 3.46524643630881e-25 * T_1 * T_2 * T_4
                + 9.58708663896158e-62 * T_1 * T_2 * T_5
                + 6.90893484407556e-77 * T_1 * T_3 * T_5
                + 1.5068822921685e-54 * T_1 * T_3 * T_6
                + 3.46524643630881e-25 * T_10 * T_4 * T_9
                + 4.35729315718308e-19 * T_10 * T_4
                + 9.58708663896158e-62 * T_10 * T_5 * T_9
                + 6.02752903394823e-56 * T_10 * T_5
                + 1.08932331364404e-17 * T_10 * T_7 * T_8
                + 3.89572398141941e-24 * T_11 * T_5 * T_8
                + 6.90893484407556e-77 * T_11 * T_5 * T_9
                + 1.08932331364404e-17 * T_11 * T_6 * T_8
                + 1.5068822921685e-54 * T_11 * T_6 * T_9
                - T_11 * _pq_x203
                + 0.0399999991059303 * T_3 * T_8
                + 3.89572398141941e-24 * T_4 * T_5 * T_7
                + 1.08932331364404e-17 * T_4 * T_6 * T_7
                + 1.55828955773728e-25 * T_4 * T_9
                - 3.89572398141941e-24 * T_4 * _pq_x104
                + 6.1963438560297e-32 * T_4 * _pq_x11
                + 4.84476537428839e-19 * T_4 * _pq_x12
                + 4.84476537428839e-19 * T_4 * _pq_x13
                + 6.1963438560297e-32 * T_4 * _pq_x14
                + 6.12323399573677e-17 * T_4 * _pq_x43
                - T_4 * _pq_x456
                + 6.12323399573677e-17 * T_4 * _pq_x46
                + 1.22464674440138e-19 * T_4
                - 3.46524643630881e-25 * T_5 * _pq_x10
                + 1.71430478277419e-68 * T_5 * _pq_x11
                + 1.34037178141418e-55 * T_5 * _pq_x12
                + 1.34037178141418e-55 * T_5 * _pq_x13
                + 1.71430478277419e-68 * T_5 * _pq_x14
                + 1.34037178141384e-55 * T_5 * _pq_x52
                - 3.46524643630881e-25 * T_5 * _pq_x9
                + 9.58708663896097e-62 * T_6 * _pq_x50
                + 3.89572398141941e-24 * T_7 * T_8 * T_9
                + 4.89858708709746e-18 * T_7 * T_8
                + 5.38903134565115e-61 * T_7 * _pq_x12
                + 5.38903134565115e-61 * T_7 * _pq_x13
                + 2.82049647921288e-29 * T_7 * _pq_x39
                + 2.82049647921288e-29 * T_7 * _pq_x48
                - 5.38903134565114e-61 * T_7 * _pq_x50
                + 5.38903134565114e-61 * T_7 * _pq_x52
                + 2.82049647921288e-29 * T_7 * _pq_x56
                - T_8 * _pq_x285
                - T_8 * _pq_x286
                - _pq_x10 * _pq_x510
                - _pq_x11 * _pq_x519
                - _pq_x11 * _pq_x520
                - _pq_x12 * _pq_x521
                - _pq_x13 * _pq_x521
                - _pq_x14 * _pq_x519
                - _pq_x14 * _pq_x520
                - _pq_x17 * _pq_x472
                - _pq_x179 * _pq_x54
                - _pq_x19 * _pq_x460
                - 6.02752903394823e-56 * _pq_x221
                - _pq_x287 * _pq_x511
                - 4.79354331948079e-62 * _pq_x292
                - 3.24795708865946e-42 * _pq_x293
                + 1.71430478277419e-68 * _pq_x294
                - _pq_x30 * _pq_x512
                - _pq_x30 * _pq_x513
                - 1.55828955773728e-25 * _pq_x300
                - 1.23926877120594e-31 * _pq_x305
                - 1.23926877120594e-31 * _pq_x306
                - 3.46524643630881e-25 * _pq_x309
                - 1.50688229216812e-54 * _pq_x311
                - 1.5068822921685e-54 * _pq_x312
                - 1.5068822921685e-54 * _pq_x313
                - 8.97228146207446e-37 * _pq_x319
                - _pq_x32 * _pq_x512
                - _pq_x32 * _pq_x513
                - _pq_x39 * _pq_x516
                - 4.84476537428839e-19 * _pq_x411
                - 2.50883158556139e-30 * _pq_x414
                - 1.08932331364404e-17 * _pq_x417
                - 3.46524643630881e-25 * _pq_x508
                - 3.50324616081204e-46 * _pq_x509
                - _pq_x510 * _pq_x9
                - _pq_x511 * _pq_x93
                - 3.45446742203778e-77 * _pq_x514
                - 3.46524643630881e-25 * _pq_x515
                - _pq_x516 * _pq_x56
                - 8.97228146207446e-37 * _pq_x517
                - 2.50883158556139e-30 * _pq_x518
                - 6.1963438560297e-32 * _pq_x522
                - 8.97228146207446e-37 * _pq_x523
                - 2.50883158556139e-30 * _pq_x524
                - 4.35729315718308e-19 * _pq_x81
                - 4.92185096808715e-40 * _pq_x82,
                1.36608809763446e-54 * T_0 * T_1 * T_5
                + 2.8204964792127e-29 * T_0 * T_1 * T_7
                + 3.29462562644083e-21 * T_0 * T_10
                + 2.32312008059081e-48 * T_0 * T_2 * T_4
                + 2.50883158556155e-30 * T_0 * T_2 * T_5
                + 1.74719536171882e-67 * T_0 * T_2 * T_6
                + 1.00868820449834e-35 * T_0 * T_3 * T_6
                + 0.00355799987410009 * T_1 * T_10
                + 4.43565446443098e-38 * T_1 * T_2 * T_5
                + 1.08932331364383e-17 * T_1 * T_2 * T_7
                + 2.8204964792127e-29 * T_1 * T_3 * T_4
                + 1.08932331364411e-17 * T_1 * T_3 * T_6
                + 9.21244041345939e-15 * T_1 * T_8
                + 0.00159999992847432 * T_1
                + 4.37967850724393e-23 * T_10 * T_11 * T_7
                + 2.32312008059081e-48 * T_10 * T_4 * T_8
                + 4.0347527278096e-37 * T_10 * T_4
                + 2.50883158556155e-30 * T_10 * T_5 * T_8
                + 4.43106268962308e-38 * T_10 * T_5 * T_9
                + 4.35729315718336e-19 * T_10 * T_5
                + 1.74719536171882e-67 * T_10 * T_6 * T_8
                + 1.08932331364383e-17 * T_10 * T_7 * T_9
                + 1.43051152786249e-8 * T_11 * T_2
                + 2.8204964792127e-29 * T_11 * T_4 * T_9
                + 1.00868820449834e-35 * T_11 * T_6 * T_8
                + 1.08932331364411e-17 * T_11 * T_6 * T_9
                + 1.75187136374019e-24 * T_11 * T_6
                - T_11 * _pq_x206
                + 4.37967850724393e-23 * T_2 * T_3 * T_7
                - 5.72204598355228e-10 * T_2
                + 0.0399999991059278 * T_3 * T_9
                - T_3 * _pq_x455
                + 2.8204964792127e-29 * T_4 * T_5 * T_7
                - 1.00868820449834e-35 * T_4 * _pq_x103
                + 8.97228146207388e-37 * T_4 * _pq_x12
                + 8.97228146207388e-37 * T_4 * _pq_x13
                - 8.97228146207388e-37 * T_4 * _pq_x14
                - 6.49591417731851e-42 * T_4 * _pq_x38
                + 8.97228146207388e-37 * T_4 * _pq_x52
                + 1.08932331364384e-17 * T_5 * T_6 * T_7
                + 1.12819856646788e-30 * T_5 * T_8
                + 4.84476537428932e-19 * T_5 * _pq_x12
                + 4.84476537428932e-19 * T_5 * _pq_x13
                + 3.2479569943159e-42 * T_5 * _pq_x39
                - T_5 * _pq_x406
                + 6.12323399573637e-17 * T_5 * _pq_x43
                + 6.12323399573637e-17 * T_5 * _pq_x46
                + 3.2479569943159e-42 * T_5 * _pq_x56
                + 1.2246467444013e-19 * T_5
                + 1.73262321815429e-25 * T_6 * _pq_x12
                + 1.73262321815429e-25 * T_6 * _pq_x13
                + 1.16156007403522e-48 * T_6 * _pq_x48
                - T_6 * _pq_x506
                + 2.18983925362196e-23 * T_6 * _pq_x54
                - 4.37967831145704e-26 * T_6
                + 2.8204964792127e-29 * T_7 * T_8 * T_9
                + 4.89858708709714e-18 * T_7 * T_9
                + 3.89572398141916e-24 * T_7 * _pq_x11
                + 3.89572398141916e-24 * T_7 * _pq_x14
                - T_7 * _pq_x488
                + 3.89572398141916e-24 * T_7 * _pq_x50
                - 3.89572398141916e-24 * T_7 * _pq_x52
                - _pq_x10 * _pq_x496
                - _pq_x104 * _pq_x422
                - _pq_x11 * _pq_x529
                - _pq_x12 * _pq_x531
                - _pq_x13 * _pq_x531
                - _pq_x14 * _pq_x529
                - _pq_x15 * _pq_x497
                - _pq_x17 * _pq_x492
                - _pq_x180 * _pq_x54
                - _pq_x181 * _pq_x43
                - _pq_x181 * _pq_x46
                - 4.35729315718336e-19 * _pq_x221
                - 2.06795153138257e-25 * _pq_x276
                + 1.73262321815429e-25 * _pq_x292
                - 6.83044048817319e-55 * _pq_x293
                - 6.19634385602931e-32 * _pq_x294
                - _pq_x30 * _pq_x525
                - 1.65681706283128e-52 * _pq_x306
                - 2.50883158556155e-30 * _pq_x309
                - 1.08932331364411e-17 * _pq_x312
                - 1.08932331364411e-17 * _pq_x313
                - 3.24795718300261e-42 * _pq_x319
                - _pq_x32 * _pq_x525
                - 3.46524643630837e-25 * _pq_x323
                - 4.84476537428684e-19 * _pq_x325
                - 2.8204964792127e-29 * _pq_x337
                - _pq_x39 * _pq_x527
                - 1.00868820449834e-35 * _pq_x417
                - 2.50883158556091e-30 * _pq_x418
                - 2.50883158556091e-30 * _pq_x419
                - _pq_x422 * _pq_x96
                - 9.21244041345939e-15 * _pq_x457
                - 1.12819856646788e-30 * _pq_x458
                - _pq_x496 * _pq_x9
                - 2.50883158556155e-30 * _pq_x508
                - 2.50883158556091e-30 * _pq_x515
                - 1.16156000655559e-48 * _pq_x518
                - 8.97228146207388e-37 * _pq_x522
                - 1.16156000655559e-48 * _pq_x524
                - 1.00868820449834e-35 * _pq_x526
                - _pq_x527 * _pq_x56
                - 8.97228146207388e-37 * _pq_x528
                - 3.46524643630881e-25 * _pq_x530
                - 4.0347527278096e-37 * _pq_x81
                - 0.00355799987410009 * _pq_x82
                - 3.29462562644083e-21 * _pq_x83,
                -4.07831527827407e-56 * T_0 * _pq_x38
                + T_0 * _pq_x545
                + T_1 * _pq_x163
                + T_1 * _pq_x164
                + T_1 * _pq_x532
                + T_1 * _pq_x535
                + T_1 * _pq_x547
                - 2.68177951271048e-42 * T_1
                - 2.99951949827904e-34 * T_10 * T_3
                - T_11 * _pq_x63
                - T_11 * _pq_x64
                + T_2 * _pq_x157
                + T_2 * _pq_x533
                + T_2 * _pq_x551
                - 7.49879857808638e-36 * T_2
                - 1.07271182906117e-40 * T_3 * T_9
                + 5.47382212626882e-48 * T_5 * T_9
                + 3.50374264916564e-26 * T_5
                + 9.79717395521038e-20 * T_6
                - 7.0739000861336e-45 * T_7 * _pq_x19
                + T_7 * _pq_x536
                + T_7 * _pq_x59
                + T_7 * _pq_x60
                - _pq_x10 * _pq_x539
                + _pq_x102 * _pq_x541
                - _pq_x103 * _pq_x537
                + _pq_x103 * _pq_x538
                + _pq_x104 * _pq_x541
                - _pq_x12 * _pq_x543
                + _pq_x12 * _pq_x549
                - _pq_x12 * _pq_x550
                - _pq_x120 * _pq_x537
                + _pq_x120 * _pq_x538
                + _pq_x14 * _pq_x548
                - _pq_x14 * _pq_x552
                - _pq_x15 * _pq_x534
                + _pq_x17 * _pq_x534
                - _pq_x23 * _pq_x43
                + _pq_x23 * _pq_x46
                - _pq_x23 * _pq_x54
                - _pq_x26 * _pq_x43
                + _pq_x26 * _pq_x46
                - _pq_x26 * _pq_x54
                + 8.03799262205844e-24 * _pq_x300
                - _pq_x32 * _pq_x540
                + _pq_x34 * _pq_x542
                - 4.77088590420656e-40 * _pq_x344
                + 2.96656320412091e-35 * _pq_x345
                + 7.82845033003564e-46 * _pq_x346
                + 1.35689970048731e-54 * _pq_x347
                - _pq_x35 * _pq_x539
                - 9.84370193617391e-40 * _pq_x352
                + 1.1382499004554e-47 * _pq_x354
                + 3.18277721736157e-41 * _pq_x355
                - _pq_x36 * _pq_x540
                + 2.12185147824104e-41 * _pq_x360
                - 1.56569006600713e-45 * _pq_x362
                - 1.56569006600713e-45 * _pq_x363
                - 2.18899323845425e-39 * _pq_x365
                + 1.56569006600713e-45 * _pq_x366
                + 7.58833266970266e-48 * _pq_x370
                + 7.58833266970266e-48 * _pq_x371
                - 4.07831529249908e-56 * _pq_x372
                - 4.07831529249908e-56 * _pq_x373
                + 2.46092553904945e-38 * _pq_x375
                - 7.0739000861336e-45 * _pq_x376
                + 2.31051105770079e-30 * _pq_x378
                + 1.08128061535404e-40 * _pq_x379
                + _pq_x38 * _pq_x542
                - _pq_x39 * _pq_x546
                + 3.02347956009223e-34 * _pq_x391
                + 7.0739000861336e-45 * _pq_x394
                + 2.18899323845436e-39 * _pq_x396
                + 2.18899323845436e-39 * _pq_x397
                + 3.97760019462587e-58 * _pq_x399
                - 2.18899323845425e-39 * _pq_x402
                + 9.36466661546431e-46 * _pq_x435
                - 7.82845033003545e-46 * _pq_x443
                + 2.46092553904933e-38 * _pq_x448
                - 2.18899323845414e-39 * _pq_x451
                + 9.84370193617392e-40 * _pq_x457
                - 8.03799262205844e-24 * _pq_x458
                - 2.46092553904945e-38 * _pq_x459
                - 2.03915763202453e-56 * _pq_x461
                + 3.97760019462587e-58 * _pq_x462
                - 2.18899323845425e-39 * _pq_x466
                - 2.18899323845425e-39 * _pq_x469
                + 1.56569006600713e-45 * _pq_x470
                + 2.12185147824104e-41 * _pq_x471
                + _pq_x48 * _pq_x544
                - _pq_x48 * _pq_x546
                + 1.08128061535404e-40 * _pq_x480
                - 8.53100929578866e-47 * _pq_x481
                + 3.02347956009223e-34 * _pq_x484
                + 3.02347956009223e-34 * _pq_x486
                - 6.67018154645341e-34 * _pq_x487
                - 1.05280120714078e-44 * _pq_x489
                + 7.0739000861336e-45 * _pq_x491
                + _pq_x50 * _pq_x548
                - _pq_x50 * _pq_x552
                + 7.82845033003584e-46 * _pq_x500
                - _pq_x52 * _pq_x543
                + _pq_x52 * _pq_x549
                - _pq_x52 * _pq_x550
                + _pq_x544 * _pq_x56
                - _pq_x546 * _pq_x56
                - 2.82955997120784e-46 * _pq_x65
                - 2.31051105770079e-30 * _pq_x81
                + 1.27447352890596e-57 * _pq_x82
                + 2.82955997120784e-46 * _pq_x83,
                2.03915763202453e-56 * T_0 * T_1 * T_4
                + 1.56569006600713e-45 * T_0 * T_1 * T_5
                + 2.18899323845425e-39 * T_0 * T_1 * T_6
                + 2.31051105770079e-30 * T_0 * T_10
                + 2.18899323845425e-39 * T_0 * T_2 * T_5
                + 1.05280120714078e-44 * T_0 * T_2 * T_7
                + 7.0739000861336e-45 * T_0 * T_3 * T_6
                + 4.77088590420656e-40 * T_1 * T_2 * T_7
                + 2.46092553904945e-38 * T_1 * T_3 * T_4
                + 4.07831529249908e-56 * T_1 * T_3 * T_6
                + 8.03799262205844e-24 * T_1 * T_8
                + 5.47382212626882e-48 * T_1 * T_9
                + 3.50374264916564e-26 * T_1
                + 8.55284707229503e-50 * T_10 * T_11 * T_6
                + 2.44929354354857e-18 * T_10 * T_3
                + 2.82955997120784e-46 * T_10 * T_4
                + 2.18899323845425e-39 * T_10 * T_5 * T_8
                + 1.05280120714078e-44 * T_10 * T_7 * T_8
                + 4.77088590420656e-40 * T_10 * T_7 * T_9
                + 2.99951949827904e-34 * T_10 * T_7
                + 2.46092553904945e-38 * T_11 * T_4 * T_9
                + 7.0739000861336e-45 * T_11 * T_6 * T_8
                + 4.07831529249908e-56 * T_11 * T_6 * T_9
                - T_11 * _pq_x61
                - T_11 * _pq_x62
                + 8.55284707229503e-50 * T_2 * T_3 * T_6
                + 9.79717395521038e-20 * T_2
                + 8.75935681870097e-25 * T_3 * T_9
                - 2.46092553904945e-38 * T_3 * _pq_x15
                - T_3 * _pq_x536
                + 2.18899323845414e-39 * T_4 * T_5 * T_6
                + 1.05280120714078e-44 * T_4 * T_6 * T_7
                + 4.07831527827407e-56 * T_4 * T_8 * T_9
                + 6.29223404460698e-46 * T_4 * _pq_x12
                + 6.29223404460698e-46 * T_4 * _pq_x13
                + 7.82845033003545e-46 * T_4 * _pq_x50
                - T_4 * _pq_x545
                + 4.77088590420656e-40 * T_5 * T_6 * T_7
                + 1.56569006600713e-45 * T_5 * T_8 * T_9
                + 9.84370193617391e-40 * T_5 * T_8
                - 1.06092573912052e-41 * T_5 * _pq_x13
                + 1.34088981629768e-39 * T_5 * _pq_x43
                + 1.34088981629768e-39 * T_5 * _pq_x46
                - T_5 * _pq_x532
                - T_5 * _pq_x535
                - T_5 * _pq_x547
                + 2.68177951271048e-42 * T_5
                + 2.18899323845425e-39 * T_6 * T_8 * T_9
                - 2.12185147824104e-41 * T_6 * _pq_x30
                - 2.12185147824104e-41 * T_6 * _pq_x32
                + 3.7493994566544e-33 * T_6 * _pq_x43
                + 3.7493994566544e-33 * T_6 * _pq_x46
                - T_6 * _pq_x533
                - T_6 * _pq_x551
                + 7.49879857808638e-36 * T_6
                + 1.07271182906117e-40 * T_7 * T_9
                + 8.53100929578867e-47 * T_7 * _pq_x11
                + 6.67018154645341e-34 * T_7 * _pq_x12
                + 6.67018154645341e-34 * T_7 * _pq_x13
                + 8.53100929578867e-47 * T_7 * _pq_x14
                + 3.39907768352814e-33 * T_7 * _pq_x39
                + 3.39907768352814e-33 * T_7 * _pq_x48
                + 8.53100929578866e-47 * T_7 * _pq_x50
                + 6.67018154645341e-34 * T_7 * _pq_x52
                + 3.39907768352814e-33 * T_7 * _pq_x56
                - _pq_x10 * _pq_x554
                - _pq_x100 * _pq_x556
                - _pq_x104 * _pq_x555
                - _pq_x11 * _pq_x558
                - _pq_x12 * _pq_x559
                - _pq_x120 * _pq_x556
                - _pq_x13 * _pq_x559
                - _pq_x14 * _pq_x558
                - _pq_x182 * _pq_x54
                - _pq_x183 * _pq_x54
                - 1.27447352890596e-57 * _pq_x221
                - _pq_x29 * _pq_x553
                - 2.96656320412091e-35 * _pq_x292
                - 7.82845033003564e-46 * _pq_x293
                - 1.35689970048731e-54 * _pq_x294
                - 1.08128061535404e-40 * _pq_x319
                - 1.1382499004554e-47 * _pq_x323
                - 3.18277721736157e-41 * _pq_x325
                - 2.46092553904945e-38 * _pq_x337
                - _pq_x38 * _pq_x553
                - _pq_x39 * _pq_x557
                - 9.36466661546431e-46 * _pq_x411
                - 3.02347956009223e-34 * _pq_x414
                - 7.0739000861336e-45 * _pq_x417
                - 2.18899323845436e-39 * _pq_x418
                - 2.18899323845436e-39 * _pq_x419
                - 7.58833266970266e-48 * _pq_x424
                - 7.58833266970266e-48 * _pq_x425
                - 2.46092553904933e-38 * _pq_x430
                - 8.03799262205844e-24 * _pq_x457
                - 9.84370193617392e-40 * _pq_x458
                - 3.97760019462587e-58 * _pq_x509
                - 3.97760019462587e-58 * _pq_x514
                - 1.08128061535404e-40 * _pq_x517
                - 3.02347956009223e-34 * _pq_x518
                - 1.08128061535404e-40 * _pq_x523
                - 3.02347956009223e-34 * _pq_x524
                - 7.0739000861336e-45 * _pq_x526
                - 7.82845033003584e-46 * _pq_x528
                - 3.79416633485133e-48 * _pq_x530
                - _pq_x554 * _pq_x9
                - _pq_x555 * _pq_x96
                - _pq_x557 * _pq_x56
                - 2.82955997120784e-46 * _pq_x81
                - 2.31051105770079e-30 * _pq_x83,
                T_0 * _pq_x560
                - T_3 * _pq_x250
                + T_7 * _pq_x260
                + 0.177899997681368 * T_8 * _pq_x100
                + 6.36220007945436e-8 * T_8 * _pq_x102
                + 0.177899997681368 * T_8 * _pq_x103
                + T_8 * _pq_x27
                + T_8 * _pq_x28
                - T_8 * _pq_x296
                - T_8 * _pq_x31
                - T_8 * _pq_x315
                - T_8 * _pq_x37
                + T_8 * _pq_x374
                + T_8 * _pq_x40
                - T_8 * _pq_x41
                + T_8 * _pq_x42
                + T_8 * _pq_x44
                - T_8 * _pq_x45
                - T_8 * _pq_x47
                + T_8 * _pq_x49
                - T_8 * _pq_x51
                - T_8 * _pq_x53
                - T_8 * _pq_x55
                + T_8 * _pq_x58
                + 0.00119999994635582 * T_8
                - _pq_x10 * _pq_x566
                + _pq_x100 * _pq_x572
                - _pq_x102 * _pq_x570
                + _pq_x103 * _pq_x572
                - _pq_x107 * _pq_x565
                + _pq_x107 * _pq_x567
                - _pq_x107 * _pq_x569
                - 0.0799999982118607 * _pq_x107
                + _pq_x11 * _pq_x575
                - _pq_x11 * _pq_x576
                - 3.52038235282138e-46 * _pq_x11
                - 1.56569006600693e-45 * _pq_x115
                - 2.18899323845383e-39 * _pq_x117
                + 3.52038235282138e-46 * _pq_x12
                - _pq_x13 * _pq_x575
                + _pq_x13 * _pq_x577
                + 3.52038235282138e-46 * _pq_x13
                + 5.59544858130873e-52 * _pq_x131
                - 1.56569006600703e-45 * _pq_x136
                + 2.46092553904892e-38 * _pq_x139
                - 3.52038235282138e-46 * _pq_x14
                + 5.59544858130873e-52 * _pq_x141
                + 5.65917689691665e-9 * _pq_x142
                - 1.56569006600703e-45 * _pq_x147
                - 7.79144778868638e-26 * _pq_x15
                + 1.46528476101376e-20 * _pq_x151
                + 1.00190545973564e-58 * _pq_x152
                - 8.80095607877014e-45 * _pq_x158
                - 2.17864657859154e-19 * _pq_x16
                + 8.80095607877014e-45 * _pq_x160
                - 2.18899323845439e-39 * _pq_x162
                + 1.46528476101376e-20 * _pq_x166
                + 7.79144778868638e-26 * _pq_x17
                + 2.17864657859154e-19 * _pq_x19
                + 3.01376451697411e-56 * _pq_x24
                + _pq_x29 * _pq_x561
                - _pq_x29 * _pq_x562
                + _pq_x29 * _pq_x563
                + 2.54487997489924e-9 * _pq_x29
                + _pq_x30 * _pq_x571
                - 9.84370193617178e-40 * _pq_x30
                - 9.84370193617178e-40 * _pq_x32
                + _pq_x34 * _pq_x561
                - _pq_x34 * _pq_x562
                + _pq_x34 * _pq_x563
                + 2.54487997489924e-9 * _pq_x34
                + _pq_x35 * _pq_x564
                - _pq_x35 * _pq_x566
                + _pq_x35 * _pq_x568
                + 0.00711599974819972 * _pq_x35
                + _pq_x36 * _pq_x571
                - 9.84370193617178e-40 * _pq_x36
                + _pq_x39 * _pq_x573
                - _pq_x39 * _pq_x574
                + 1.84248808269199e-14 * _pq_x39
                + _pq_x48 * _pq_x573
                - _pq_x48 * _pq_x574
                + 1.84248808269199e-14 * _pq_x48
                + _pq_x50 * _pq_x575
                - _pq_x50 * _pq_x576
                - 3.52038235282138e-46 * _pq_x50
                - _pq_x52 * _pq_x575
                + _pq_x52 * _pq_x577
                + 3.52038235282138e-46 * _pq_x52
                + _pq_x56 * _pq_x573
                - _pq_x56 * _pq_x574
                + 1.84248808269199e-14 * _pq_x56
                + _pq_x564 * _pq_x9
                - _pq_x565 * _pq_x93
                - _pq_x566 * _pq_x9
                + _pq_x567 * _pq_x93
                + _pq_x568 * _pq_x9
                - _pq_x569 * _pq_x93
                - _pq_x570 * _pq_x96
                + 1.00190545973564e-58 * _pq_x578
                + 1.46528476101376e-20 * _pq_x579
                - 2.18899323845439e-39 * _pq_x580
                - 3.01376451697411e-56 * _pq_x77
                + 7.82845033003514e-46 * _pq_x86
                + 5.30431646238053e-26 * _pq_x87
                - 2.79967217318129e-52 * _pq_x88
                + 0.00711599974819972 * _pq_x9
                - 0.0799999982118607 * _pq_x93,
                -4.29153448766421e-10 * T_10
                - T_3 * _pq_x252
                + T_3 * _pq_x253
                + T_7 * _pq_x262
                - T_7 * _pq_x263
                + 0.00119999994635574 * T_9
                + _pq_x10 * _pq_x586
                - 6.58925125288165e-21 * _pq_x10
                - _pq_x100 * _pq_x594
                + _pq_x100 * _pq_x595
                + _pq_x100 * _pq_x596
                + 2.86102305572499e-8 * _pq_x100
                - _pq_x102 * _pq_x589
                + _pq_x102 * _pq_x591
                - _pq_x102 * _pq_x593
                - 0.0799999982118555 * _pq_x102
                - _pq_x103 * _pq_x594
                + _pq_x103 * _pq_x595
                + _pq_x103 * _pq_x596
                + 2.86102305572499e-8 * _pq_x103
                - 0.0799999982118555 * _pq_x104
                - 4.60622030968674e-13 * _pq_x105
                - _pq_x107 * _pq_x585
                + _pq_x107 * _pq_x588
                - _pq_x11 * _pq_x599
                + _pq_x11 * _pq_x600
                + _pq_x11 * _pq_x601
                + 2.54487997489908e-9 * _pq_x11
                + 5.65917689691592e-9 * _pq_x115
                - 1.46528476101366e-20 * _pq_x116
                + 0.00791210229375515 * _pq_x117
                - 2.54487997489908e-9 * _pq_x12
                + 2.86102305572499e-8 * _pq_x120
                + 1.06086329247604e-25 * _pq_x123
                - 3.79394300823431e-32 * _pq_x128
                - _pq_x13 * _pq_x602
                + _pq_x13 * _pq_x604
                - _pq_x13 * _pq_x605
                - 2.54487997489908e-9 * _pq_x13
                + 0.0158242045875143 * _pq_x131
                - 7.23783035983899e-22 * _pq_x136
                - 0.177899997681334 * _pq_x139
                + 2.54487997489908e-9 * _pq_x14
                + 0.0158242045875143 * _pq_x141
                + 4.09723291206518e-14 * _pq_x142
                + 1.06086329247604e-25 * _pq_x145
                - 3.79394300823431e-32 * _pq_x146
                - 7.23783035983899e-22 * _pq_x147
                + 5.64099283233938e-31 * _pq_x15
                - 5.30431630830579e-26 * _pq_x151
                + 5.65917689691665e-9 * _pq_x152
                + 6.36220007945395e-8 * _pq_x158
                - 2.0173763639048e-37 * _pq_x16
                - 6.36220007945395e-8 * _pq_x160
                - 0.0079121022937592 * _pq_x162
                + 1.46528476101366e-20 * _pq_x165
                + 5.30431661645458e-26 * _pq_x166
                - 5.64099283233938e-31 * _pq_x17
                - _pq_x185 * _pq_x43
                + _pq_x185 * _pq_x46
                + _pq_x185 * _pq_x54
                + _pq_x186 * _pq_x43
                - _pq_x186 * _pq_x46
                - _pq_x186 * _pq_x54
                + 2.0173763639048e-37 * _pq_x19
                + 2.17864657859168e-19 * _pq_x24
                + _pq_x29 * _pq_x581
                - _pq_x29 * _pq_x582
                - _pq_x29 * _pq_x583
                + 1.84248808269188e-14 * _pq_x29
                - _pq_x30 * _pq_x590
                + _pq_x30 * _pq_x592
                + 0.00711599974819836 * _pq_x30
                + 0.00711599974819836 * _pq_x32
                + _pq_x34 * _pq_x581
                - _pq_x34 * _pq_x582
                - _pq_x34 * _pq_x583
                + 1.84248808269188e-14 * _pq_x34
                - _pq_x35 * _pq_x584
                + _pq_x35 * _pq_x586
                - _pq_x35 * _pq_x587
                - 6.58925125288165e-21 * _pq_x35
                - _pq_x36 * _pq_x590
                + _pq_x36 * _pq_x592
                + 0.00711599974819836 * _pq_x36
                + 1.84248808269188e-14 * _pq_x38
                + _pq_x39 * _pq_x597
                + _pq_x39 * _pq_x598
                + _pq_x48 * _pq_x597
                + _pq_x48 * _pq_x598
                - _pq_x50 * _pq_x599
                + _pq_x50 * _pq_x600
                + _pq_x50 * _pq_x601
                + 2.54487997489908e-9 * _pq_x50
                - _pq_x52 * _pq_x602
                + _pq_x52 * _pq_x604
                - _pq_x52 * _pq_x605
                - 2.54487997489908e-9 * _pq_x52
                + 5.65917689691665e-9 * _pq_x578
                - 5.30431630830579e-26 * _pq_x579
                - 0.0079121022937592 * _pq_x580
                - _pq_x584 * _pq_x9
                - _pq_x585 * _pq_x93
                + _pq_x586 * _pq_x9
                - _pq_x587 * _pq_x9
                + _pq_x588 * _pq_x93
                - _pq_x589 * _pq_x96
                + _pq_x591 * _pq_x96
                - _pq_x593 * _pq_x96
                - 1.89697155921846e-32 * _pq_x603
                - 2.17864657859168e-19 * _pq_x77
                - 2.82958844845814e-9 * _pq_x86
                + 1.11549558500113e-38 * _pq_x87
                + 1.01193974627516e-15 * _pq_x88
                - 6.58925125288165e-21 * _pq_x9
                - 0.0799999982118555 * _pq_x96,
            ],
            [
                -0.00199999991059303 * T_0
                + T_11 * _pq_x250
                - 9.79717395521101e-20 * T_4
                - T_7 * _pq_x606
                - T_8 * _pq_x560
                - 3.01376451697411e-56 * _pq_x215
                + 3.01376451697411e-56 * _pq_x221
                + 7.79144778868638e-26 * _pq_x300
                + 2.46092553904892e-38 * _pq_x344
                + _pq_x377
                - 2.17864657859154e-19 * _pq_x378
                - 7.79144778868638e-26 * _pq_x458
                + 2.17864657859154e-19 * _pq_x81,
                -0.00199999991059291 * T_1
                + 8.75935681870097e-25 * T_10 * T_7
                + T_11 * _pq_x252
                - T_11 * _pq_x253
                + 7.15255747944035e-10 * T_2
                - 9.79717395521038e-20 * T_5
                + 3.50374264916564e-26 * T_6
                - T_7 * _pq_x607
                - 2.17864657859168e-19 * _pq_x215
                + 2.17864657859168e-19 * _pq_x221
                - 5.64099283233938e-31 * _pq_x300
                - 2.0173763639048e-37 * _pq_x378
                + _pq_x403
                + 5.64099283233938e-31 * _pq_x458
                + 2.0173763639048e-37 * _pq_x81,
                2.44929354354873e-18 * T_0 * T_11
                - 8.75811540203011e-47 * T_0 * T_8
                - 9.79717395521101e-20 * T_0
                + 7.79144778868638e-26 * T_1 * T_8
                + 2.17864657859154e-19 * T_2 * T_8
                + 3.01376451697411e-56 * T_2 * T_9
                - T_3 * _pq_x606
                + 0.00199999991059303 * T_4
                - _pq_x318
                - 7.79144778868638e-26 * _pq_x457
                - 2.17864657859154e-19 * _pq_x65
                - 3.01376451697411e-56 * _pq_x73,
                5.64099283233938e-31 * T_0 * T_9
                + 2.44929354354857e-18 * T_1 * T_11
                - 9.79717395521038e-20 * T_1
                + 8.75935681870097e-25 * T_10 * T_3
                - T_11 * _pq_x263
                + 2.0173763639048e-37 * T_2 * T_8
                + 2.17864657859168e-19 * T_2 * T_9
                + 3.50374264916564e-26 * T_2
                - T_3 * _pq_x607
                + 0.00199999991059291 * T_5
                - 7.15255747944035e-10 * T_6
                - _pq_x343
                - 5.64099283233938e-31 * _pq_x352
                - 2.0173763639048e-37 * _pq_x65
                - 2.17864657859168e-19 * _pq_x73,
                -4.37967831145704e-26 * T_1
                + T_11 * _pq_x266
                + T_11 * _pq_x267
                - 1.2246467444013e-19 * T_2
                - 2.14542361016838e-42 * T_5
                - 5.9990388624691e-36 * T_6
                - T_7 * _pq_x608
                - T_7 * _pq_x609
                - 6.37236764452981e-58 * _pq_x215
                - 4.92185096808696e-40 * _pq_x300
                - 1.41477998560392e-46 * _pq_x378
                + _pq_x454
                + 4.92185096808696e-40 * _pq_x458
                + 1.41477998560392e-46 * _pq_x81,
                4.92185096808696e-40 * T_0 * T_9
                + 5.36355914530584e-41 * T_1 * T_11
                - 2.14542361016838e-42 * T_1
                + 1.49975974913952e-34 * T_11 * T_2
                + 1.41477998560392e-46 * T_2 * T_8
                - 5.9990388624691e-36 * T_2
                - T_3 * _pq_x608
                - T_3 * _pq_x609
                + 4.37967831145704e-26 * T_5
                + 1.2246467444013e-19 * T_6
                - 4.92185096808696e-40 * _pq_x352
                - _pq_x431
                - 1.41477998560392e-46 * _pq_x65
                - 6.37236764452981e-58 * _pq_x73,
                3.89572398141941e-24 * T_0 * T_1 * T_11
                + 1.08932331364404e-17 * T_0 * T_10 * T_3
                + 1.08932331364404e-17 * T_0 * T_11 * T_2
                + 3.89572398141941e-24 * T_0 * T_3 * T_9
                + 4.89858708709746e-18 * T_0 * T_3
                + 0.0399999991059303 * T_0 * T_7
                - T_0 * _pq_x278
                - T_0 * _pq_x279
                + 1.5068822921685e-54 * T_1 * T_10 * T_3
                + 3.46524643630881e-25 * T_1 * T_2 * T_8
                + 9.58708663896158e-62 * T_1 * T_2 * T_9
                + 6.02752903394669e-56 * T_1 * T_2
                + 6.90893484407556e-77 * T_1 * T_3 * T_9
                + 1.27243998744962e-9 * T_1 * T_4
                + 1.08932331364404e-17 * T_10 * T_11 * T_8
                + 1.08932331364404e-17 * T_10 * T_4 * T_7
                + 1.5068822921685e-54 * T_10 * T_5 * T_7
                + 6.02752903394669e-56 * T_10 * T_9
                + 9.58708663896097e-62 * T_10 * _pq_x14
                + 3.89572398141941e-24 * T_11 * T_4 * T_5
                + 1.08932331364404e-17 * T_11 * T_4 * T_6
                + 3.89572398141941e-24 * T_11 * T_8 * T_9
                + 4.89858708709746e-18 * T_11 * T_8
                + 5.38903134565114e-61 * T_11 * _pq_x12
                + 5.38903134565115e-61 * T_11 * _pq_x13
                + 2.82049647921288e-29 * T_11 * _pq_x39
                + 2.82049647921288e-29 * T_11 * _pq_x48
                + 5.38903134565115e-61 * T_11 * _pq_x52
                + 2.82049647921288e-29 * T_11 * _pq_x56
                + 0.00355799987409986 * T_2 * T_4
                + 4.92185096808715e-40 * T_2 * T_5
                - T_3 * _pq_x213
                + 3.89572398141941e-24 * T_4 * T_7 * T_9
                + 4.89858708709746e-18 * T_4 * T_7
                + 3.46524643630881e-25 * T_5 * T_6 * T_8
                + 9.58708663896158e-62 * T_5 * T_6 * T_9
                + 6.02752903394669e-56 * T_5 * T_6
                + 6.90893484407556e-77 * T_5 * T_7 * T_9
                + 6.1963438560297e-32 * T_8 * _pq_x11
                + 4.84476537428839e-19 * T_8 * _pq_x13
                + 6.12323399573677e-17 * T_8 * _pq_x46
                + 6.1963438560297e-32 * T_8 * _pq_x50
                + 4.84476537428839e-19 * T_8 * _pq_x52
                + 6.12323399573677e-17 * T_8 * _pq_x54
                - 7.34788046640825e-20 * T_8
                + 1.71430478277419e-68 * T_9 * _pq_x11
                + 1.34037178141384e-55 * T_9 * _pq_x12
                + 1.34037178141418e-55 * T_9 * _pq_x13
                - 3.46524643630881e-25 * T_9 * _pq_x35
                + 1.71430478277419e-68 * T_9 * _pq_x50
                + 1.34037178141418e-55 * T_9 * _pq_x52
                - 3.46524643630881e-25 * T_9 * _pq_x9
                - 4.35729315718308e-19 * _pq_x10
                - _pq_x100 * _pq_x617
                - _pq_x100 * _pq_x618
                - _pq_x102 * _pq_x615
                - _pq_x103 * _pq_x617
                - _pq_x103 * _pq_x618
                - _pq_x107 * _pq_x613
                - _pq_x11 * _pq_x621
                + 2.15561249007876e-62 * _pq_x11
                - 4.84476537428839e-19 * _pq_x116
                - 2.15561249007876e-62 * _pq_x12
                - 3.45446742203778e-77 * _pq_x128
                - _pq_x13 * _pq_x622
                - 2.15561249007876e-62 * _pq_x13
                - 3.42666896342803e-68 * _pq_x131
                - 1.50688229216812e-54 * _pq_x139
                + 2.15561249007876e-62 * _pq_x14
                - 3.42666896342803e-68 * _pq_x141
                - 3.46524643630881e-25 * _pq_x142
                - 8.97228146207446e-37 * _pq_x151
                - 6.12304350556196e-75 * _pq_x152
                - 5.38903134565114e-61 * _pq_x160
                - 6.1963438560297e-32 * _pq_x165
                - 8.97228146207446e-37 * _pq_x166
                - _pq_x187 * _pq_x43
                - _pq_x283 * _pq_x34
                - _pq_x284 * _pq_x35
                - _pq_x29 * _pq_x611
                - 1.55828955773728e-25 * _pq_x29
                - _pq_x30 * _pq_x614
                - _pq_x34 * _pq_x611
                - 1.55828955773728e-25 * _pq_x34
                - _pq_x35 * _pq_x612
                - 4.35729315718308e-19 * _pq_x35
                - _pq_x36 * _pq_x614
                - 1.55828955773728e-25 * _pq_x38
                - _pq_x39 * _pq_x620
                - 1.12819856646795e-30 * _pq_x39
                - _pq_x48 * _pq_x620
                - 1.12819856646795e-30 * _pq_x48
                - _pq_x50 * _pq_x621
                + 2.15561249007876e-62 * _pq_x50
                - _pq_x52 * _pq_x622
                - 2.15561249007876e-62 * _pq_x52
                - 1.12819856646795e-30 * _pq_x56
                - 6.12304350556196e-75 * _pq_x578
                - 8.97228146207446e-37 * _pq_x579
                - 2.50883158556139e-30 * _pq_x603
                - 3.46524643630881e-25 * _pq_x610
                - _pq_x612 * _pq_x9
                - _pq_x613 * _pq_x93
                - _pq_x615 * _pq_x96
                - 3.46524643630881e-25 * _pq_x616
                - 2.50883158556139e-30 * _pq_x619
                - 2.50883158556139e-30 * _pq_x623
                - 4.92185096808715e-40 * _pq_x77
                - 4.79354331948079e-62 * _pq_x86
                - 3.24795708865946e-42 * _pq_x87
                + 1.71430478277419e-68 * _pq_x88
                - 4.35729315718308e-19 * _pq_x9,
                2.8204964792127e-29 * T_0 * T_1 * T_11
                + 1.74719536171882e-67 * T_0 * T_10 * T_2
                + 1.00868820449834e-35 * T_0 * T_10 * T_3
                + 2.32312008059081e-48 * T_0 * T_2 * T_8
                + 2.50883158556155e-30 * T_0 * T_2 * T_9
                + 4.0347527278096e-37 * T_0 * T_2
                + 9.21244041345939e-15 * T_0 * T_5
                + 1.08932331364411e-17 * T_1 * T_10 * T_3
                + 1.08932331364383e-17 * T_1 * T_11 * T_2
                + 4.43106268962308e-38 * T_1 * T_2 * T_9
                + 2.8204964792127e-29 * T_1 * T_3 * T_8
                + 4.89858708709714e-18 * T_1 * T_3
                + 0.0399999991059278 * T_1 * T_7
                + 1.08932331364384e-17 * T_10 * T_11 * T_9
                + 1.74719536171882e-67 * T_10 * T_4 * T_6
                + 1.00868820449834e-35 * T_10 * T_4 * T_7
                + 1.08932331364411e-17 * T_10 * T_5 * T_7
                + 4.0347527278096e-37 * T_10 * T_8
                + 1.73262321815429e-25 * T_10 * _pq_x13
                + 2.18983925362196e-23 * T_10 * _pq_x43
                - T_10 * _pq_x505
                + 1.73262321815429e-25 * T_10 * _pq_x52
                + 1.16156007403522e-48 * T_10 * _pq_x56
                + 2.62780698687423e-26 * T_10
                + 4.37967850724393e-23 * T_11 * T_2 * T_3
                + 2.8204964792127e-29 * T_11 * T_4 * T_5
                + 1.08932331364383e-17 * T_11 * T_5 * T_6
                + 4.37967850724393e-23 * T_11 * T_6 * T_7
                + 2.8204964792127e-29 * T_11 * T_8 * T_9
                + 4.89858708709714e-18 * T_11 * T_9
                - 1.00868820449834e-35 * T_11 * _pq_x10
                + 3.89572398141916e-24 * T_11 * _pq_x11
                + 3.89572398141916e-24 * T_11 * _pq_x14
                + 3.89572398141916e-24 * T_11 * _pq_x50
                + 3.29462562644083e-21 * T_2 * T_4
                + 0.00355799987410009 * T_2 * T_5
                - 2.06795153138257e-25 * T_2 * T_6
                + 1.43051152786249e-8 * T_3 * T_6
                - T_3 * _pq_x224
                + 1.36608809763446e-54 * T_4 * T_5 * T_9
                + 2.32312008059081e-48 * T_4 * T_6 * T_8
                + 2.50883158556155e-30 * T_4 * T_6 * T_9
                + 4.0347527278096e-37 * T_4 * T_6
                + 4.43565446443098e-38 * T_5 * T_6 * T_9
                + 2.8204964792127e-29 * T_5 * T_7 * T_8
                + 4.89858708709714e-18 * T_5 * T_7
                - T_7 * _pq_x207
                - 1.00868820449834e-35 * T_8 * _pq_x100
                - 1.00868820449834e-35 * T_8 * _pq_x103
                - 8.97228146207388e-37 * T_8 * _pq_x11
                + 8.97228146207388e-37 * T_8 * _pq_x12
                + 8.97228146207388e-37 * T_8 * _pq_x13
                - 2.50883158556091e-30 * T_8 * _pq_x30
                - 2.50883158556091e-30 * T_8 * _pq_x36
                - 8.97228146207388e-37 * T_8 * _pq_x50
                + 8.97228146207388e-37 * T_8 * _pq_x52
                + 4.84476537428932e-19 * T_9 * _pq_x13
                - 1.65681706283128e-52 * T_9 * _pq_x29
                + 3.2479569943159e-42 * T_9 * _pq_x39
                + 6.12323399573637e-17 * T_9 * _pq_x46
                + 3.2479569943159e-42 * T_9 * _pq_x48
                + 4.84476537428932e-19 * T_9 * _pq_x52
                + 6.12323399573637e-17 * T_9 * _pq_x54
                - 7.34788046640778e-20 * T_9
                - _pq_x100 * _pq_x626
                - 1.75187136374019e-24 * _pq_x100
                - _pq_x101 * _pq_x102
                - _pq_x101 * _pq_x96
                - _pq_x103 * _pq_x626
                - 1.75187136374019e-24 * _pq_x103
                - _pq_x107 * _pq_x625
                - _pq_x11 * _pq_x628
                - 1.55828955773718e-25 * _pq_x11
                - 3.46524643630837e-25 * _pq_x115
                - 4.84476537428684e-19 * _pq_x117
                + 1.55828955773718e-25 * _pq_x12
                - 1.75187136374019e-24 * _pq_x120
                - 6.49591417731851e-42 * _pq_x123
                - _pq_x13 * _pq_x629
                + 1.55828955773718e-25 * _pq_x13
                - 9.68953074857615e-19 * _pq_x131
                - 1.55828955773718e-25 * _pq_x14
                - 9.68953074857615e-19 * _pq_x141
                - 2.50883158556091e-30 * _pq_x142
                - 6.49591417731851e-42 * _pq_x145
                - 3.46524643630881e-25 * _pq_x152
                - 3.89572398141916e-24 * _pq_x158
                - 3.29462562644083e-21 * _pq_x16
                - 8.97228146207388e-37 * _pq_x165
                - 3.24795718300261e-42 * _pq_x166
                - 9.21244041345939e-15 * _pq_x17
                - _pq_x188 * _pq_x46
                - _pq_x188 * _pq_x54
                - _pq_x255 * _pq_x43
                - 1.12819856646788e-30 * _pq_x29
                - 4.35729315718225e-19 * _pq_x30
                - 4.35729315718225e-19 * _pq_x32
                - 1.12819856646788e-30 * _pq_x34
                - _pq_x35 * _pq_x624
                - 4.35729315718225e-19 * _pq_x36
                - 1.12819856646788e-30 * _pq_x38
                - _pq_x39 * _pq_x627
                - _pq_x48 * _pq_x627
                - _pq_x50 * _pq_x628
                - 1.55828955773718e-25 * _pq_x50
                - _pq_x52 * _pq_x629
                + 1.55828955773718e-25 * _pq_x52
                - 2.50883158556155e-30 * _pq_x610
                - 2.50883158556155e-30 * _pq_x616
                - 1.16156000655559e-48 * _pq_x619
                - 1.16156000655559e-48 * _pq_x623
                - _pq_x624 * _pq_x9
                - _pq_x625 * _pq_x93
                - 0.00355799987410009 * _pq_x77
                + 1.73262321815429e-25 * _pq_x86
                - 6.83044048817319e-55 * _pq_x87
                - 6.19634385602931e-32 * _pq_x88,
            ],
        ],
        dtype=np.float64,
    )

    return p_sin, p_cos, p_one, q


def _solve_algebraic(T_target):
    """Tier-2 Raghavan-Roth IK candidates with INLINED P/Q matrices.

    Per-arm DH constants substituted into the 4 RR matrix builders
    (CSE'd above); generic linear algebra (eliminate_q0_q1,
    Weierstrass, eigvals, back-substitution) stays imported.
    """
    T = np.asarray(T_target, dtype=np.float64)
    T_dh = _T_PRE_INV @ T @ _T_POST_INV

    # Destructure T_dh's free entries (top 3 rows, 4 cols).
    T_0, T_1, T_2, T_3 = T_dh[0, 0], T_dh[0, 1], T_dh[0, 2], T_dh[0, 3]
    T_4, T_5, T_6, T_7 = T_dh[1, 0], T_dh[1, 1], T_dh[1, 2], T_dh[1, 3]
    T_8, T_9, T_10, T_11 = T_dh[2, 0], T_dh[2, 1], T_dh[2, 2], T_dh[2, 3]
    p_sin, p_cos, p_one, q_mat = _build_pq_matrices(
        T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11
    )

    # Generic linear algebra: arm-agnostic, runtime-imported. Phase 4
    # Cython links these to LAPACK directly.
    e_sin, e_cos, e_one = _ssik_eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = _ssik_weierstrass(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = _ssik_build_m_matrix(e_quad, e_lin, e_const)
    # Companion route where accurate, QZ on the pencil where not, and
    # every branch vector a repeated root carries (#595, #640), as
    # solve_all_ik and the native core do.
    roots, eigvecs = _ssik_solve_x2_branches(m_quad, m_lin, m_const)

    q_pinv = np.linalg.pinv(q_mat).astype(np.float64)

    # Back-substitute per real root. ``_back_substitute_inner``
    # returns ``(q_dh, fk_err_alg)`` -- we drop fk_err here because
    # the outer ``solve()`` re-runs FK on each candidate (mirroring
    # the wrapper version's behavior, kept for parity).
    inner_qs = []
    for x_lin, eigvec in zip(roots, eigvecs):
        bs_result = _ssik_back_substitute_inner(
            x_lin,
            eigvec,
            p_sin,
            p_cos,
            p_one,
            q_pinv,
            _DH_TUPLE,
            T_dh,
            _RR_META,
        )
        if bs_result is None:
            continue
        q_dh, _fk_err = bs_result
        inner_qs.append(q_dh)

    # Map DH-frame q back to POE frame.
    return [
        list(np.asarray(q_dh, dtype=np.float64) - _DH_THETA_OFFSET) for q_dh in inner_qs
    ]


# Module-scope ``2*pi`` constant referenced inside the dedup hot
# loop (Cython compiles ``_TWO_PI`` to a typed C ``double``).
_TWO_PI: float = 2.0 * math.pi

# The POE chain accepted candidates are polished on (ssik.refinement.polish).
_POLISH_CHAIN = _PolishChain.from_kinbody(_KB)

# Cached 4x4 identity reused inside ``_fk`` / ``_spatial_jacobian``
# so each call avoids ``len(_JOINT_AXES)+1`` per-iteration ``np.eye(4)``
# allocations -- the orchestrator's #1 hotspot per Slice 4 profile
# (~22% of ``_fk`` cost on Puma 560).
_FK_EYE4 = np.eye(4, dtype=np.float64)
_FK_EYE4.flags.writeable = False


@cython.ccall
@cython.locals(
    i=cython.int,
    n=cython.int,
    ax=cython.double,
    ay=cython.double,
    az=cython.double,
    qi=cython.double,
    c=cython.double,
    s=cython.double,
    oc=cython.double,
    r00=cython.double,
    r01=cython.double,
    r02=cython.double,
    r10=cython.double,
    r11=cython.double,
    r12=cython.double,
    r20=cython.double,
    r21=cython.double,
    r22=cython.double,
    l00=cython.double,
    l01=cython.double,
    l02=cython.double,
    l03=cython.double,
    l10=cython.double,
    l11=cython.double,
    l12=cython.double,
    l13=cython.double,
    l20=cython.double,
    l21=cython.double,
    l22=cython.double,
    l23=cython.double,
    m00=cython.double,
    m01=cython.double,
    m02=cython.double,
    m03=cython.double,
    m10=cython.double,
    m11=cython.double,
    m12=cython.double,
    m13=cython.double,
    m20=cython.double,
    m21=cython.double,
    m22=cython.double,
    m23=cython.double,
    t00=cython.double,
    t01=cython.double,
    t02=cython.double,
    t03=cython.double,
    t10=cython.double,
    t11=cython.double,
    t12=cython.double,
    t13=cython.double,
    t20=cython.double,
    t21=cython.double,
    t22=cython.double,
    t23=cython.double,
    n00=cython.double,
    n01=cython.double,
    n02=cython.double,
    n03=cython.double,
    n10=cython.double,
    n11=cython.double,
    n12=cython.double,
    n13=cython.double,
    n20=cython.double,
    n21=cython.double,
    n22=cython.double,
    n23=cython.double,
    a00=cython.double,
    a01=cython.double,
    a02=cython.double,
    a03=cython.double,
    a10=cython.double,
    a11=cython.double,
    a12=cython.double,
    a13=cython.double,
    a20=cython.double,
    a21=cython.double,
    a22=cython.double,
    a23=cython.double,
    b00=cython.double,
    b01=cython.double,
    b02=cython.double,
    b03=cython.double,
    b10=cython.double,
    b11=cython.double,
    b12=cython.double,
    b13=cython.double,
    b20=cython.double,
    b21=cython.double,
    b22=cython.double,
    b23=cython.double,
)
def _fk(q):
    """POE forward kinematics using the baked chain constants.

    Hand-rolled scalar 4x4 matmul + inline Rodrigues -- no per-call
    ``np.eye(4)`` allocations and no per-joint numpy ``@`` dispatch.
    Each numpy ``@`` on a 4x4 has ~3 us of dispatch overhead;
    inlining the ~85 scalar ops per joint turns the inner loop into
    a single native-code chunk under Cython compile.

    Bottom row of the accumulator stays [0, 0, 0, 1] implicitly.
    """
    n = len(_JOINT_AXES)
    # Identity accumulator (the bottom row [0,0,0,1] is implicit).
    a00 = 1.0
    a01 = 0.0
    a02 = 0.0
    a03 = 0.0
    a10 = 0.0
    a11 = 1.0
    a12 = 0.0
    a13 = 0.0
    a20 = 0.0
    a21 = 0.0
    a22 = 1.0
    a23 = 0.0
    for i in range(n):
        # Inline Rodrigues for this joint's axis.
        ax = float(_JOINT_AXES[i][0])
        ay = float(_JOINT_AXES[i][1])
        az = float(_JOINT_AXES[i][2])
        qi = float(q[i])
        c = math.cos(qi)
        s = math.sin(qi)
        oc = 1.0 - c
        r00 = c + ax * ax * oc
        r01 = ax * ay * oc - az * s
        r02 = ax * az * oc + ay * s
        r10 = ay * ax * oc + az * s
        r11 = c + ay * ay * oc
        r12 = ay * az * oc - ax * s
        r20 = az * ax * oc - ay * s
        r21 = az * ay * oc + ax * s
        r22 = c + az * az * oc
        # T_left[i] entries.
        Tl = _JOINT_T_LEFTS[i]
        l00 = float(Tl[0, 0])
        l01 = float(Tl[0, 1])
        l02 = float(Tl[0, 2])
        l03 = float(Tl[0, 3])
        l10 = float(Tl[1, 0])
        l11 = float(Tl[1, 1])
        l12 = float(Tl[1, 2])
        l13 = float(Tl[1, 3])
        l20 = float(Tl[2, 0])
        l21 = float(Tl[2, 1])
        l22 = float(Tl[2, 2])
        l23 = float(Tl[2, 3])
        # M = T_left[i] @ R (R is the homogeneous version of the 3x3
        # rotation above with column 3 = [0,0,0,1]^T).
        m00 = l00 * r00 + l01 * r10 + l02 * r20
        m01 = l00 * r01 + l01 * r11 + l02 * r21
        m02 = l00 * r02 + l01 * r12 + l02 * r22
        m03 = l03
        m10 = l10 * r00 + l11 * r10 + l12 * r20
        m11 = l10 * r01 + l11 * r11 + l12 * r21
        m12 = l10 * r02 + l11 * r12 + l12 * r22
        m13 = l13
        m20 = l20 * r00 + l21 * r10 + l22 * r20
        m21 = l20 * r01 + l21 * r11 + l22 * r21
        m22 = l20 * r02 + l21 * r12 + l22 * r22
        m23 = l23
        # T_right[i] entries.
        Tr = _JOINT_T_RIGHTS[i]
        t00 = float(Tr[0, 0])
        t01 = float(Tr[0, 1])
        t02 = float(Tr[0, 2])
        t03 = float(Tr[0, 3])
        t10 = float(Tr[1, 0])
        t11 = float(Tr[1, 1])
        t12 = float(Tr[1, 2])
        t13 = float(Tr[1, 3])
        t20 = float(Tr[2, 0])
        t21 = float(Tr[2, 1])
        t22 = float(Tr[2, 2])
        t23 = float(Tr[2, 3])
        # N = M @ T_right[i]
        n00 = m00 * t00 + m01 * t10 + m02 * t20
        n01 = m00 * t01 + m01 * t11 + m02 * t21
        n02 = m00 * t02 + m01 * t12 + m02 * t22
        n03 = m00 * t03 + m01 * t13 + m02 * t23 + m03
        n10 = m10 * t00 + m11 * t10 + m12 * t20
        n11 = m10 * t01 + m11 * t11 + m12 * t21
        n12 = m10 * t02 + m11 * t12 + m12 * t22
        n13 = m10 * t03 + m11 * t13 + m12 * t23 + m13
        n20 = m20 * t00 + m21 * t10 + m22 * t20
        n21 = m20 * t01 + m21 * t11 + m22 * t21
        n22 = m20 * t02 + m21 * t12 + m22 * t22
        n23 = m20 * t03 + m21 * t13 + m22 * t23 + m23
        # T_acc = T_acc @ N
        b00 = a00 * n00 + a01 * n10 + a02 * n20
        b01 = a00 * n01 + a01 * n11 + a02 * n21
        b02 = a00 * n02 + a01 * n12 + a02 * n22
        b03 = a00 * n03 + a01 * n13 + a02 * n23 + a03
        b10 = a10 * n00 + a11 * n10 + a12 * n20
        b11 = a10 * n01 + a11 * n11 + a12 * n21
        b12 = a10 * n02 + a11 * n12 + a12 * n22
        b13 = a10 * n03 + a11 * n13 + a12 * n23 + a13
        b20 = a20 * n00 + a21 * n10 + a22 * n20
        b21 = a20 * n01 + a21 * n11 + a22 * n21
        b22 = a20 * n02 + a21 * n12 + a22 * n22
        b23 = a20 * n03 + a21 * n13 + a22 * n23 + a23
        a00, a01, a02, a03 = b00, b01, b02, b03
        a10, a11, a12, a13 = b10, b11, b12, b13
        a20, a21, a22, a23 = b20, b21, b22, b23
    return np.array(
        [
            [a00, a01, a02, a03],
            [a10, a11, a12, a13],
            [a20, a21, a22, a23],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@cython.ccall
@cython.locals(i=cython.int, n=cython.int)
def _spatial_jacobian(q):
    """6 x n_dof spatial Jacobian using the baked chain constants.

    Math identical to ssik.refinement.kinbody_jacobian: column i
    is (p_i x z_i, z_i) where z_i is the i-th joint axis in the
    world frame at q and p_i is the i-th joint origin. This is
    the SPATIAL twist representation -- T(q+dq) @ T(q)^-1 ~
    exp([J @ dq]) -- matching the residual extracted by
    ssik.refinement.se3_log_residual. Per-arm version with
    baked _JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS so
    there's no KinBody walk at runtime.
    """
    n = len(_JOINT_AXES)
    cum = _FK_EYE4.copy()
    cums = [cum.copy()]
    rot = _FK_EYE4.copy()
    for i in range(n):
        rot[:3, :3] = _rotation_matrix(_JOINT_AXES[i], float(q[i]))
        cum = cum @ _JOINT_T_LEFTS[i] @ rot @ _JOINT_T_RIGHTS[i]
        cums.append(cum.copy())
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        t_pre = cums[i] @ _JOINT_T_LEFTS[i]
        axis_unit = _JOINT_AXES[i] / np.linalg.norm(_JOINT_AXES[i])
        z_i = t_pre[:3, :3] @ axis_unit
        p_i = t_pre[:3, 3]
        # Scalar cross product p_i x z_i: bit-identical to np.cross but
        # ~11x faster for 3-vectors (np.cross's moveaxis / axis-normalize
        # overhead dominates the LM-refine loop). Mirrors the live
        # kinbody_jacobian, which is already scalar-inlined.
        J[0, i] = p_i[1] * z_i[2] - p_i[2] * z_i[1]
        J[1, i] = p_i[2] * z_i[0] - p_i[0] * z_i[2]
        J[2, i] = p_i[0] * z_i[1] - p_i[1] * z_i[0]
        J[3:, i] = z_i
    return J


@cython.ccall
def _wrap_to_pi(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``. Called inside the per-IK
    dedup hot loop (235k+ times on Franka 7R)."""
    return ((a + math.pi) % _TWO_PI) - math.pi


@cython.ccall
@cython.locals(
    i=cython.int,
    n=cython.int,
    diff=cython.double,
    ai=cython.double,
    bi=cython.double,
)
def _q_close_wrap(a, b, tol: float) -> bool:
    """Return ``True`` if joint vectors ``a`` and ``b`` agree (mod 2pi)
    within ``tol`` per element. Replaces the
    ``np.array([_wrap_to_pi(...)]) -> np.all(np.abs(...) < tol)``
    pipeline that allocated a numpy array per dedup-loop iteration --
    a per-element scalar loop avoids the array creation and the
    ``np.all`` reduction overhead, which together dominated the
    artifact's ``solve()`` body at the per-IK level."""
    n = len(a)
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        diff = ((ai - bi + math.pi) % _TWO_PI) - math.pi
        if abs(diff) > tol:
            return False
    return True


def solve(
    T_target,
    *,
    max_solutions: int | None = None,
    q_seed=None,
    respect_limits: bool | str = True,
    allow_refinement: bool = False,
    allow_rescue: bool = True,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    refinement_max_iters: int = 15,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    enumerate_windings: bool = True,
    native: bool = True,
):
    """Inverse kinematics. Returns ``list[Solution]``.

    :param T_target: 4x4 SE(3) target end-effector pose.
    :param max_solutions: optional cap on returned IKs (post-dedup,
        post-limits filter). ``None`` = full redundancy enumeration.
        Combine with ``q_seed`` for the "give me the IK closest to
        where I am now" trajectory-tracking idiom.
    :param q_seed: optional joint configuration. When provided,
        returned solutions are sorted by distance from ``q_seed``
        (closest first, via ``seed_metric``); with ``max_solutions``
        this returns the nearest ``max_solutions`` to the seed -- the
        trajectory-tracking idiom.
    :param seed_metric: distance used to rank against ``q_seed``.
        ``"wrap_linf"`` (default) minimises the *largest* single-joint
        wrap-to-pi move, which holds the branch during tracking;
        ``"wrap_l2"`` minimises the summed move (can favour a flip
        "paid for" by smaller moves elsewhere). Ignored when
        ``q_seed`` is ``None``.
    :param seed_tolerance: optional max per-joint deviation from
        ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
        *every* joint within ``seed_tolerance`` are returned -- a hard
        tracking guarantee that may return an empty list when no branch
        qualifies. ``None`` (default) keeps the best-effort behaviour.
        Requires ``q_seed``.
    :param respect_limits: when ``True`` (default), solutions
        outside URDF joint limits are dropped. Pass ``False`` for
        the raw geometric set (e.g. analysis / debugging), or
        ``"wrap"`` for the full set wrapped into the joint ranges
        with nothing dropped.
    :param allow_refinement: opt into Newton polish for near-miss
        algebraic candidates that don't quite meet ``fk_atol``.
        Default ``False`` -- the algebraic path is already at
        machine precision on tier-0 / SRS arms. On tier-2 RR
        arms (JACO 2, Rizon 4, Kassow), polish can recover
        edge-case candidates whose algebraic FK drifts above
        ``fk_atol``, at ~100-300 us per polished branch.
    :param allow_rescue: when ``True`` (default), if the analytical
        path returns no solutions for a target within the arm's
        reach (a measure-zero rank-deficient RR ridge -- a
        reachable pose the algebraic path can't extract),
        ``solve()`` recovers the IK via the T-perturbation rescue
        (#319), returning machine-precision solutions tagged
        ``refinement_used="lm"``. Set ``False`` for a guaranteed
        purely-analytical result (returns ``[]`` at such ridges).
        Gated by a reach-sphere, so far-field unreachable targets
        stay cheap.
    :param policy: tolerance policy (FK closure + dedup tolerance).
        Rarely customised.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    :param native: use the shipped native (C++) backend for this arm's
        solver family (the default; typically 2-100x faster). Returns the
        same solution *set*; the *order* without a seed and the
        near-singular *representative* may differ (numpy vs Eigen), and
        redundant-7R arms may sample the self-motion manifold differently.
        Silently falls back to the Python path when the native extension
        isn't bundled (Windows / source installs), and when ``policy`` is
        not equal to ``DEFAULT_TOLERANCE_POLICY`` or ``allow_refinement``
        is ``True``, which native does not implement. Pass ``native=False``
        for the pure-Python path (identical algorithm, no C++ dependency).
        Default ``True``.
    :returns: list of :class:`Solution`; empty list iff no IK
        closed within ``policy.subproblem_numerical`` (or all
        IKs were filtered by ``respect_limits=True``).
    """
    T_target, q_seed, max_solutions = _check_solve_inputs(
        T_target,
        q_seed,
        DOF,
        max_solutions=max_solutions,
        seed_tolerance=seed_tolerance,
        refinement_max_iters=refinement_max_iters,
        policy=policy,
    )
    if max_solutions == 0:  # at most 0 solutions: checked, nothing to solve
        return []
    if native:
        _native_sols = _try_native_solve(
            SOLVER_NAME,
            _KB,
            T_target,
            respect_limits=respect_limits,
            q_seed=q_seed,
            seed_metric=seed_metric,
            seed_tolerance=seed_tolerance,
            max_solutions=max_solutions,
            allow_rescue=allow_rescue,
            enumerate_windings=enumerate_windings,
            refinement_max_iters=refinement_max_iters,
            policy=policy,
            allow_refinement=allow_refinement,
            rr_geometry=_rr_native_geometry(),
        )
        if _native_sols is not None:
            return _native_sols
    T = np.asarray(T_target, dtype=np.float64)
    candidates = _solve_algebraic(T)
    # Per candidate, whether it may sample a singular continuum
    # (ssik.continuum): set from its Jacobian below.
    _free = [_NOT_FLAGGED] * len(candidates)

    fk_atol = policy.subproblem_numerical
    dedup_atol = policy.subproblem_dedup

    # Three-bucket sort: exact (closes within fk_atol), near-miss
    # (refinable when allow_refinement=True), or drop.
    # The Jacobians the near-miss refinement evaluates, the last of which
    # flags a refined candidate (ssik.continuum.has_null).
    _lm_jac: list[np.ndarray] = []

    def _lm_jacobian(x: np.ndarray) -> np.ndarray:
        j = _spatial_jacobian(x)
        _lm_jac.append(j)
        return j

    verified: list[tuple[np.ndarray, float, str, int, int]] = []
    for _ci, cand_q in enumerate(candidates):
        q = np.asarray(cand_q, dtype=np.float64)
        if not np.all(np.isfinite(q)):
            continue
        T_check = _fk(q)
        residual = float(np.linalg.norm(T_check - T))
        if residual <= fk_atol:
            verified.append((q, residual, "none", 0, _free[_ci]))
            continue
        if not (allow_refinement or True):
            continue
        # Refine only NEAR-misses. An algebraic candidate whose FK is
        # already >0.1 off (Frobenius) is an eigensolve root that does not
        # correspond to a real IK solution here (genuine near-double-root
        # marginals sit at ~1e-4); polishing it just burns Newton iters
        # that stall out and add nothing (#490: force_refine was 5-10x on
        # degenerate arms doing exactly this). 0.1 is 100x looser than the
        # 1e-3 band that once dropped genuine piper solutions -- it only
        # skips candidates clearly not near any solution.
        if residual >= 0.1:
            continue
        # Newton polish using the per-arm spatial Jacobian.
        _lm_jac.clear()
        refined = _lm_refine(
            q,
            _fk,
            T,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=_lm_jacobian,
        )
        if refined is None:
            continue
        q_ref, resid_ref, iters = refined
        # A refined near-miss is flagged from the last Jacobian its
        # refinement evaluated (one step from q_ref).
        _f = (
            _FREE_FROM_KERNEL if not _lm_jac or _has_null(_lm_jac[-1]) else _NOT_FLAGGED
        )
        verified.append((q_ref, resid_ref, "lm", iters, _f))

    # Polish every accepted candidate to machine precision on its own
    # branch, before the dedup sees it (ssik.refinement.polish).
    _accepted = [i for i, v in enumerate(verified) if v[2] == "none"]
    if _accepted:
        _q_pol, _r_pol, _polished, _singular = _polish_flagged(
            np.array([verified[i][0] for i in _accepted]),
            T,
            _POLISH_CHAIN,
        )
        for _k, _i in enumerate(_accepted):
            _f = _FREE_FROM_KERNEL if _singular[_k] else _NOT_FLAGGED
            if _polished[_k]:
                verified[_i] = (_q_pol[_k], float(_r_pol[_k]), "none", 0, _f)
            else:
                verified[_i] = (*verified[_i][:4], _f)

    # Same-root dedup (#600), mirroring ``ssik.refinement.dedup_same_root``:
    # a pair within ``dedup_atol`` merges only if its midpoint also
    # closes FK, so twin roots near a fold both survive. The survivor
    # changes only on a residual gain above round-off. ``_q_close_wrap``
    # is the cheap pre-filter (typed scalar loop, #137 Slice 3).
    _floor = _same_root_floor(T)
    deduped: list[tuple[np.ndarray, float, str, int, int]] = []
    for cand_q, cand_res, ref_used, ref_iters, cand_free in verified:
        dup_idx = None
        for j, (existing_q, existing_res, _, _, _) in enumerate(deduped):
            if _q_close_wrap(cand_q, existing_q, dedup_atol) and _is_same_root(
                cand_q, cand_res, existing_q, existing_res, _fk, T, _floor
            ):
                dup_idx = j
                break
        if dup_idx is None:
            deduped.append((cand_q, cand_res, ref_used, ref_iters, cand_free))
            continue
        # A merge keeps the flag either side had.
        _kept = deduped[dup_idx]
        _f = _kept[4] if _kept[4] != _NOT_FLAGGED else cand_free
        if cand_res < _kept[1] - _floor:
            deduped[dup_idx] = (cand_q, cand_res, ref_used, ref_iters, _f)
        elif _f != _kept[4]:
            deduped[dup_idx] = (*_kept[:4], _f)

    solutions = [
        Solution(
            q=q,
            fk_residual=residual,
            refinement_used=ref_used,
        )
        for q, residual, ref_used, _ref_iters, _f in deduped
    ]
    # The singular-continuum slide (ssik.continuum, docs/api.md#singular-
    # continua): each flagged solution moves to its continuum's point under
    # the rule, seeded or not. Only flagged solutions pay.
    solutions = _slide_continua(
        solutions,
        [_f for _q, _r, _u, _i, _f in deduped],
        _KB,
        T,
        fk=_fk,
        jac=_spatial_jacobian,
        q_seed=q_seed,
        respect_limits=respect_limits,
        dedup_atol=dedup_atol,
    )

    # Bulletproof fallback (#319): the analytical path found nothing.
    # If the target is within the arm's max reach it may be a
    # measure-zero rank-deficient ridge (a reachable pose the algebraic
    # path can't extract) rather than an unreachable target -- recover
    # via the T-perturbation rescue. The reach-sphere (sum of link
    # lengths; an exact upper bound by the triangle inequality, so it
    # never rejects a reachable pose) is the gate: it is checked only
    # here in the rare empty branch and keeps genuinely far-field
    # targets cheap. (The RR real-root count is NOT used as a gate -- it
    # is an unreliable reachability signal: some reachable ridges, e.g.
    # Rizon 4's, yield only complex roots, so gating on it would
    # silently drop real solutions.) The perturbed re-solves run with
    # allow_rescue=False (recursion guard + analytical-only escape
    # hatch). Rescued sols carry refinement_used="lm", FK-gated to
    # machine precision.
    # Shared post-processing pipeline (limits -> seed -> truncate); the
    # one definition lives in ssik.postprocess.finalize_solutions.
    # Limit pass only (no seed/tolerance/truncate yet): the rescue gate is
    # "no in-limits solution exists", so it must not depend on the
    # seed-tolerance / max_solutions filters (a seed filter emptying a
    # non-empty in-limits set is a user preference, not a missing solution).
    _in_limits = _ps_finalize(solutions, _KB, respect_limits=respect_limits, T_target=T)
    # Rescue on LIMIT-empty (#524): fire when nothing survives the limit
    # filter (not just when the analytical count was zero), so a pose
    # whose only analytical candidates were out-of-limits still gets
    # rescued. Perturbed re-solves run with allow_rescue=False.
    if not _in_limits and allow_rescue:
        _reach_radius = sum(
            float(np.linalg.norm(np.asarray(_t)[:3, 3]))
            for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
        )
        if float(np.linalg.norm(T[:3, 3])) <= _reach_radius:
            _in_limits = _ps_finalize(
                _rescue_via_T_perturbation(
                    _fk,
                    _functools.partial(solve, allow_rescue=False),
                    T,
                    jacobian_fn=_spatial_jacobian,
                ),
                _KB,
                T_target=T,
                respect_limits=respect_limits,
            )
    # Seed tolerance / ranking / truncate over the in-limits set.
    return _ps_finalize(
        _in_limits,
        _KB,
        T_target=T,
        respect_limits=False,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
        # The single lifting stage (#562): the call that yields the
        # returned set. Skipped when the caller asked for raw geometry.
        enumerate_windings=enumerate_windings and respect_limits,
    )


def fk(q):
    """Forward kinematics: the 4x4 base->ee pose at ``q``, shape ``(DOF,)``."""
    return _fk(_check_joints(q, DOF, "q", finite=False))


__all__ = [
    "BASE_LINK",
    "DISPATCH_REASON",
    "DOF",
    "EE_LINK",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "T_HOME",
    "fk",
    "solve",
]
