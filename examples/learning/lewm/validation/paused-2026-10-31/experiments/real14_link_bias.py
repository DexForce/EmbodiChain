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

"""Private COM bias-acceleration model for articulated solver velocities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from cube_point_kinematics import PointKinematics


class LinkBiasKinematics(PointKinematics):
    def frames(self, q, v):
        root = np.array(self.asset["root_pose"])
        z = np.zeros(3)
        frames = {self.root_name: (root[:3, :3], root[:3, 3], z, z, z, z)}
        for parent, child, offset, origin, axis, index in self.joints:
            r, p, lv, av, la, aa = frames[parent]
            arm = r @ offset
            world_axis = r @ origin @ axis
            speed = 0 if index is None else v[index]
            angle = 0 if index is None else q[index]
            frames[child] = (
                r @ origin @ Rotation.from_rotvec(axis * angle).as_matrix(),
                p + arm,
                lv + np.cross(av, arm),
                av + world_axis * speed,
                la + np.cross(aa, arm) + np.cross(av, np.cross(av, arm)),
                aa + np.cross(av, world_axis * speed),
            )
        return frames

    def com_states(self, q, v):
        frames = self.frames(q, v)
        states = {}
        for body in self.asset["bodies"]:
            r, p, lv, av, la, aa = frames[body["name"]]
            offset = r @ body["ipos"]
            states[body["name"]] = (
                p + offset,
                lv + np.cross(av, offset),
                av,
                la + np.cross(aa, offset) + np.cross(av, np.cross(av, offset)),
                aa,
            )
        return states

    def point_solver_bias(self, q, v, *, common_point=True):
        """Extra point velocity/h from integrated COM/angular bias acceleration.

        Native articulated free motion advances body spatial velocities with
        Coriolis acceleration before external constraint preparation. The point
        moment arm is evaluated at old transforms. D6 actor A uses anchor B.
        """
        frames = self.frames(q, v)
        centers = self.com_states(q, v)
        result = []
        for index, ends in self.points:
            (a, local_a, _), (b, local_b, _) = ends
            point_a = frames[a][1] + frames[a][0] @ local_a
            point_b = frames[b][1] + frames[b][0] @ local_b
            ca, _, _, bias_a, alpha_a = centers[a]
            cb, _, _, bias_b, alpha_b = centers[b]
            # B minus A, in world axes, at the common native force point.
            result.append(
                bias_b
                + np.cross(alpha_b, point_b - cb)
                - bias_a
                - np.cross(alpha_a, (point_b if common_point else point_a) - ca)
            )
        return result
