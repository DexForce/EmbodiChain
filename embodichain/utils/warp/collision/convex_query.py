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

import warp as wp


@wp.kernel(enable_backward=False)
def convex_signed_distance_kernel(
    query_points: wp.array(dtype=wp.float32, ndim=3),
    plane_equations: wp.array(dtype=wp.float32, ndim=3),
    plane_equation_counts: wp.array(dtype=wp.int32, ndim=1),
    signed_distances: wp.array(dtype=wp.float32, ndim=3),
):
    """
    Compute the signed distance from query points to convex hulls defined by plane equations.

    Args:
        query_points: [n_pose, n_point, 3] coordinates of query points.
        plane_equations: [n_convex, n_max_equation, 4] plane equations of convex hulls, where each plane equation is represented as (normal_x, normal_y, normal_z, offset).
        plane_equation_counts: [n_convex, ] number of valid plane equations for each convex hull.

    Returns:
        signed_distances: [n_pose, n_point, n_convex] output signed distances from query points to convex hulls. Should be initialized as +inf before calling this kernel.
    """
    pose_id, point_id, convex_id = wp.tid()
    n_equation = plane_equation_counts[convex_id]
    for i in range(n_equation):
        normal_x = plane_equations[convex_id, i, 0]
        normal_y = plane_equations[convex_id, i, 1]
        normal_z = plane_equations[convex_id, i, 2]
        offset = plane_equations[convex_id, i, 3]
        signed_distance = (
            query_points[pose_id, point_id, 0] * normal_x
            + query_points[pose_id, point_id, 1] * normal_y
            + query_points[pose_id, point_id, 2] * normal_z
            + offset
        )
        # should initialize as -inf
        signed_distances[pose_id, point_id, convex_id] = max(
            signed_distance, signed_distances[pose_id, point_id, convex_id]
        )
