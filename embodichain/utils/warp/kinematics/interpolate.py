# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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


@wp.kernel
def pairwise_distances(
    points: wp.array(dtype=float),  # flattened: length B*N*M
    distances: wp.array(dtype=float),  # flattened: length B*(N-1)
    B: int,
    N: int,
    M: int,
):
    """Compute Euclidean distances between consecutive points along N using 1D flattened storage.

    Memory layout (row-major):
        points(b, i, j)  => b*N*M + i*M + j
        distances(b, i)  => b*(N-1) + i
    Result:
        distances[b,i] = ||points[b,i+1,:] - points[b,i,:]||_2
    """
    tid = wp.tid()
    total = B * (N - 1)
    if tid >= total:
        return

    b = tid // (N - 1)
    i = tid - b * (N - 1)

    base_points = b * N * M
    s = float(0.0)
    for j in range(M):
        p0 = points[base_points + i * M + j]
        p1 = points[base_points + (i + 1) * M + j]
        d = p1 - p0
        s = s + d * d
    distances[b * (N - 1) + i] = wp.sqrt(s)


@wp.kernel
def cumsum_distances(
    distances: wp.array(dtype=float),  # flattened: length B*(N-1)
    cumulative: wp.array(dtype=float),  # flattened: length B*N
    B: int,
    N: int,
):
    """Compute per-batch cumulative distances with flattened indexing.

    Layout:
        distances(b,i)  => b*(N-1) + i
        cumulative(b,i) => b*N + i
    Definition:
        cumulative[b,0] = 0
        cumulative[b,i] = sum_{k=0}^{i-1} distances[b,k]
    """
    b = wp.tid()
    if b >= B:
        return

    cumulative[b * N + 0] = float(0.0)
    acc = float(0.0)
    for i in range(N - 1):
        acc = acc + distances[b * (N - 1) + i]
        cumulative[b * N + (i + 1)] = acc


@wp.kernel
def repeat_first_point(
    points: wp.array(dtype=float),  # flattened: length B*N*M (N may be 1)
    out: wp.array(dtype=float),  # flattened: length B*T*M
    B: int,
    T: int,
    M: int,
    N: int,
):
    """Repeat the first waypoint of each batch across T samples (used when N==1).

    First point (b,j):  b*N*M + j (i=0)
    Output (b,t,j):    b*T*M + t*M + j
    """
    tid = wp.tid()
    total = B * T
    if tid >= total:
        return

    b = tid // T
    t = tid - b * T

    base_in = b * N * M  # N expected 1 in usage
    base_out = b * T * M + t * M
    for j in range(M):
        out[base_out + j] = points[base_in + j]


@wp.kernel
def interpolate_along_distance(
    points: wp.array(dtype=float),  # flattened B*N*M
    cumulative: wp.array(dtype=float),  # flattened B*N
    out: wp.array(dtype=float),  # flattened B*T*M
    B: int,
    N: int,
    M: int,
    T: int,
):
    """Piecewise-linear interpolation at uniformly spaced cumulative-distance samples.

    Indexing (flattened):
        points(b,i,j)   => b*N*M + i*M + j
        cumulative(b,i) => b*N + i
        out(b,t,j)      => b*T*M + t*M + j
    Steps:
        1. Compute target distance new_d in [0, total_len].
        2. Binary search cumulative to find segment [lo, hi].
        3. Linear interpolate each dimension.
    """
    tid = wp.tid()
    total_threads = B * T
    if tid >= total_threads:
        return

    b = tid // T
    t = tid - b * T

    # total path length for batch b
    total_len = cumulative[b * N + (N - 1)]

    # evenly spaced target distance
    new_d = float(0.0)
    if T > 1:
        new_d = total_len * float(t) / float(T - 1)
    else:
        new_d = float(0.0)

    # binary search for segment boundaries
    lo = int(0)
    hi = N - 1
    while (lo + 1) < hi:
        mid = (lo + hi) // 2
        if cumulative[b * N + mid] <= new_d:
            lo = mid
        else:
            hi = mid

    c_lo = cumulative[b * N + lo]
    c_hi = cumulative[b * N + hi]
    denom = c_hi - c_lo

    alpha = float(0.0)
    if denom > float(0.0):
        alpha = (new_d - c_lo) / denom

    base_points = b * N * M
    base_out = b * T * M + t * M
    p_lo_offset = base_points + lo * M
    p_hi_offset = base_points + hi * M
    for j in range(M):
        p_lo = points[p_lo_offset + j]
        p_hi = points[p_hi_offset + j]
        out[base_out + j] = p_lo + alpha * (p_hi - p_lo)
