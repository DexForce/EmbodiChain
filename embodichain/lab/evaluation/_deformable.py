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

"""Canonical identity for mesh-bound particle-centroid observations."""

from __future__ import annotations

__all__: list[str] = []


def _centroid_field(uid: str, digest: str, indices: tuple[int, ...]) -> str:
    field = f"deformables/{uid}/{digest}/{','.join(map(str, indices))}/centroid"
    _centroid_selection(field)
    return field


def _centroid_selection(field: str) -> tuple[str, str, tuple[int, ...]]:
    parts = field.split("/")
    if len(parts) != 5 or parts[0] != "deformables" or parts[-1] != "centroid":
        raise ValueError("Invalid deformable centroid field.")
    _, uid, digest, selection, _ = parts
    if (
        not uid
        or uid != uid.strip()
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Centroid evidence requires an exact UID and mesh SHA-256.")
    try:
        indices = tuple(int(value) for value in selection.split(","))
    except ValueError as exc:
        raise ValueError("Invalid centroid particle selection.") from exc
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or indices[0] < 0
        or selection != ",".join(map(str, indices))
    ):
        raise ValueError("Centroid particle indices must be canonical and unique.")
    return uid, digest, indices
