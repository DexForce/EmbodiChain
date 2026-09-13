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
"""Backend-specific articulation topology accessors."""

from __future__ import annotations

__all__ = ["get_joint_descriptor"]


def get_joint_descriptor(
    entity: object,
    joint_name: str,
    *,
    is_newton: bool,
) -> object:
    """Read one joint descriptor through the active backend contract.

    Newton exposes backend-neutral joint descriptors directly. Default first
    uses its legacy joint-info API and falls back to the descriptor API when
    that legacy query is unavailable.
    """
    if is_newton:
        return _required_descriptor(entity, joint_name)

    get_joint_info = getattr(entity, "get_joint_info", None)
    native = get_joint_info(joint_name) if callable(get_joint_info) else None
    if native is not None:
        return native
    return _required_descriptor(entity, joint_name)


def _required_descriptor(entity: object, joint_name: str) -> object:
    get_joint_desc = getattr(entity, "get_joint_desc", None)
    if not callable(get_joint_desc):
        raise ValueError(
            "Native articulation has no joint topology for " f"{joint_name!r}."
        )
    try:
        return get_joint_desc(joint_name)
    except (KeyError, StopIteration) as exc:
        raise ValueError(
            "Native articulation has no joint topology for " f"{joint_name!r}."
        ) from exc
