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

"""Resolve camera attachment targets from an explicit scene asset mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexsim.engine import Node

    from embodichain.lab.sim.objects import Articulation

__all__ = ["resolve_parent_nodes"]


def resolve_parent_nodes(
    parent: str, assets: Mapping[str, Articulation], num_envs: int
) -> list[Node]:
    """Resolve a canonical Robot or Articulation link in each environment.

    Args:
        parent: A canonical link name, or ``"<asset_uid>/<link_name>"`` to
            disambiguate links shared by multiple assets.
        assets: Scene asset UIDs mapped to Articulation or Robot instances.
        num_envs: Expected number of camera and asset instances.

    Returns:
        Parent render nodes ordered by environment index.

    Raises:
        ValueError: If the parent link is missing or ambiguous.
        RuntimeError: If the asset count differs from ``num_envs``, or an
            environment is missing the link or its render node.
    """
    asset_uid: str | None = None
    link_name = parent
    if "/" in parent:
        candidate_uid, candidate_link = parent.split("/", maxsplit=1)
        if candidate_uid in assets:
            asset_uid, link_name = candidate_uid, candidate_link

    matches: list[tuple[str, list[Node]]] = []
    for uid, asset in assets.items():
        if asset_uid is not None and uid != asset_uid:
            continue
        if link_name not in asset.link_names:
            continue
        if asset.num_instances != num_envs:
            raise RuntimeError(
                f"Camera parent asset {uid!r} has {asset.num_instances} instances "
                f"for {num_envs} arenas."
            )
        matches.append((uid, asset.get_link_render_nodes(link_name)))

    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        owners = ", ".join(uid for uid, _ in matches)
        raise ValueError(
            f"Camera parent link {link_name!r} is ambiguous across assets "
            f"[{owners}]; use '<asset_uid>/{link_name}'."
        )
    scope = f" on asset {asset_uid!r}" if asset_uid is not None else ""
    raise ValueError(
        f"Camera parent link {link_name!r} was not found{scope} in any "
        "registered Robot or Articulation."
    )
