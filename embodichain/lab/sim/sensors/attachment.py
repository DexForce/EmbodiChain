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

"""Resolve semantic sensor attachments and adapt them for camera render nodes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexsim.engine import Node

    from .base_sensor import SensorCfg

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
    _, asset, link = _resolve_link(parent, assets, num_envs)
    return asset.get_link_render_nodes(link)


def _resolve_link(parent: str, assets: Mapping[str, Articulation], num_envs: int):
    asset_uid, sep, qualified_link = parent.partition("/")
    if sep and asset_uid in assets:
        candidates = [(asset_uid, assets[asset_uid])]
        link = qualified_link
    else:
        candidates = assets.items()
        link = parent
    matches = []
    for uid, asset in candidates:
        if link not in asset.link_names:
            continue
        if asset.num_instances != num_envs:
            raise RuntimeError(
                f"Sensor parent asset {uid!r} has {asset.num_instances} instances "
                f"for {num_envs} arenas."
            )
        matches.append((uid, asset, link))
    if len(matches) == 1:
        return matches[0]
    if matches:
        owners = ", ".join(uid for uid, _, _ in matches)
        raise ValueError(
            f"Sensor parent link {link!r} is ambiguous across assets "
            f"[{owners}]; use '<asset_uid>/<link_name>'."
        )
    raise ValueError(
        f"Sensor parent link {parent!r} was not found in any registered Robot or Articulation."
    )


def _qualify_robot_mount(
    cfg: SensorCfg, robot_uid: str, used_uids: set[str]
) -> SensorCfg:
    from .camera import CameraCfg

    if not isinstance(cfg.uid, str) or not cfg.uid.strip():
        raise ValueError("Mounted sensors require a non-empty uid.")
    if cfg.uid in used_uids:
        raise ValueError(f"Sensor {cfg.uid!r} already exists.")
    offset = cfg.extrinsics if isinstance(cfg, CameraCfg) else cfg.mount
    parent = offset.parent if offset is not None else None
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("Mounted sensors require a parent link.")
    link = parent.removeprefix(f"{robot_uid}/")
    if not link or "/" in link:
        raise ValueError(
            f"Mounted sensor {cfg.uid!r} must refer to a local link on robot {robot_uid!r}."
        )
    bound = deepcopy(cfg)
    bound_offset = bound.extrinsics if isinstance(bound, CameraCfg) else bound.mount
    bound_offset.parent = f"{robot_uid}/{link}"
    used_uids.add(cfg.uid)
    return bound
