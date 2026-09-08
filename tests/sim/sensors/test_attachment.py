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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodichain.lab.sim.sensors.attachment import resolve_parent_nodes

pytestmark = pytest.mark.no_sim


def _make_asset(
    num_envs: int = 2, link_name: str = "wrist"
) -> tuple[SimpleNamespace, list[object]]:
    """Only expose public asset queries, without manager or native handles."""
    nodes = [object() for _ in range(num_envs)]
    return (
        SimpleNamespace(
            link_names=[link_name],
            num_instances=num_envs,
            get_link_render_nodes=MagicMock(return_value=nodes),
        ),
        nodes,
    )


@pytest.mark.parametrize("num_envs", [1, 3])
def test_resolve_parent_nodes_preserves_environment_order(num_envs: int) -> None:
    asset, nodes = _make_asset(num_envs)

    assert resolve_parent_nodes("wrist", {"arm": asset}, num_envs) == nodes
    asset.get_link_render_nodes.assert_called_once_with("wrist")


def test_resolve_parent_nodes_requires_asset_uid_for_ambiguous_links() -> None:
    arm, arm_nodes = _make_asset()
    tool, tool_nodes = _make_asset()
    assets = {"arm": arm, "tool": tool}

    with pytest.raises(ValueError, match="ambiguous.*asset_uid"):
        resolve_parent_nodes("wrist", assets, 2)

    arm.get_link_render_nodes.reset_mock()
    tool.get_link_render_nodes.reset_mock()
    assert resolve_parent_nodes("arm/wrist", assets, 2) == arm_nodes
    arm.get_link_render_nodes.assert_called_once_with("wrist")
    tool.get_link_render_nodes.assert_not_called()
    assert resolve_parent_nodes("tool/wrist", assets, 2) == tool_nodes


def test_resolve_parent_nodes_preserves_slashes_in_canonical_link_names() -> None:
    asset, nodes = _make_asset(link_name="tool/wrist")

    assert resolve_parent_nodes("tool/wrist", {"arm": asset}, 2) == nodes
    assert resolve_parent_nodes("arm/tool/wrist", {"arm": asset}, 2) == nodes


@pytest.mark.parametrize("parent", ["missing", "arm/missing", "unknown/wrist"])
def test_resolve_parent_nodes_rejects_unknown_links(parent: str) -> None:
    asset, _ = _make_asset()

    with pytest.raises(ValueError, match="was not found"):
        resolve_parent_nodes(parent, {"arm": asset}, 2)

    asset.get_link_render_nodes.assert_not_called()


def test_resolve_parent_nodes_rejects_empty_asset_mapping() -> None:
    with pytest.raises(ValueError, match="was not found"):
        resolve_parent_nodes("wrist", {}, 2)


def test_resolve_parent_nodes_rejects_instance_count_mismatch() -> None:
    asset, _ = _make_asset(num_envs=1)

    with pytest.raises(RuntimeError, match="1 instances for 2 arenas"):
        resolve_parent_nodes("wrist", {"arm": asset}, 2)

    asset.get_link_render_nodes.assert_not_called()


def test_resolve_parent_nodes_propagates_asset_query_failure() -> None:
    asset, _ = _make_asset()
    error = RuntimeError("missing render node in arena 1")
    asset.get_link_render_nodes.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        resolve_parent_nodes("wrist", {"arm": asset}, 2)

    assert raised.value is error
