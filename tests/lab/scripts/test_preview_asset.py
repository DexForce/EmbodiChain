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

import argparse

from unittest.mock import Mock

from embodichain.lab.scripts.preview_asset import (
    _create_parser,
    _publish_loaded_assets,
    _run_preview_mode,
    build_sim_cfg,
)

ASSET_PATH = "asset.usda"
VISER_HOST = "0.0.0.0"
VISER_PORT = 9090


def test_viser_arguments_enable_headless_browser_visualization() -> None:
    """The asset-preview CLI should forward Viser options to simulation config."""
    args = _create_parser().parse_args(
        [
            "--asset_path",
            ASSET_PATH,
            "--viser",
            "--viser-host",
            VISER_HOST,
            "--viser-port",
            str(VISER_PORT),
        ]
    )

    sim_cfg = build_sim_cfg(args)

    assert sim_cfg.headless is True
    assert sim_cfg.visualization.backend == "viser"
    assert sim_cfg.visualization.allow_commands is True
    assert sim_cfg.visualization.viser_server.host == VISER_HOST
    assert sim_cfg.visualization.viser_server.port == VISER_PORT


def test_loaded_assets_are_published_immediately_in_viser() -> None:
    """Assets added after manager construction should be captured before waiting."""
    sim = Mock()
    args = argparse.Namespace(viser=True)

    _publish_loaded_assets(sim, args)

    sim.capture_visualization_safely.assert_called_once_with(force=True)


def test_loaded_assets_are_not_published_without_viser() -> None:
    """Native and headless-only previews should not request a Viser capture."""
    sim = Mock()
    args = argparse.Namespace(viser=False)

    _publish_loaded_assets(sim, args)

    sim.capture_visualization_safely.assert_not_called()


def test_viser_preview_stays_alive_when_simulation_is_headless() -> None:
    """Viser should keep stepping even though its simulation window is headless."""
    sim = Mock()
    sim.update.side_effect = KeyboardInterrupt
    args = argparse.Namespace(preview=False, headless=True, viser=True)

    _run_preview_mode(sim, [], args)

    sim.update.assert_called_once_with(step=1)


def test_headless_check_exits_without_starting_update_loop() -> None:
    """Plain headless mode should preserve its one-shot validation behavior."""
    sim = Mock()
    args = argparse.Namespace(preview=False, headless=True, viser=False)

    _run_preview_mode(sim, [], args)

    sim.update.assert_not_called()
