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

from .cfg import VisualizationCfg, ViserServerCfg
from embodichain.cli._visualization import add_viser_args_to_parser

__all__ = ["add_viser_args_to_parser", "visualization_cfg_from_args"]


def visualization_cfg_from_args(
    args: argparse.Namespace,
) -> VisualizationCfg:
    """Build visualization configuration from parsed CLI arguments.

    Args:
        args: Namespace populated by :func:`add_viser_args_to_parser`.

    Returns:
        Visualization configuration including Viser server settings.
    """
    defaults = VisualizationCfg()
    server_defaults = defaults.viser_server
    enabled = bool(getattr(args, "viser", False))
    image_fps_arg = getattr(args, "viser_image_fps", defaults.sensor_image_fps)
    env_ids_arg = list(
        getattr(
            args,
            "viser_env_ids",
            ["all"] if defaults.env_ids is None else defaults.env_ids,
        )
    )
    if "all" in env_ids_arg:
        if env_ids_arg != ["all"]:
            raise ValueError("'all' cannot be combined with explicit Viser env IDs.")
        env_ids = None
    else:
        env_ids = [int(env_id) for env_id in env_ids_arg]
    visualization_cfg = VisualizationCfg(
        backend="viser" if enabled else "none",
        scene_fps=float(getattr(args, "viser_fps", defaults.scene_fps)),
        sensor_image_fps=(None if image_fps_arg is None else float(image_fps_arg)),
        soft_body_fps=float(
            getattr(args, "viser_soft_body_fps", defaults.soft_body_fps)
        ),
        env_ids=env_ids,
        allow_commands=enabled,
        viser_server=ViserServerCfg(
            host=str(getattr(args, "viser_host", server_defaults.host)),
            port=int(getattr(args, "viser_port", server_defaults.port)),
        ),
    )
    return visualization_cfg
