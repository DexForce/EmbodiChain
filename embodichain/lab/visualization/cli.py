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

__all__ = ["add_viser_args_to_parser", "visualization_cfg_from_args"]


def _parse_viser_env_id(value: str) -> int | str:
    """Parse one environment ID or the ``all`` selector."""
    if value.lower() == "all":
        return "all"
    try:
        env_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative environment ID or 'all', received {value!r}."
        ) from exc
    if env_id < 0:
        raise argparse.ArgumentTypeError("Environment IDs must be non-negative.")
    return env_id


def add_viser_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add the standard EmbodiChain Viser command-line options.

    Args:
        parser: Parser receiving the Viser options.
    """
    visualization_defaults = VisualizationCfg()
    server_defaults = visualization_defaults.viser_server
    parser.add_argument(
        "--viser",
        action="store_true",
        help=(
            "Enable the headless Viser browser scene; configured Gizmos are "
            "interactive. Only expose it to trusted clients."
        ),
    )
    parser.add_argument(
        "--viser-host",
        default=server_defaults.host,
        help="Viser bind host.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=server_defaults.port,
        help="Viser bind port.",
    )
    parser.add_argument(
        "--viser-fps",
        type=float,
        default=visualization_defaults.scene_fps,
        help="Maximum Viser scene update rate.",
    )
    parser.add_argument(
        "--viser-image-fps",
        type=float,
        default=visualization_defaults.sensor_image_fps,
        help=(
            "Maximum Viser camera RGB preview rate. run-env synchronizes once "
            "per environment step when this option is omitted."
        ),
    )
    parser.add_argument(
        "--viser-soft-body-fps",
        type=float,
        default=visualization_defaults.soft_body_fps,
        help="Maximum Viser soft-body and cloth mesh update rate.",
    )
    parser.add_argument(
        "--viser-env-ids",
        type=_parse_viser_env_id,
        nargs="+",
        default=(
            ["all"]
            if visualization_defaults.env_ids is None
            else list(visualization_defaults.env_ids)
        ),
        help="Environment IDs published to Viser, or 'all'.",
    )


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
