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

__all__: list[str] = []

# Shared with the runtime config without importing the simulation package.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_DEFAULT_SCENE_FPS = 15.0
_DEFAULT_IMAGE_FPS = 2.0
_DEFAULT_SOFT_BODY_FPS = 5.0
_DEFAULT_ENV_IDS = (0,)


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
        default=_DEFAULT_HOST,
        help="Viser bind host.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=_DEFAULT_PORT,
        help="Viser bind port.",
    )
    parser.add_argument(
        "--viser-fps",
        type=float,
        default=_DEFAULT_SCENE_FPS,
        help="Maximum Viser scene update rate.",
    )
    parser.add_argument(
        "--viser-image-fps",
        type=float,
        default=_DEFAULT_IMAGE_FPS,
        help=(
            "Maximum Viser camera RGB preview rate. run-env synchronizes once "
            "per environment step when this option is omitted."
        ),
    )
    parser.add_argument(
        "--viser-soft-body-fps",
        type=float,
        default=_DEFAULT_SOFT_BODY_FPS,
        help="Maximum Viser soft-body and cloth mesh update rate.",
    )
    parser.add_argument(
        "--viser-env-ids",
        type=_parse_viser_env_id,
        nargs="+",
        default=(["all"] if _DEFAULT_ENV_IDS is None else list(_DEFAULT_ENV_IDS)),
        help="Environment IDs published to Viser, or 'all'.",
    )
