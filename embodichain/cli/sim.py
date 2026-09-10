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

"""Simulation CLI options independent of Gym configuration loading.

Seed registration is opt-in. Parsing never initializes a simulator or random
stream; callers explicitly apply the resolved seed to their owned generators.
"""

from __future__ import annotations

import argparse
import secrets

__all__ = ["add_sim_args_to_parser", "add_seed_arg_to_parser", "resolve_seed"]


def add_sim_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Register standalone simulation and visualization options.

    Args:
        parser: Parser receiving the options. Device omission preserves the
            backend default; the other standalone defaults are concrete.
    """
    parser.add_argument(
        "--num_envs",
        "--num-envs",
        help="Number of parallel environments; omission uses launcher/config defaults.",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device used by environment tensors and the selected physics "
        "backend, e.g. 'cpu' or 'cuda:0'. When omitted, the selected backend "
        "default is preserved unless this option is set.",
    )

    parser.add_argument(
        "--headless",
        help="Whether to perform the simulation in headless mode.",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--renderer",
        type=str,
        choices=["auto", "hybrid", "fast-rt", "rt"],
        default="auto",
        help="Renderer backend; omission preserves the launcher/config default.",
    )

    parser.add_argument(
        "--physics",
        type=str,
        choices=["default", "newton"],
        default="default",
        help="Physics backend. For Gym configs, this may only confirm the file-owned backend.",
    )

    parser.add_argument(
        "--arena_space",
        "--arena-space",
        help="The size of the arena space.",
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--gpu_id",
        "--gpu-id",
        help="The GPU ID to use for the simulation.",
        default=0,
        type=int,
    )

    from ._visualization import add_viser_args_to_parser

    add_viser_args_to_parser(parser)


def _parse_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed must be an integer.") from exc
    if not -1 <= seed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be -1 or in [0, 2**32).")
    return seed


def add_seed_arg_to_parser(
    parser: argparse.ArgumentParser,
    *,
    default: int | None = 0,
    scope: str = "random sampling",
) -> None:
    """Register one seed option for a caller that actually consumes it.

    Args:
        parser: Parser receiving the seed option.
        default: Seed when omitted; None preserves a Gym config's seed.
        scope: User-facing description of the random streams controlled.
    """
    if default is not None:
        _parse_seed(str(default))
    parser.add_argument(
        "--seed",
        type=_parse_seed,
        default=default,
        help=f"Seed for {scope}; -1 selects a random seed. "
        + (
            "Omission preserves the task config."
            if default is None
            else f"Default: {default}."
        ),
    )


def resolve_seed(seed: int) -> int:
    """Resolve the random-run sentinel without changing global RNG state.

    Args:
        seed: A non-negative 32-bit seed, or -1 to generate a fresh seed.

    Returns:
        Effective seed to log and pass to the caller's random generators.

    Raises:
        ValueError: If seed is outside the portable 32-bit range.
    """
    if not -1 <= seed < 2**32:
        raise ValueError("Seed must be -1 or in [0, 2**32).")
    return secrets.randbits(32) if seed == -1 else seed
