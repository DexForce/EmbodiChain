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

"""Command-line interface for URDF convex decomposition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Generate convex collision meshes and an updated URDF.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain decompose-urdf",
        description="Generate convex collision meshes for a URDF.",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the source URDF.",
    )
    parser.add_argument(
        "--output_urdf_name",
        type=str,
        default="articulation_acd.urdf",
        help="Name of the generated URDF.",
    )
    parser.add_argument(
        "--max_convex_hull_num",
        type=int,
        default=8,
        help="Maximum number of convex hulls for each mesh.",
    )
    parser.add_argument(
        "--recompute_inertia",
        action="store_true",
        help="Recompute inertia after convex decomposition.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Scale the URDF by three per-axis factors.",
    )
    args = parser.parse_args(argv)

    from embodichain.toolkits.acd.urdf_modifider import (
        generate_urdf_collision_convexes,
    )

    generate_urdf_collision_convexes(
        args.urdf_path,
        args.output_urdf_name,
        max_convex_hull_num=args.max_convex_hull_num,
        recompute_inertia=args.recompute_inertia,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
