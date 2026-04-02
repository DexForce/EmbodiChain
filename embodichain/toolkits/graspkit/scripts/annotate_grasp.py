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

"""CLI for interactive grasp region annotation on a mesh.

Loads a mesh file via *trimesh*, launches a browser-based annotator so the
user can select the graspable region, and saves the resulting antipodal
point pairs to the grasp-annotator cache.

Usage examples::

    python -m embodichain annotate-grasp --mesh_path /path/to/object.ply
    python -m embodichain annotate-grasp --mesh_path mug.obj --force_regenerate
"""

from __future__ import annotations

import argparse

import torch
import trimesh

from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.utils.logger import log_info


def cli() -> None:
    """Command-line interface for grasp pose annotation.

    Parses CLI arguments, loads the mesh, and launches interactive
    annotation via the Viser browser UI.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Interactively annotate a grasp region on a mesh and "
            "compute antipodal point pairs."
        ),
    )

    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Path to the mesh file (e.g. .ply, .obj, .stl).",
    )
    parser.add_argument(
        "--viser_port",
        type=int,
        default=15531,
        help="Port for the browser-based annotation UI (default: 15531).",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=20000,
        help="Number of surface points to sample (default: 20000).",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=0.1,
        help="Maximum distance between antipodal pairs in metres (default: 0.1).",
    )
    parser.add_argument(
        "--min_length",
        type=float,
        default=0.001,
        help="Minimum distance between antipodal pairs in metres (default: 0.001).",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        default=False,
        help="Force re-annotation, ignoring cached antipodal pairs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device, e.g. 'cpu' or 'cuda' (default: cpu).",
    )

    args = parser.parse_args()

    # Load mesh via trimesh
    log_info(f"Loading mesh from {args.mesh_path}", color="green")
    mesh = trimesh.load(args.mesh_path, force="mesh")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=args.device)
    triangles = torch.tensor(mesh.faces, dtype=torch.int64, device=args.device)

    # Build configuration
    sampler_cfg = AntipodalSamplerCfg(
        n_sample=args.n_sample,
        max_length=args.max_length,
        min_length=args.min_length,
    )
    cfg = GraspGeneratorCfg(
        viser_port=args.viser_port,
        antipodal_sampler_cfg=sampler_cfg,
        force_regenerate=args.force_regenerate,
    )

    # Create generator and run annotation
    generator = GraspGenerator(vertices=vertices, triangles=triangles, cfg=cfg)
    log_info(
        "Annotate the grasp region in the browser window:\n"
        "  1. Open http://localhost:{port}\n"
        "  2. Click 'Rect Select Region' and drag to select\n"
        "  3. Click 'Confirm Selection' to finish",
        color="green",
    )
    hit_point_pairs = generator.annotate()

    log_info(
        f"Annotation complete. {hit_point_pairs.shape[0]} antipodal pairs cached.",
        color="green",
    )


if __name__ == "__main__":
    cli()
