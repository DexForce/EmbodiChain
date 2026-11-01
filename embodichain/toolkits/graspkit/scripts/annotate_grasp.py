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

    embodichain annotate-grasp --mesh_path mug.obj --gripper-model franka_panda_hand
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for grasp pose annotation.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain annotate-grasp",
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
        "--gripper-model",
        required=True,
        help="Explicit built-in gripper identity; supplies the actual URDF collision geometry.",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=None,
        help="Maximum antipodal contact distance in metres (default: selected model).",
    )
    parser.add_argument(
        "--min_length",
        type=float,
        default=None,
        help="Minimum antipodal contact distance in metres (default: selected model).",
    )
    parser.add_argument(
        "--gripper-side",
        choices=("left", "right"),
        help="Explicit side for the asymmetric Marvin built-in model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device, e.g. 'cpu' or 'cuda' (default: cpu).",
    )

    args = parser.parse_args(argv)

    import torch
    import trimesh

    from embodichain.toolkits.graspkit import (
        ParallelJawGripperModelCfg,
        get_parallel_jaw_gripper_model,
    )
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
    )
    from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
        get_parallel_jaw_collision_geometry,
    )
    from embodichain.utils.logger import log_info

    selected_model = get_parallel_jaw_gripper_model(args.gripper_model)
    # Retain the existing contact-pair filters; they never synthesize geometry.
    model = ParallelJawGripperModelCfg(
        model_id=selected_model.model_id,
        min_opening_width=(
            selected_model.min_opening_width
            if args.min_length is None
            else args.min_length
        ),
        max_opening_width=(
            selected_model.max_opening_width
            if args.max_length is None
            else args.max_length
        ),
    )
    geometry = get_parallel_jaw_collision_geometry(
        args.gripper_model, side=args.gripper_side
    )

    # Load mesh via trimesh
    log_info(f"Loading mesh from {args.mesh_path}", color="green")
    mesh = trimesh.load(args.mesh_path, force="mesh")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=args.device)
    triangles = torch.tensor(mesh.faces, dtype=torch.int64, device=args.device)

    generator = AntipodalGraspPoseGenerator(
        model,
        collision_geometry=geometry,
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(sample_count=args.n_sample),
        annotation_cfg=GraspAnnotationCfg(
            selection_mode="interactive",
            viser_port=args.viser_port,
            force_refresh=True,
        ),
    )
    log_info(
        "Annotate the grasp region in the browser window:\n"
        f"  1. Open http://localhost:{args.viser_port}\n"
        "  2. Click 'Rect Select Region' and drag to select\n"
        "  3. Click 'Confirm Selection' to finish",
        color="green",
    )
    hit_point_pairs = generator.prepare_mesh(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
    )

    log_info(
        f"Annotation complete. {hit_point_pairs.shape[0]} antipodal pairs cached.",
        color="green",
    )


if __name__ == "__main__":
    cli()


__all__ = ["cli"]
