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

"""Prepare matching renderer previews before approving any benchmark matrix.

Run: python -m scripts.benchmark.rendering.preview --help
This command captures review images only. It produces no performance numbers
and never marks a scene or quality profile as approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json
from scripts.benchmark.core.execution import execute_worker
from scripts.benchmark.core.provenance import (
    git_revision,
    package_version,
    software_snapshot,
)
from scripts.benchmark.rendering.scenes import (
    SCENE_NAMES,
    build_scene,
    camera_intrinsics,
    load_scene,
    scene_fingerprint,
)

__all__ = ["compare_geometry", "main", "write_comparison"]


def _read_render_profile(
    path: Path, expected_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the exact copied profile and bind its bytes and resolved values."""
    path = Path(path).resolve()
    profile = read_json_object(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen preview render profile checksum mismatch")
    return profile, {
        "path": str(path),
        "sha256": digest,
        "profile_sha256": stable_hash(profile),
    }


def _freeze_render_profile(
    source: Path | None, output: Path
) -> tuple[Path | None, dict[str, Any] | None]:
    """Copy a profile once; reuse cannot overwrite a different frozen profile."""
    frozen = output / "render_profile.json"
    if source is None:
        if not frozen.exists():
            return None, None
        previous = read_json_object(output / "review.json").get("render_profile")
        if not isinstance(previous, dict):
            raise ValueError(
                "Existing render_profile.json has no frozen review binding"
            )
        profile, record = _read_render_profile(frozen, previous["sha256"])
        if stable_hash(profile) != previous["profile_sha256"]:
            raise ValueError("Frozen preview render profile values changed")
    else:
        source = Path(source).resolve()
        _, record = _read_render_profile(source)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Render profile changed while preparing the preview")
        if frozen.exists():
            if frozen.read_bytes() != data:
                raise FileExistsError(
                    "A different render profile is already frozen; use a new output directory"
                )
        else:
            frozen.write_bytes(data)
        _, record = _read_render_profile(frozen, record["sha256"])
    return frozen, {**record, "path": frozen.name}


def compare_geometry(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare paired depth and instance evidence without certifying quality.

    Args:
        first: First renderer's canonical segmentation and optical-axis depth.
        second: Second renderer's corresponding arrays.

    Returns:
        Agreement and depth differences over common foreground pixels.
    """
    import numpy as np

    shape = first["seg"].shape
    if any(
        value.shape != shape
        for value in (first["depth"], second["depth"], second["seg"])
    ):
        raise ValueError("Paired depth and segmentation must have identical shapes")
    foreground = (first["seg"] > 0) | (second["seg"] > 0)
    same = first["seg"] == second["seg"]
    valid = (
        foreground & same & np.isfinite(first["depth"]) & np.isfinite(second["depth"])
    )
    difference = np.abs(first["depth"][valid] - second["depth"][valid])
    return {
        "pixel_instance_agreement_ratio": float(same.mean()),
        "foreground_instance_agreement_ratio": (
            float(same[foreground].mean()) if foreground.any() else None
        ),
        "common_depth_pixels": int(valid.sum()),
        "depth_median_absolute_difference_m": (
            float(np.median(difference)) if len(difference) else None
        ),
        "interpretation": "paired consistency only; no independent reference or quality qualification",
    }


def _worker(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "preview.json").exists():
        raise FileExistsError(f"Refusing to overwrite a preview: {args.output}")
    result: dict[str, Any] = {
        "purpose": "scene_preview",
        "approval_status": "pending_user_review",
        "benchmark_executed": False,
        "backend": args.worker,
        "status": "failed",
        "python": sys.version,
        "embodichain_commit": git_revision(ROOT),
        "software": software_snapshot(
            ROOT,
            [
                Path(__file__).resolve(),
                Path(__file__).with_name("scenes.py"),
                Path(__file__).with_name("color.py"),
                Path(__file__).parent / "backends" / f"scene_{args.worker}.py",
            ],
        ),
    }
    renderer = None
    num_envs = getattr(args, "num_envs", 1)
    try:
        profile_options = {}
        if args.render_profile is not None:
            profile, record = _read_render_profile(
                args.render_profile, args.render_profile_sha256
            )
            profile_options["profile"] = profile
            result["render_profile"] = record
        if args.worker == "embodichain":
            from scripts.benchmark.rendering.backends.scene_embodichain import (
                EmbodiChainScene,
            )

            cls = EmbodiChainScene
            result["dexsim"] = package_version("dexsim-engine")
        else:
            from scripts.benchmark.rendering.backends.scene_isaaclab import (
                IsaacLabScene,
            )

            cls = IsaacLabScene
            result["isaaclab_commit"] = git_revision(args.isaaclab_root)
            result["isaacsim"] = package_version("isaacsim")
        renderer = cls(
            args.manifest,
            width=args.width,
            height=args.height,
            gpu_id=args.gpu_id,
            modalities=("rgb", "depth", "seg"),
            **({"num_envs": num_envs} if num_envs != 1 else {}),
            **profile_options,
        )
        for _ in range(args.settle_frames):
            arrays = renderer.capture()
        names = renderer.metadata["camera_names"]
        if arrays["rgb"].shape != (num_envs * len(names), args.height, args.width, 3):
            raise ValueError("Rendered camera batch does not match the frozen scene")
        for index, name in enumerate(names):
            rgb = arrays["rgb"][index]
            if rgb.dtype != np.uint8 or rgb.std() < 1.0:
                raise ValueError(f"Empty or incorrectly typed RGB image: {name}")
            Image.fromarray(rgb).save(args.output / f"{name}.png")
            np.savez_compressed(
                args.output / f"{name}.npz",
                **{key: value[index] for key, value in arrays.items()},
            )
        result.update(
            status="completed",
            renderer=renderer.metadata,
            camera_names=names,
            num_envs=num_envs,
        )
        if num_envs > 1:
            np.savez_compressed(args.output / "batch_preview.npz", **arrays)
    except Exception as exc:
        result.update(
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        write_json(args.output / "preview.json", result)
        write_json(args.output / "result.json", result)
        if renderer is not None:
            renderer.close()
    if result["status"] != "completed":
        raise SystemExit(1)


def write_comparison(root: Path, scene_names: tuple[str, ...]) -> list[Path]:
    """Assemble labelled scientific comparison figures from untouched RGB outputs.

    Args:
        root: Preview directory containing scene/backend image directories.
        scene_names: Ordered scenes included in the review.

    Returns:
        Paths of the generated scene and multi-camera comparison figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    from scipy.spatial.transform import Rotation

    outputs = []
    audit = []
    review = read_json_object(root / "review.json")
    expected_profile = review.get("render_profile")
    if expected_profile is not None:
        _, frozen = _read_render_profile(
            root / expected_profile["path"], expected_profile["sha256"]
        )
        if frozen["profile_sha256"] != expected_profile["profile_sha256"]:
            raise ValueError("Comparison render profile differs from the frozen review")
    for name in scene_names:
        scene = load_scene(root / name / "scene.json")
        metadata = [
            json.loads((root / name / backend / "preview.json").read_text())
            for backend in ("embodichain", "isaaclab")
        ]
        if any(record["status"] != "completed" for record in metadata):
            raise ValueError(f"Cannot compare incomplete previews: {name}")
        for record in metadata:
            used = record.get("render_profile")
            if (
                (expected_profile is None) != (used is None)
                or expected_profile is not None
                and any(
                    used.get(key) != expected_profile[key]
                    for key in ("sha256", "profile_sha256")
                )
            ):
                raise ValueError(
                    f"Preview render profile fingerprints do not match: {name}"
                )
        lhs, rhs = [record["renderer"] for record in metadata]
        if any(
            record["scene_sha256"] != scene_fingerprint(scene) for record in (lhs, rhs)
        ):
            raise ValueError(f"Preview scene fingerprints do not match: {name}")
        if not np.allclose(
            lhs.get("batch_intrinsics", lhs["intrinsics"]),
            rhs.get("batch_intrinsics", rhs["intrinsics"]),
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError(f"Preview camera intrinsics do not match: {name}")
        native_poses = np.asarray(lhs["camera_poses"])
        usd_poses = np.tile(np.eye(4), (len(native_poses), 1, 1))
        usd_poses[:, :3, :3] = Rotation.from_quat(
            rhs["camera_quaternions_xyzw"][: len(native_poses)]
        ).as_matrix()
        usd_poses[:, :3, 3] = rhs["camera_positions"][: len(native_poses)]
        if not np.allclose(native_poses, usd_poses, atol=1e-5, rtol=0):
            raise ValueError(f"Resolved camera poses do not match: {name}")
        for view in scene["cameras"]:
            with (
                np.load(root / name / "embodichain" / (view["name"] + ".npz")) as first,
                np.load(root / name / "isaaclab" / (view["name"] + ".npz")) as second,
            ):
                audit.append(
                    {
                        "scene": name,
                        "camera": view["name"],
                        "pose_and_image_scope": "environment_0; batch intrinsics checked separately",
                        **compare_geometry(dict(first), dict(second)),
                    }
                )
    write_json(
        root / "geometry_checks.json",
        {"quality_status": "not_qualified", "views": audit},
    )
    scene_titles = {
        "calibration": "Calibration table",
        "workshop": "Workshop",
        "franka_workstation": "Franka workstation",
    }

    def compose(rows: list[tuple[str, str]], name: str) -> None:
        fig, axes = plt.subplots(
            len(rows), 2, figsize=(14, 5.1 * len(rows)), squeeze=False
        )
        fig.patch.set_facecolor("#f3f5f7")
        for row, (scene, camera) in enumerate(rows):
            for column, backend in enumerate(("embodichain", "isaaclab")):
                ax = axes[row, column]
                path = root / scene / backend / f"{camera}.png"
                ax.imshow(mpimg.imread(path))
                label = (
                    "EmbodiChain / DexSim"
                    if backend == "embodichain"
                    else "Isaac Lab / RTX"
                )
                ax.set_title(
                    f"{label}  |  {scene_titles[scene]}  |  {camera}", fontsize=10
                )
                ax.axis("off")
        fig.suptitle(
            (
                "Shared geometry and camera poses — declared RGB delivery"
                if expected_profile is not None
                else "Shared geometry and camera poses — raw renderer output"
            ),
            fontsize=16,
            y=0.995,
        )
        fig.text(
            0.5,
            0.008,
            "Preview only · no benchmark executed · lighting / image quality not yet qualified",
            ha="center",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.024, 1, 0.975))
        path = root / name
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(path)

    compose([(name, "external_front") for name in scene_names], "scene_comparison.png")
    if "franka_workstation" in scene_names:
        scene = load_scene(root / "franka_workstation" / "scene.json")
        compose(
            [("franka_workstation", camera["name"]) for camera in scene["cameras"]],
            "franka_camera_comparison.png",
        )
    review_path = root / "review.json"
    review.update(
        preview_execution_status="completed",
        comparison_figures=[path.name for path in outputs],
        geometry_checks="geometry_checks.json",
    )
    write_json(review_path, review)
    return outputs


def main(argv: list[str] | None = None) -> None:
    """Freeze scenes, render isolated previews and write a review manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("all", *SCENE_NAMES), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--settle-frames", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--embodichain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument("--franka-urdf", type=Path)
    parser.add_argument(
        "--render-profile",
        type=Path,
        help="Optional render-profile JSON copied and hashed into this preview; no benchmark/quality approval is implied.",
    )
    parser.add_argument("--render-profile-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-scenes",
        action="store_true",
        help="Reuse verified frozen scene manifests; worker outputs must be new.",
    )
    parser.add_argument(
        "--worker", choices=("embodichain", "isaaclab"), help=argparse.SUPPRESS
    )
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.render_profile_sha256 is not None and args.render_profile is None:
        parser.error("--render-profile-sha256 requires --render-profile")
    if (
        min(args.width, args.height, args.settle_frames, args.num_envs) < 1
        or args.gpu_id < 0
    ):
        parser.error(
            "width, height and settle-frames must be positive; gpu-id must be nonnegative"
        )
    args.output = args.output.resolve()
    if args.worker:
        _worker(args)
        return
    if not args.prepare_only and (
        args.isaaclab_root is None or args.isaaclab_python is None
    ):
        parser.error(
            "Both --isaaclab-root and --isaaclab-python are required for previews"
        )
    args.output.mkdir(parents=True, exist_ok=args.reuse_scenes)
    frozen_profile, profile_record = _freeze_render_profile(
        args.render_profile, args.output
    )
    scene_names = SCENE_NAMES if args.scene == "all" else (args.scene,)
    records = []
    for name in scene_names:
        path = args.output / name / "scene.json"
        if not args.reuse_scenes:
            path = build_scene(name, path.parent, franka_urdf=args.franka_urdf)
        scene = load_scene(path)
        records.append(
            {
                "scene": name,
                "manifest": str(path.relative_to(args.output)),
                "sha256": scene_fingerprint(scene),
                "intrinsics": camera_intrinsics(scene, args.width, args.height),
            }
        )
    write_json(
        args.output / "review.json",
        {
            "purpose": "scene_preview",
            "approval_status": "pending_user_review",
            "benchmark_executed": False,
            "scenes": records,
            "width": args.width,
            "height": args.height,
            "gpu_id": args.gpu_id,
            **(
                {"render_profile": profile_record} if profile_record is not None else {}
            ),
        },
    )
    if args.prepare_only:
        print(args.output / "review.json")
        return
    for name in scene_names:
        for backend in ("embodichain", "isaaclab"):
            output = args.output / name / backend
            command = [
                str(
                    args.embodichain_python
                    if backend == "embodichain"
                    else args.isaaclab_python
                ),
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--manifest",
                str(args.output / name / "scene.json"),
                "--output",
                str(output),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--gpu-id",
                str(args.gpu_id),
                "--settle-frames",
                str(args.settle_frames),
                "--num-envs",
                str(args.num_envs),
                "--isaaclab-root",
                str(args.isaaclab_root),
            ]
            if frozen_profile is not None:
                command.extend(
                    [
                        "--render-profile",
                        str(frozen_profile),
                        "--render-profile-sha256",
                        profile_record["sha256"],
                    ]
                )
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment["PYTHONUNBUFFERED"] = "1"
            imports = [str(ROOT)]
            if backend == "isaaclab":
                imports += [
                    str(path)
                    for path in sorted((args.isaaclab_root / "source").iterdir())
                    if path.is_dir()
                ]
                environment.pop("CONDA_PREFIX", None)
                environment["VIRTUAL_ENV"] = str(
                    args.isaaclab_python.absolute().parent.parent
                )
                environment["ACCEPT_EULA"] = "Y"
                environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
            environment["PYTHONPATH"] = os.pathsep.join(imports)
            log_path = output / "worker.log"
            print(f"Rendering {name} / {backend}; log: {log_path}", flush=True)
            result = execute_worker(
                command,
                output,
                backend=backend,
                repeat=0,
                timeout_s=args.timeout_s,
                env=environment,
            )
            if result["status"] != "completed":
                raise RuntimeError(
                    f"Preview failed: {name}/{backend}; inspect {log_path}"
                )
    for path in write_comparison(args.output, tuple(scene_names)):
        print(path)


if __name__ == "__main__":
    main()
