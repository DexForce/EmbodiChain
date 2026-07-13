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

from pathlib import Path
import time

from PIL import Image

from embodichain.gen_sim.prompt2scene.agent_tools.clients.geometry_generation_client import (
    GeometryGenerationClient,
    GeometryGenerationError,
    GeometryGenerationServerRequest,
    MultiObjectGenerationError,
    MultiObjectGenerationServerRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_generation_manager.schemas import (
    GeometryGenerationRequest,
    GeometryGenerationResult,
    MultiObjectGenerationObject,
    MultiObjectGenerationRequest,
    MultiObjectGenerationResult,
    RgbaImageToGeometryRequest,
    RgbaImagesToGeometriesObject,
    RgbaImagesToGeometriesRequest,
    RgbaImagesToGeometriesResult,
)


class GeometryGenerationManager:
    """Geometry generation domain operations."""

    def __init__(self, *, client: GeometryGenerationClient | None = None) -> None:
        self.client = client or GeometryGenerationClient()

    def generate_single_object_mesh(
        self,
        request: GeometryGenerationRequest,
    ) -> GeometryGenerationResult:
        image_path = request.image_path.expanduser().resolve()
        output_path = request.output_path.expanduser().resolve()
        _validate_single_object_request(image_path=image_path, output_path=output_path)

        started_perf = time.perf_counter()
        response = self.client.generate(
            GeometryGenerationServerRequest(
                image_path=image_path,
                output_path=output_path,
            ),
        )
        elapsed_seconds = _elapsed_seconds(started_perf)
        if isinstance(response, GeometryGenerationError):
            raise RuntimeError(response.error_message)

        return GeometryGenerationResult(
            output_path=Path(response.result.geometry_path).expanduser().resolve(),
            sam3d_generation_elapsed_seconds=elapsed_seconds,
        )

    def generate_multi_object_meshes(
        self,
        request: MultiObjectGenerationRequest,
    ) -> MultiObjectGenerationResult:
        image_path = request.image_path.expanduser().resolve()
        output_dir = request.output_dir.expanduser().resolve()
        _validate_multi_object_request(
            image_path=image_path,
            mask_paths=request.mask_paths,
            output_dir=output_dir,
        )

        started_perf = time.perf_counter()
        response = self.client.generate_multiple_objects(
            MultiObjectGenerationServerRequest(
                image_path=image_path,
                mask_paths=[p.expanduser().resolve() for p in request.mask_paths],
            ),
            output_dir=output_dir,
        )
        elapsed_seconds = _elapsed_seconds(started_perf)
        if isinstance(response, MultiObjectGenerationError):
            raise RuntimeError(response.error_message)

        objects = [
            MultiObjectGenerationObject(
                name=item.name,
                geometry_path=Path(item.geometry_path).expanduser().resolve(),
                rotation_quaternion_wxyz=item.rotation_quaternion_wxyz,
                translation=item.translation,
                scale=item.scale,
            )
            for item in response.result.objects
        ]
        return MultiObjectGenerationResult(
            objects=objects,
            sam3d_generation_elapsed_seconds=elapsed_seconds,
        )

    def convert_rgba_image_to_geometry(
        self,
        request: RgbaImageToGeometryRequest,
    ) -> Path:
        image_path = request.image_path.expanduser().resolve()
        output_path = request.output_path.expanduser().resolve()
        _validate_rgba_image(image_path)

        result = self.generate_single_object_mesh(
            GeometryGenerationRequest(image_path=image_path, output_path=output_path)
        )
        return _postprocess_mesh(result.output_path)

    def convert_rgba_images_to_geometries(
        self,
        request: RgbaImagesToGeometriesRequest,
    ) -> RgbaImagesToGeometriesResult:
        image_path = request.image_path.expanduser().resolve()
        output_dir = request.output_dir.expanduser().resolve()
        _validate_rgba_images_request(image_path, request.mask_paths)

        result = self.generate_multi_object_meshes(
            MultiObjectGenerationRequest(
                image_path=image_path,
                mask_paths=request.mask_paths,
                output_dir=output_dir,
            )
        )
        objects = [
            RgbaImagesToGeometriesObject(
                name=item.name,
                geometry_path=_postprocess_mesh(item.geometry_path),
                rotation_quaternion_wxyz=item.rotation_quaternion_wxyz,
                translation=item.translation,
                scale=item.scale,
            )
            for item in result.objects
        ]
        return RgbaImagesToGeometriesResult(
            objects=objects,
            sam3d_generation_elapsed_seconds=result.sam3d_generation_elapsed_seconds,
        )


def _validate_single_object_request(*, image_path: Path, output_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Geometry generation input not found: {image_path}")
    if output_path.suffix.lower() != ".glb":
        raise ValueError("Geometry generation output_path must be a GLB file path.")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"Geometry generation output_path is a directory: {output_path}")


def _validate_multi_object_request(
    *,
    image_path: Path,
    mask_paths: list[Path],
    output_dir: Path,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Multi-object geometry generation input not found: {image_path}"
        )
    if not mask_paths:
        raise ValueError("mask_paths must be non-empty.")
    for mask_path in mask_paths:
        mask_path_resolved = mask_path.expanduser().resolve()
        if not mask_path_resolved.is_file():
            raise FileNotFoundError(
                f"Multi-object geometry mask not found: {mask_path_resolved}"
            )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"Multi-object geometry output_dir is not a directory: {output_dir}"
        )


def _validate_rgba_image(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"RGBA image not found: {image_path}")

    with Image.open(image_path) as image:
        if image.mode in {"RGBA", "LA"}:
            return
        if image.mode == "P" and "transparency" in image.info:
            return
        raise ValueError(
            "Geometry tool requires an image with an alpha channel, "
            f"got mode={image.mode!r}: {image_path}"
        )


def _validate_rgba_images_request(
    image_path: Path,
    mask_paths: list[Path],
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Scene image not found: {image_path}")
    with Image.open(image_path):
        pass
    if not mask_paths:
        raise ValueError("mask_paths must be non-empty.")
    for mask_path in mask_paths:
        if not mask_path.expanduser().resolve().is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")


def _postprocess_mesh(mesh_path: Path) -> Path:
    return mesh_path.expanduser().resolve()


def _elapsed_seconds(started_perf: float) -> float:
    return round(max(0.0, time.perf_counter() - started_perf), 6)
