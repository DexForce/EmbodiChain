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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = ["InputKind", "Prompt2SceneInput"]

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


class InputKind(str, Enum):
    """Supported prompt2scene input kinds."""

    IMAGE = "image"
    EDIT = "edit"


@dataclass(frozen=True)
class Prompt2SceneInput:
    """Normalized prompt2scene input."""

    input_kind: InputKind
    output_root: Path
    image_path: Path | None = None
    prompt: str | None = None
    gravity_settle_mode: str = "geometry"
    z_axis_align_assets: bool = True

    @classmethod
    def from_cli_args(
        cls,
        *,
        image_path: Path | None,
        prompt: str | None,
        output_root: Path,
        gravity_settle_mode: str = "geometry",
        z_axis_align_assets: bool = True,
    ) -> "Prompt2SceneInput":
        """Create a prompt2scene input from CLI arguments.

        Args:
            image_path: Input image path, if image mode is selected.
            prompt: Optional edit prompt.
            output_root: Directory where prompt2scene outputs are written.

        Returns:
            Normalized prompt2scene input.

        Raises:
            FileNotFoundError: If the image input path does not exist.
            ValueError: If the image path is invalid.
        """
        output_root = output_root.expanduser().resolve()
        prompt_text = prompt.strip() if prompt is not None else None
        if prompt_text == "":
            prompt_text = None
        gravity_settle_mode = cls._validate_gravity_settle_mode(gravity_settle_mode)

        if image_path is not None:
            image_path = image_path.expanduser().resolve()
            cls._validate_image_path(image_path)
            return cls(
                input_kind=InputKind.IMAGE,
                image_path=image_path,
                output_root=output_root,
                prompt=prompt_text,
                gravity_settle_mode=gravity_settle_mode,
                z_axis_align_assets=bool(z_axis_align_assets),
            )

        return cls(
            input_kind=InputKind.EDIT,
            output_root=output_root,
            prompt=cls._validate_edit_only_prompt(prompt_text, output_root),
            gravity_settle_mode=gravity_settle_mode,
            z_axis_align_assets=bool(z_axis_align_assets),
        )

    def to_manifest(self) -> dict[str, object]:
        """Convert the input to a JSON-serializable manifest."""
        manifest: dict[str, object] = {
            "input_kind": self.input_kind.value,
            "output_root": str(self.output_root),
            "gravity_settle_mode": self.gravity_settle_mode,
            "z_axis_align_assets": self.z_axis_align_assets,
        }
        if self.input_kind == InputKind.IMAGE:
            image_path = self.image_path
            manifest["image_path"] = str(image_path)
        if self.prompt is not None:
            manifest["prompt"] = self.prompt
        return manifest

    @staticmethod
    def _validate_edit_only_prompt(prompt: str | None, output_root: Path) -> str:
        if prompt is None:
            raise ValueError(
                "Provide --image or --prompt with an existing output_root."
            )
        scene_state = output_root / "gym_export" / "scene_state" / "result.json"
        if not scene_state.is_file():
            raise FileNotFoundError(
                "Edit-only mode requires an existing scene state: " f"{scene_state}"
            )
        return prompt

    @staticmethod
    def _validate_gravity_settle_mode(gravity_settle_mode: str) -> str:
        mode = str(gravity_settle_mode or "geometry").strip().lower()
        if mode not in {"geometry", "physics"}:
            raise ValueError(
                "gravity_settle_mode must be 'geometry' or 'physics', "
                f"got {gravity_settle_mode!r}."
            )
        return mode

    @staticmethod
    def _validate_image_path(image_path: Path) -> None:
        """Validate supported image input paths."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image input not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Image input is not a file: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(
                "Image input must have one of these extensions: .jpg, .jpeg, .png"
            )
