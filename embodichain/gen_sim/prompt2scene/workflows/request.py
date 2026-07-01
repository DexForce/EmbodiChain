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
    TEXT = "text"


@dataclass(frozen=True)
class Prompt2SceneInput:
    """Normalized prompt2scene input."""

    input_kind: InputKind
    output_root: Path
    image_path: Path | None = None
    text: str | None = None

    @classmethod
    def from_cli_args(
        cls,
        *,
        image_path: Path | None,
        text: str | None,
        output_root: Path,
    ) -> "Prompt2SceneInput":
        """Create a prompt2scene input from CLI arguments.

        Args:
            image_path: Input image path, if image mode is selected.
            text: Text prompt, if text mode is selected.
            output_root: Directory where prompt2scene outputs are written.

        Returns:
            Normalized prompt2scene input.

        Raises:
            FileNotFoundError: If the image input path does not exist.
            ValueError: If the image path is invalid or text input is empty.
        """
        output_root = output_root.expanduser().resolve()

        if image_path is not None:
            image_path = image_path.expanduser().resolve()
            cls._validate_image_path(image_path)
            return cls(
                input_kind=InputKind.IMAGE,
                image_path=image_path,
                output_root=output_root,
            )

        if text is None or not text.strip():
            raise ValueError("Text input must be non-empty.")

        return cls(
            input_kind=InputKind.TEXT,
            text=text.strip(),
            output_root=output_root,
        )

    def to_manifest(self) -> dict[str, str]:
        """Convert the input to a JSON-serializable manifest."""
        manifest: dict[str, str] = {
            "input_kind": self.input_kind.value,
            "output_root": str(self.output_root),
        }
        if self.input_kind == InputKind.IMAGE:
            image_path = self.image_path
            manifest["image_path"] = str(image_path)
        else:
            text = self.text
            manifest["text"] = "" if text is None else text
        return manifest

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
