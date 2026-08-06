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

"""Validate wheel dependency metadata against public-index requirements."""

from __future__ import annotations

import argparse
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from packaging.requirements import InvalidRequirement, Requirement

__all__ = ["WheelMetadataError", "read_wheel_metadata", "validate_wheel"]


class WheelMetadataError(ValueError):
    """Raised when wheel metadata cannot be published to a public index."""


def read_wheel_metadata(wheel_path: Path) -> Message:
    """Read the single ``METADATA`` document from a wheel.

    Args:
        wheel_path: Path to the wheel archive.

    Returns:
        Parsed core metadata for the wheel.

    Raises:
        WheelMetadataError: If the wheel is missing, invalid, or does not
            contain exactly one ``.dist-info/METADATA`` document.
    """
    if not wheel_path.is_file():
        raise WheelMetadataError(f"Wheel does not exist: {wheel_path}")

    try:
        with ZipFile(wheel_path) as wheel:
            metadata_names = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise WheelMetadataError(
                    f"Expected one METADATA document in {wheel_path}, "
                    f"found {metadata_names}"
                )
            metadata_bytes = wheel.read(metadata_names[0])
    except BadZipFile as exc:
        raise WheelMetadataError(f"Invalid wheel archive: {wheel_path}") from exc

    return BytesParser(policy=policy.default).parsebytes(metadata_bytes)


def validate_wheel(wheel_path: Path) -> None:
    """Reject direct URL dependencies that public indexes do not accept.

    Args:
        wheel_path: Path to the wheel archive.

    Raises:
        WheelMetadataError: If dependency metadata is invalid or contains a
            direct URL reference.
    """
    metadata = read_wheel_metadata(wheel_path)
    direct_dependencies: list[str] = []

    for raw_requirement in metadata.get_all("Requires-Dist", []):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as exc:
            raise WheelMetadataError(
                f"Invalid Requires-Dist in {wheel_path}: {raw_requirement}"
            ) from exc
        if requirement.url is not None:
            direct_dependencies.append(raw_requirement)

    if direct_dependencies:
        details = "\n".join(f"- {requirement}" for requirement in direct_dependencies)
        raise WheelMetadataError(
            f"PyPI does not accept direct dependency references in {wheel_path}:\n"
            f"{details}"
        )


def main() -> int:
    """Validate wheel paths supplied on the command line."""
    parser = argparse.ArgumentParser(
        description="Reject wheel dependencies that use direct URL references."
    )
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    for wheel_path in args.wheels:
        validate_wheel(wheel_path)
        print(f"Validated PyPI dependency metadata: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
