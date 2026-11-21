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

"""Correct the pinned cp311 bpy artifact's upstream cp39 WHEEL metadata tag."""

from __future__ import annotations

import base64
import csv
import hashlib
from importlib.metadata import distribution
import io
import os
from pathlib import Path
import sys
import tempfile

__all__ = []

_OLD_TAG = "Tag: cp39-cp39-manylinux_2_28_x86_64"
_NEW_TAG = "Tag: cp311-cp311-manylinux_2_28_x86_64"


def _replace(path: Path, data: bytes) -> None:
    # uv may hardlink installed files to its cache. Replace the inode instead
    # of writing through it, so the original downloaded artifact stays intact.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.chmod(0o644)
    os.replace(temporary, path)


def _repair_files(wheel: Path, record: Path) -> None:
    source = wheel.read_text(encoding="utf-8")
    if _NEW_TAG in source:
        return
    if source.count(_OLD_TAG) != 1:
        raise RuntimeError("Unexpected bpy wheel tag; review the pinned artifact")
    updated = source.replace(_OLD_TAG, _NEW_TAG).encode("utf-8")
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    entry = next(row for row in rows if row[0] == f"{wheel.parent.name}/WHEEL")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(updated).digest()).rstrip(b"=").decode()
    )
    entry[1:] = [f"sha256={digest}", str(len(updated))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    _replace(wheel, updated)
    _replace(record, output.getvalue().encode("utf-8"))


def _main() -> None:
    package = distribution("bpy")
    if sys.version_info[:2] != (3, 11) or package.version != "4.5.15":
        raise RuntimeError(
            "This correction applies only to the pinned bpy 4.5.15 cp311 artifact"
        )
    relative = next(
        path for path in package.files or [] if str(path).endswith(".dist-info/WHEEL")
    )
    wheel = Path(package.locate_file(relative))
    _repair_files(wheel, wheel.with_name("RECORD"))
    print(
        "bpy 4.5.15: corrected upstream WHEEL tag to match its cp311 filename/Requires-Python"
    )


if __name__ == "__main__":
    _main()
