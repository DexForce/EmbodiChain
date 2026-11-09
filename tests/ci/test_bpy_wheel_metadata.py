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

"""Validate the targeted metadata correction without changing cached artifacts."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import os
from pathlib import Path


def test_bpy_metadata_repair_updates_record_without_mutating_cache(
    tmp_path: Path,
) -> None:
    path = Path(__file__).resolve().parents[2] / "docker/repair-bpy-wheel.py"
    spec = importlib.util.spec_from_file_location("bpy_metadata_repair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp39-cp39-manylinux_2_28_x86_64\n"
    cached = tmp_path / "cached-wheel"
    cached.write_bytes(original)
    metadata = tmp_path / "bpy-4.5.15.dist-info"
    metadata.mkdir()
    wheel, record = metadata / "WHEEL", metadata / "RECORD"
    os.link(cached, wheel)
    record.write_text(
        f"{metadata.name}/WHEEL,sha256=original,{len(original)}\n{metadata.name}/RECORD,,\n"
    )

    module._repair_files(wheel, record)
    actual = wheel.read_bytes()
    assert b"Tag: cp311-cp311-manylinux_2_28_x86_64\n" in actual
    assert cached.read_bytes() == original
    entries = list(csv.reader(io.StringIO(record.read_text())))
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(actual).digest()).rstrip(b"=").decode()
    )
    assert entries[0] == [
        f"{metadata.name}/WHEEL",
        f"sha256={digest}",
        str(len(actual)),
    ]
    assert entries[1] == [f"{metadata.name}/RECORD", "", ""]
    previous = record.read_bytes()
    module._repair_files(wheel, record)
    assert record.read_bytes() == previous
