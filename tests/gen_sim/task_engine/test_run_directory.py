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

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.run_directory import reserve_run_directory

_NOW = datetime(2026, 8, 20, 7, 24, 36, tzinfo=timezone(timedelta(hours=8)))


def test_run_directory_uses_local_second_timestamp(tmp_path: Path) -> None:
    root = tmp_path / "task2_2"

    with reserve_run_directory(root, now=_NOW) as allocation:
        assert allocation.run_id == "20260820_072436"
        assert allocation.path == root / "20260820_072436"
        assert not allocation.path.exists()
        allocation.path.mkdir()

    assert allocation.path.is_dir()
    assert not (root / ".20260820_072436.reserve").exists()


def test_run_directory_adds_suffix_for_same_second_runs(tmp_path: Path) -> None:
    root = tmp_path / "task2_2"
    (root / "20260820_072436").mkdir(parents=True)

    with reserve_run_directory(root, now=_NOW) as first:
        with reserve_run_directory(root, now=_NOW) as second:
            assert first.run_id == "20260820_072436_01"
            assert second.run_id == "20260820_072436_02"


def test_run_directory_rejects_naive_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        with reserve_run_directory(
            tmp_path,
            now=datetime(2026, 8, 20, 7, 24, 36),
        ):
            pass
