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

import json
from pathlib import Path

from embodichain.data_analysis import DIMENSION_KEYS, import_lerobot_metadata


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_import_lerobot_sidecar_preserves_known_metadata_and_source(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "meta" / "embodichain_episodes.jsonl",
        [
            {
                "episode_index": 4,
                "task": "pick-place",
                "seed": 9,
                "robot_id": "franka",
                "dimensions": {
                    "material": {
                        "value": "wood",
                        "source": "configured",
                        "unit": "",
                        "frame": "world",
                        "scope": "episode",
                    }
                },
                "segments": [{"name": "pick"}],
            }
        ],
    )

    records = import_lerobot_metadata(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["episode_id"] == f"lerobot:{tmp_path.resolve()}:4"
    assert record["dimensions"]["material"]["value"] == "wood"
    assert record["dimensions"]["light"]["source"] == "unknown"
    assert record["provenance"]["historical_metadata"] == "available"
    assert record["segments"] == [{"name": "pick"}]


def test_import_standard_lerobot_metadata_marks_missing_analysis_unavailable(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick cube"], "length": 12},
            {"episode_index": 1, "tasks": ["place cube"], "length": 8},
        ],
    )

    records = import_lerobot_metadata(tmp_path)
    assert [record["episode_id"].rsplit(":", 1)[-1] for record in records] == ["0", "1"]
    assert records[0]["artifacts"] == {}
    assert records[0]["provenance"]["historical_metadata"] == "unavailable"
    assert set(records[0]["dimensions"]) == set(DIMENSION_KEYS)
    assert all(
        measurement["value"] is None
        for measurement in records[0]["dimensions"].values()
    )
    assert all(
        measurement["missing_reason"] == "historical_metadata_unavailable"
        for measurement in records[0]["dimensions"].values()
    )


def test_old_embodichain_sidecar_without_dimensions_is_unavailable(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "meta" / "embodichain_episodes.jsonl",
        [{"lerobot_episode_index": 2, "instruction": "pick cube", "segments": []}],
    )

    record = import_lerobot_metadata(tmp_path)[0]
    assert record["task_id"] == "pick cube"
    assert record["provenance"]["historical_metadata"] == "unavailable"


def test_import_legacy_info_count_without_episode_rows(tmp_path: Path) -> None:
    info = {"total_episodes": 2, "fps": 30, "robot_type": "franka"}
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")

    records = import_lerobot_metadata(tmp_path)
    assert len(records) == 2
    assert records[0]["robot_id"] == "franka"
    assert records[0]["metrics"] == {"fps": 30}
