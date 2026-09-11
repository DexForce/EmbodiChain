"""Task100 smoke matrix for executable E1-E5 scene exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path("gym_project/task100")

TASK100_CASES = (
    ("task1101", "E2", "扶正桌上红色的罐头。"),
    ("task1103", "E1", "把红色杯子放入托盘。"),
    ("task1108", "E3", "拿起水壶，移到蓝色杯子上方并完成一次倾倒动作。"),
    ("task1110", "E5", "用双臂共同拿起桌面中央的托盘。"),
    ("task1111", "E1", "把苹果放入白色碗中。"),
    ("task1113", "E2", "扶正倒在桌边的杯子。"),
    ("task1115", "E3", "拿起量杯，移到白色碗上方并完成一次倾倒动作。"),
    ("task1117", "E4", "竖直握持黄色瓶子，从左臂交接给右臂并由右臂继续保持竖直。"),
    ("task1122", "E1", "把绿色罐头放进收纳托盘。"),
    ("task1125", "E2", "扶正倒下的杯子中最大的一个。"),
    ("task1127", "E5", "用双臂共同拿起大盘子。"),
)


@pytest.mark.parametrize("task_id,route,instruction", TASK100_CASES)
def test_task100_case_has_complete_scene_export(
    task_id: str, route: str, instruction: str
) -> None:
    """Keep the executable matrix limited to complete portable scene exports."""
    export = ROOT / task_id / "scene_export"
    assert route in {"E1", "E2", "E3", "E4", "E5"}
    assert instruction
    documents = {
        name: json.loads((export / name).read_text())
        for name in ("scene_config.json", "scene.json", "scene_graph.json")
    }
    assert documents["scene_config.json"]["format"] == "embodichain.scene-export/v1"

    asset_paths: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"fpath", "urdf_path", "usd_path"} and isinstance(child, str):
                    asset_paths.append(child)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(documents["scene_config.json"])
    assert asset_paths
    assert all((export / path).is_file() for path in asset_paths)


def test_task100_matrix_covers_e1_to_e5() -> None:
    """Require each requested semantic route in the selected matrix."""
    assert {route for _, route, _ in TASK100_CASES} == {"E1", "E2", "E3", "E4", "E5"}
