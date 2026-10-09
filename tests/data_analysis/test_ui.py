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

import io
import json
import warnings
from embodichain.data_analysis.catalog import Catalog
from embodichain.data_analysis.ui import query_view, export_slice, compare_snapshots
from .test_catalog import _record, _measurement


def test_ui_copy_defaults_to_english_and_supports_chinese():
    from embodichain.data_analysis._i18n import (
        DEFAULT_LOCALE,
        dimension_label,
        normalize_locale,
        text,
    )

    assert DEFAULT_LOCALE == "en"
    assert normalize_locale(None) == "en"
    assert normalize_locale("unsupported") == "en"
    assert text("workbench_title") == "Data Diversity Analysis Workbench"
    assert text("workbench_title", "zh") == "数据多样性分析工作台"
    assert dimension_label("pose_x") == "Initial X"
    assert dimension_label("pose_x", "zh") == "初始 X"


def test_chinese_plot_uses_a_font_with_cjk_glyphs(tmp_path):
    from embodichain.data_analysis.ui import _dimension_plot

    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        catalog.upsert(_record("ep", "committed"))
    figure = _dimension_plot(query_view(path), "affordance", "zh")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        figure.savefig(io.BytesIO(), format="png")
    assert not [warning for warning in captured if "Glyph" in str(warning.message)]


def test_filter_counts_charts_and_export_share_exact_ids(tmp_path):
    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as c:
        for i, material in enumerate(["wood", "metal", "wood"]):
            r = _record(f"ep-{i}", "committed")
            r["dimensions"]["material"] = _measurement(material)
            c.upsert(r)
    view = query_view(path, {"material": "wood"})
    assert view["ids"] == ["ep-0", "ep-2"]
    assert (
        sum(row["count"] for row in view["distributions"]["material"])
        == len(view["rows"])
        == 2
    )
    exported = json.loads(
        export_slice(path, {"material": "wood"}, directory=tmp_path).read_text()
    )
    assert exported["episode_ids"] == view["ids"]
    assert query_view(path, {"material": "missing"})["ids"] == []


def test_snapshot_comparison_reports_added_ids_without_mutating_base(tmp_path):
    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as c:
        c.upsert(_record("ep-a", "committed"))
        first = c.snapshot("first")
        c.upsert(_record("ep-b", "committed"))
        second = c.snapshot("second")
    delta = compare_snapshots(path, first, second)
    assert delta["added"] == ["ep-b"]
    assert delta["removed"] == []
    assert delta["before_count"] == 1 and delta["after_count"] == 2


def test_export_applied_view_stays_frozen_when_catalog_changes(tmp_path):
    from embodichain.data_analysis.ui import _export_view

    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        catalog.upsert(_record("ep-a", "committed"))
    applied = query_view(path, {"material": "wood"})
    with Catalog(path) as catalog:
        catalog.upsert(_record("ep-b", "committed"))
    result = json.loads(
        _export_view(
            path, applied, filters={"material": "wood"}, directory=tmp_path
        ).read_text()
    )
    assert result["episode_ids"] == ["ep-a"]
    assert [r["episode_id"] for r in result["records"]] == ["ep-a"]


def test_structured_filters_joint_drill_and_export_keep_same_population(tmp_path):
    from embodichain.data_analysis.dimensions import build_filters
    from embodichain.data_analysis.ui import _drill_view, _export_view

    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        for i, (material, x) in enumerate(
            [("wood", -0.4), ("metal", -0.3), ("wood", -0.2)]
        ):
            record = _record(f"ep-{i}", "committed")
            record["dimensions"]["material"] = _measurement(material)
            record["dimensions"]["pose_x"] = {**_measurement(x), "unit": "m"}
            record["dimensions"]["light"] = {
                **_measurement(3.0),
                "unit": "renderer_intensity",
            }
            catalog.upsert(record)
    filters = build_filters({"material": ["wood", "metal"]}, {"pose_x": (-0.35, -0.1)})
    view = query_view(path, filters)
    assert view["ids"] == ["ep-1", "ep-2"]
    assert all(
        sum(c["count"] for c in cells) == 2 for cells in view["joint_cells"].values()
    )
    cell = next(c for c in view["joint_cells"]["material_light"] if c["x"] == "wood")
    selected = _drill_view(view, cell)
    assert selected["ids"] == ["ep-2"]
    assert all(
        sum(c["count"] for c in cells) == 1
        for cells in selected["joint_cells"].values()
    )
    exported = json.loads(
        _export_view(path, selected, filters=filters, directory=tmp_path).read_text()
    )
    assert exported["episode_ids"] == ["ep-2"]
    assert exported["joint_cell"]["x"] == "wood"
    assert exported["definition_version"] == 1


def test_gradio_build_wires_joint_selection_and_phase_analysis(tmp_path):
    from embodichain.data_analysis.ui import build_app

    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        catalog.upsert(_record("ep", "committed"))
    app = build_app(tmp_path / "catalog.sqlite")
    names = {dependency.get("api_name") for dependency in app.config["dependencies"]}
    assert {
        "refresh",
        "drill",
        "curves",
        "phases",
        "save_slice",
        "set_language",
    } <= names
    assert app.config["title"] == "EmbodiChain · Data Diversity Workbench"
    app.close()


def test_pose_bins_stay_comparable_across_filters(tmp_path):
    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        for i, x in enumerate([-0.6, -0.3, -0.1]):
            record = _record(f"ep-{i}", "committed")
            record["dimensions"]["pose_x"] = {**_measurement(x), "unit": "m"}
            catalog.upsert(record)
    full = query_view(path)
    filtered = query_view(path, {"pose_x": {"min": -0.4}})
    assert full["pose_bins"] == filtered["pose_bins"]


def test_overview_plot_handles_incompatible_assets_and_excludes_wrong_pose_units(
    tmp_path,
):
    from embodichain.data_analysis.ui import _plots

    path = tmp_path / "catalog.sqlite"
    with Catalog(path) as catalog:
        for i, asset in enumerate(["cube", ["bad"]]):
            record = _record(f"ep-{i}", "committed")
            record["dimensions"].update(
                asset=_measurement(asset),
                pose_x={**_measurement(0.1), "unit": "m" if i == 0 else "cm"},
                pose_y={**_measurement(0.2), "unit": "m"},
            )
            catalog.upsert(record)
    fig = _plots(query_view(path))
    assert sum(len(c.get_offsets()) for c in fig.axes[2].collections) == 1
