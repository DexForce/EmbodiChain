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

"""Lightweight Gradio workbench backed by the shared offline catalog."""

from __future__ import annotations

import html
import json
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from ._i18n import (
    DEFAULT_LOCALE,
    LANGUAGE_CHOICES,
    availability_choices,
    dimension_choices,
    dimension_label,
    joint_choices,
    normalize_locale,
    status_choices,
    text,
)
from .catalog import Catalog
from .statistics import distribution, joint_distribution
from .dimensions import (
    DIMENSIONS,
    build_filters,
    dimension_summary,
    joint_cells,
    select_cell,
    match_dimension,
)
from .schema import DIMENSION_KEYS

__all__ = ["query_view", "export_slice", "compare_snapshots", "build_app"]

_PLOT_LOCK = threading.Lock()


def _apply_plot_locale(figure: Any, locale: str) -> Any:
    if normalize_locale(locale) == "zh":
        from matplotlib import font_manager
        from matplotlib.font_manager import FontProperties
        from matplotlib.text import Text

        fonts = sorted(
            path
            for path in font_manager.findSystemFonts()
            if path.endswith("NotoSansCJK-Regular.ttc")
        )
        if fonts:
            properties = FontProperties(fname=fonts[0])
            for label in figure.findobj(match=Text):
                label.set_fontproperties(properties)
    return figure


def query_view(
    catalog_path: str | Path,
    filters: dict[str, Any] | None = None,
    status: str | None = "committed",
) -> dict[str, Any]:
    """Produce a single filtered population for tables, plots and export."""
    with Catalog(catalog_path) as catalog:
        records = catalog.records(filters=filters, status=status)
        summary = catalog.summary()
        reference = catalog.records(status=status)

    all_values = [
        r["dimensions"]["pose_x"]["value"]
        for r in reference
        if match_dimension(r, "pose_x", {"availability": "known"})
    ]
    numeric = [
        v for v in all_values if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if numeric:
        lo, hi = min(numeric), max(numeric)
        if lo == hi:
            lo, hi = lo - 0.005, hi + 0.005
        pose_bins = [lo + (hi - lo) * i / 5 for i in range(6)]
    else:
        pose_bins = [-1.0, 0.0, 1.0]
    return _build_view(records, summary, pose_bins)


def _build_view(
    records: list[dict[str, Any]], summary: dict[str, Any], pose_bins: list[float]
) -> dict[str, Any]:
    def value(record: dict[str, Any], key: str) -> Any:
        return record["dimensions"].get(key, {}).get("value")

    view = {
        "ids": [r["episode_id"] for r in records],
        "records": records,
        "summary": summary,
        "rows": [
            [
                r["episode_id"],
                r["status"],
                value(r, "asset"),
                value(r, "material"),
                value(r, "light"),
                value(r, "pose_x"),
                value(r, "pose_y"),
                r["metrics"].get("lift_height_m"),
                r["metrics"].get("final_xy_error_m"),
                len(r["segments"]),
            ]
            for r in records
        ],
        "distributions": {k: distribution(records, k) for k in DIMENSION_KEYS},
        "joint": joint_distribution(records, "material", "light"),
    }

    view["completeness"] = dimension_summary(records)
    view["joint_cells"] = {
        "asset_pose": joint_cells(records, "asset", "pose_x", y_bins=pose_bins),
        "affordance_approach": joint_cells(records, "affordance", "approach"),
        "material_light": joint_cells(records, "material", "light"),
    }
    view["pose_bins"] = pose_bins
    return view


def export_slice(
    catalog_path: str | Path,
    filters: dict[str, Any] | None = None,
    *,
    status: str | None = "committed",
    directory: str | Path | None = None,
) -> Path:
    """Export exact references and selection criteria without copying source data."""
    view = query_view(catalog_path, filters, status)
    return _export_view(
        catalog_path, view, filters=filters, status=status, directory=directory
    )


def _export_view(
    catalog_path: str | Path,
    view: dict[str, Any],
    *,
    filters: dict[str, Any] | None = None,
    status: str | None = "committed",
    directory: str | Path | None = None,
) -> Path:
    root = (
        Path(directory)
        if directory is not None
        else Path(tempfile.mkdtemp(prefix="embodichain-slice-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"slice-{uuid.uuid4().hex[:8]}.json"
    payload = {
        "schema_version": 1,
        "definition_version": 1,
        "joint_cell": view.get("selected_cell"),
        "joint_cells": view.get("selected_cells", []),
        "pose_bins": view["pose_bins"],
        "catalog": str(Path(catalog_path).resolve()),
        "filters": filters or {},
        "status": status,
        "episode_ids": view["ids"],
        "records": view["records"],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def compare_snapshots(
    catalog_path: str | Path, before: int, after: int
) -> dict[str, Any]:
    """Compare committed populations in two immutable dataset versions."""
    with Catalog(catalog_path) as catalog:
        a = [r for r in catalog.snapshot_records(before) if r["status"] == "committed"]
        b = [r for r in catalog.snapshot_records(after) if r["status"] == "committed"]
    ids_a = {r["episode_id"] for r in a}
    ids_b = {r["episode_id"] for r in b}
    return {
        "before_count": len(a),
        "after_count": len(b),
        "added": sorted(ids_b - ids_a),
        "removed": sorted(ids_a - ids_b),
        "material_before": distribution(a, "material"),
        "material_after": distribution(b, "material"),
    }


def _plots(view: dict[str, Any], locale: str = DEFAULT_LOCALE) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import numpy as np

    with _PLOT_LOCK:
        fig = Figure(figsize=(13, 6.2), layout="constrained", facecolor="#ffffff")
        axes = fig.subplots(2, 2)
        for ax, key, title in [
            (axes[0, 0], "asset", text("asset_distribution", locale)),
            (axes[0, 1], "material", text("material_distribution", locale)),
        ]:
            rows = view["distributions"][key]
            ax.bar(
                [
                    (
                        str(r["value"])
                        if r["value"] is not None
                        else text("unknown", locale)
                    )
                    for r in rows
                ],
                [r["count"] for r in rows],
                color="#0d9488",
                width=0.55,
            )
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel(text("episodes", locale))
            ax.yaxis.get_major_locator().set_params(integer=True)
        ax = axes[1, 0]
        records = view["records"]
        measured = [
            r
            for r in records
            if all(
                match_dimension(r, key, {"availability": "known"})
                for key in ("pose_x", "pose_y")
            )
        ]

        def asset_label(record: dict[str, Any]) -> str:
            if match_dimension(record, "asset", {"availability": "known"}):
                return record["dimensions"]["asset"]["value"]
            return text("unknown_asset", locale)

        for asset in sorted({asset_label(r) for r in measured}):
            group = [r for r in measured if asset_label(r) == asset]
            ax.scatter(
                [r["dimensions"]["pose_x"]["value"] for r in group],
                [r["dimensions"]["pose_y"]["value"] for r in group],
                label=asset,
                s=65,
                alpha=0.8,
            )
        if measured:
            ax.legend(fontsize=8)
        ax.set_title(text("initial_position", locale), loc="left", fontweight="bold")
        ax.set_xlabel(text("world_x", locale))
        ax.set_ylabel(text("world_y", locale))
        ax.ticklabel_format(useOffset=False)
        ax = axes[1, 1]
        durations = [
            r["dimensions"]["duration_s"]["value"]
            for r in records
            if match_dimension(r, "duration_s", {"availability": "known"})
        ]
        if durations:
            ax.hist(
                durations,
                bins=min(10, max(1, len(set(durations)))),
                color="#0d9488",
                edgecolor="white",
            )
        ax.set_title(text("trajectory_duration", locale), loc="left", fontweight="bold")
        ax.set_xlabel(text("recorded_time", locale))
        ax.set_ylabel(text("episodes", locale))
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        return _apply_plot_locale(fig, locale)


def _completeness_rows(
    view: dict[str, Any], locale: str = DEFAULT_LOCALE
) -> list[list[Any]]:
    return [
        [
            dimension_label(r["key"], locale),
            r["family"],
            r["unit"] or text("category", locale),
            r["version"],
            r["known"],
            r["unknown"],
            r["missing"],
            r["invalid"],
            (
                round(100 * r["completeness"], 1)
                if r["completeness"] is not None
                else None
            ),
            json.dumps(r["sources"], ensure_ascii=False),
        ]
        for r in view["completeness"]
    ]


def _dimension_plot(
    view: dict[str, Any], key: str, locale: str = DEFAULT_LOCALE
) -> Any:
    from matplotlib.figure import Figure
    import numpy as np

    spec = DIMENSIONS[key]
    fig = Figure(figsize=(10, 3), layout="constrained")
    ax = fig.subplots()
    records = view["records"]
    rows = view["distributions"][key]
    if spec["kind"] == "number":
        from .dimensions import match_dimension

        values = [
            r["dimensions"][key]["value"]
            for r in records
            if match_dimension(r, key, {"availability": "known"})
        ]
        if values:
            ax.hist(
                values,
                bins=min(10, max(1, len(set(values)))),
                color="#0d9488",
                edgecolor="white",
            )
        ax.set_xlabel(f"{key} ({spec['unit']})")
    else:
        ax.bar(
            [
                str(r["value"]) if r["value"] is not None else text("unknown", locale)
                for r in rows
            ],
            [r["count"] for r in rows],
            color="#0d9488",
        )
        ax.tick_params(axis="x", labelsize=8)
    row = next(r for r in view["completeness"] if r["key"] == key)
    ax.set_title(
        f"{dimension_label(key, locale)} | "
        f"{text('known', locale)}={row['known']} "
        f"{text('unknown', locale)}={row['unknown']} "
        f"{text('missing', locale)}={row['missing']} "
        f"{text('incompatible', locale)}={row['invalid']}",
        loc="left",
    )
    ax.set_ylabel(text("episodes", locale))
    ax.spines[["top", "right"]].set_visible(False)
    return _apply_plot_locale(fig, locale)


def _joint_plot(cells: list[dict[str, Any]], locale: str = DEFAULT_LOCALE) -> Any:
    from matplotlib.figure import Figure
    import numpy as np

    fig = Figure(figsize=(10, 3), layout="constrained")
    ax = fig.subplots()
    # Tokens keep literal labels distinct from unavailable values.
    xs = list(dict.fromkeys(c["x_token"] for c in cells))
    ys = list(dict.fromkeys(c["y_token"] for c in cells))
    labels = {
        "未知": text("unknown", locale),
        "缺失": text("missing", locale),
        "不兼容": text("incompatible", locale),
    }
    if cells:
        grid = np.zeros((len(xs), len(ys)))
        for c in cells:
            grid[xs.index(c["x_token"]), ys.index(c["y_token"])] = c["count"]
        ax.imshow(grid, cmap="GnBu", aspect="auto", vmin=0)
        ax.set_yticks(
            range(len(xs)),
            [
                labels.get(
                    next(c["x"] for c in cells if c["x_token"] == x),
                    next(c["x"] for c in cells if c["x_token"] == x),
                )
                for x in xs
            ],
        )
        ax.set_xticks(
            range(len(ys)),
            [
                labels.get(
                    next(c["y"] for c in cells if c["y_token"] == y),
                    next(c["y"] for c in cells if c["y_token"] == y),
                )
                for y in ys
            ],
            rotation=15,
        )
        for i, j in np.ndindex(grid.shape):
            ax.text(j, i, str(int(grid[i, j])), ha="center", va="center")
        ax.set_ylabel(dimension_label(cells[0]["x_key"], locale))
        ax.set_xlabel(dimension_label(cells[0]["y_key"], locale))
    return _apply_plot_locale(fig, locale)


def _trajectory_figure(result: dict[str, Any], locale: str = DEFAULT_LOCALE) -> Any:
    from .trajectory import trajectory_plot

    figure = trajectory_plot(result)
    if normalize_locale(locale) == "zh":
        figure.axes[0].set_ylabel("TCP 位置（m）")
        figure.axes[1].set_ylabel("TCP 速度（m/s）")
        figure.axes[2].set_ylabel("关节位置")
        figure.axes[2].set_xlabel("归一化阶段时间")
    return _apply_plot_locale(figure, locale)


def _replay_markup(url: str | None, locale: str = DEFAULT_LOCALE) -> str:
    if not url:
        placeholder = text("replay_placeholder", locale)
        return f'<div style="padding:70px;text-align:center;background:#f8fafc;border-radius:12px;color:#64748b">{placeholder}</div>'
    safe_url = html.escape(url, quote=True)
    link = text("open_viser", locale)
    return f'<iframe src="{safe_url}" style="width:100%;height:640px;border:1px solid #e2e8f0;border-radius:12px" title="Episode replay"></iframe><a href="{safe_url}" target="_blank">{link}</a>'


def _drill_view(view: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    selected = list(select_cell(view["records"], cell))
    result = _build_view(selected, view["summary"], view["pose_bins"])
    result["selected_cell"] = {k: v for k, v in cell.items() if k != "episode_ids"}
    result["selected_cells"] = [
        *view.get("selected_cells", []),
        result["selected_cell"],
    ]
    return result


def _overview(view: dict[str, Any], locale: str = DEFAULT_LOCALE) -> str:
    summary = view["summary"]
    records = view["records"]
    statuses = summary["statuses"]
    passed = sum(r["metrics"].get("physical_success") is True for r in records)
    unknown = sum(
        any(r["dimensions"].get(k, {}).get("value") is None for k in DIMENSION_KEYS)
        for r in records
    )
    cards = [
        (text("total_attempts", locale), summary["total"]),
        (text("committed", locale), statuses.get("committed", 0)),
        (text("filtered", locale), len(records)),
        (text("physical_passed", locale), passed),
        (text("unknown_dimensions", locale), unknown),
    ]
    elements = "".join(
        f'<div style="flex:1;min-width:120px;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:12px;padding:18px"><div style="font-size:13px;color:#475569">{label}</div><strong style="font-size:30px;color:#0f766e">{value}</strong></div>'
        for label, value in cards
    )
    states = html.escape(
        " · ".join(f"{key}: {value}" for key, value in statuses.items())
    )
    ledger = text("ledger", locale)
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{elements}</div><p style="color:#64748b;margin:12px 0">{ledger}: {states}</p>'


def build_app(catalog_path: str | Path) -> Any:
    """Build a local workbench; create and dispose one Viser viewer per session."""
    global gr
    import gradio as gr

    catalog_path = Path(catalog_path).resolve()
    initial = query_view(catalog_path)
    with Catalog(catalog_path) as catalog:
        all_records = catalog.records()

    def choices(key: str) -> list[str]:
        return sorted(
            {
                str(r["dimensions"].get(key, {}).get("value"))
                for r in all_records
                if match_dimension(r, key, {"availability": "known"})
            }
        )

    viewers: dict[str, Any] = {}
    viewer_lock = threading.Lock()

    category_keys = ["asset", "material", "affordance", "approach", "trajectory_family"]
    numeric_keys = ["pose_x", "pose_y", "pose_z", "yaw", "light", "duration_s"]

    def controls_to_filters(
        values: tuple[Any, ...],
    ) -> tuple[dict[str, Any], str | None, str, str]:
        categories = dict(zip(category_keys, values[: len(category_keys)]))
        offset = len(category_keys)
        bounds = {
            key: (values[offset + 2 * i], values[offset + 2 * i + 1])
            for i, key in enumerate(numeric_keys)
        }
        offset += 2 * len(numeric_keys)
        missing_key, availability, status, dimension, joint_kind = values[offset:]
        return (
            build_filters(categories, bounds, missing_key or None, availability),
            None if status == "all" else status,
            dimension,
            joint_kind,
        )

    def completeness_headers(locale: str) -> list[str]:
        return [
            text("dimension", locale),
            text("dimension_family", locale),
            text("unit_type", locale),
            text("definition_version", locale),
            text("known", locale),
            text("unknown", locale),
            text("missing", locale),
            text("incompatible", locale),
            text("completeness_rate", locale),
            text("source_counts", locale),
        ]

    def episode_headers(locale: str) -> list[str]:
        return [
            "Episode",
            text("status", locale),
            text("asset", locale),
            text("material", locale),
            text("light", locale),
            text("initial_x_m", locale),
            text("initial_y_m", locale),
            text("lift_m", locale),
            text("placement_error_m", locale),
            text("segment_count", locale),
        ]

    def comparison_choices(ids: list[str], locale: str) -> list[Any]:
        return [(text("no_comparison", locale), ""), *ids]

    def outputs(
        view: dict[str, Any],
        applied: dict[str, Any],
        dimension: str,
        joint_kind: str,
        locale: str,
    ) -> tuple[Any, ...]:
        locale = normalize_locale(locale)
        ids = view["ids"]
        selectable = [
            r["episode_id"] for r in view["records"] if "trajectory" in r["artifacts"]
        ]
        cells = view["joint_cells"][joint_kind]
        return (
            _overview(view, locale),
            _plots(view, locale),
            view["rows"],
            gr.update(choices=ids, value=ids[0] if ids else None),
            gr.update(choices=comparison_choices(selectable, locale), value=""),
            applied,
            _completeness_rows(view, locale),
            _dimension_plot(view, dimension, locale),
            _joint_plot(cells, locale),
            [[c["x"], c["y"], c["count"]] for c in cells],
            cells,
            text("slice_count", locale, count=len(ids))
            + (text("slice_drilled", locale) if "selected_cell" in view else ""),
        )

    def refresh(*values: Any) -> tuple[Any, ...]:
        try:
            locale = normalize_locale(values[-1])
            conditions, status, dimension, joint_kind = controls_to_filters(values[:-1])
            view = query_view(catalog_path, conditions, status)
            applied = {"view": view, "filters": conditions, "status": status}
            return outputs(view, applied, dimension, joint_kind, locale)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def change_joint(
        applied: dict[str, Any], kind: str, locale: str
    ) -> tuple[Any, ...]:
        cells = applied["view"]["joint_cells"][kind]
        return (
            _joint_plot(cells, locale),
            [[c["x"], c["y"], c["count"]] for c in cells],
            cells,
        )

    def drill(
        applied: dict[str, Any],
        cells: list[dict[str, Any]],
        dimension: str,
        kind: str,
        locale: str,
        event: gr.SelectData,
    ) -> tuple[Any, ...]:
        index = (
            event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        )
        if not isinstance(index, int) or not 0 <= index < len(cells):
            raise gr.Error(text("invalid_cell", locale))
        view = _drill_view(applied["view"], cells[index])
        return outputs(view, {**applied, "view": view}, dimension, kind, locale)

    def details(episode_id: str | None, locale: str) -> tuple[Any, str]:
        if not episode_id:
            return {}, text("episode_not_selected", locale)
        with Catalog(catalog_path) as catalog:
            record = catalog.get(episode_id)
        if record is None:
            return {}, text("record_missing", locale)
        metrics = record["metrics"]
        verdict = (
            text("passed", locale)
            if metrics.get("physical_success") is True
            else text("not_passed", locale)
        )
        note = (
            f"**{text('physical_check', locale)}: {verdict}** · "
            f"{text('program_success', locale)}: "
            f"{metrics.get('program_reported_success', text('unknown', locale))}\n\n"
            f"{metrics.get('physical_check', text('no_physical_evidence', locale))}"
            f"\n\n{record.get('reason') or ''}"
        )
        # The full native event payload remains in the exported record, not a huge UI tree.
        compact = {k: v for k, v in record.items() if k != "provenance"}
        compact["provenance"] = {
            k: v for k, v in record["provenance"].items() if k != "native_demo_metadata"
        }
        return compact, note

    def replay(
        episode_id: str | None,
        comparison: str,
        locale: str,
        request: gr.Request,
    ) -> tuple[str, str]:
        if not episode_id:
            raise gr.Error(text("open_episode_first", locale))
        from .replay import ReplayViewer

        with Catalog(catalog_path) as catalog:
            record = catalog.get(episode_id)
            other = catalog.get(comparison) if comparison else None
        if record is None or "trajectory" not in record["artifacts"]:
            raise gr.Error(text("no_replay", locale))
        for selected in (record, other):
            if selected:
                selected["artifacts"] = {
                    k: str((catalog_path.parent / Path(v)).resolve())
                    for k, v in selected["artifacts"].items()
                }
        key = request.session_hash
        with viewer_lock:
            viewer = viewers.get(key)
            if viewer is None:
                viewer = ReplayViewer()
                viewers[key] = viewer
            viewer.load(record, compare_record=other)
        return _replay_markup(viewer.url, locale), viewer.url

    def phases(episode_id: str | None) -> Any:
        from .trajectory import phase_options

        with Catalog(catalog_path) as catalog:
            record = catalog.get(episode_id) if episode_id else None
        options = phase_options(record) if record else ["whole"]
        return gr.update(choices=options, value="whole")

    def curves(
        episode_id: str | None, comparison: str, phase: str, locale: str
    ) -> tuple[Any, Any, dict[str, Any]]:
        from .trajectory import trajectory_comparison

        try:
            with Catalog(catalog_path) as catalog:
                record = catalog.get(episode_id) if episode_id else None
                other = catalog.get(comparison) if comparison else None
            if record is None:
                raise ValueError(text("open_episode_first", locale))
            result = trajectory_comparison(
                record, other, phase=phase, base_dir=catalog_path.parent
            )
            return _trajectory_figure(result, locale), result["summary"], result
        except (ValueError, KeyError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def cleanup(request: gr.Request) -> None:
        with viewer_lock:
            viewer = viewers.pop(request.session_hash, None)
        if viewer is not None:
            viewer.close()

    def save_slice(applied: dict[str, Any]) -> str:
        return str(
            _export_view(
                catalog_path,
                applied["view"],
                filters=applied["filters"],
                status=applied["status"],
            )
        )

    def snapshot(name: str, locale: str) -> str:
        with Catalog(catalog_path) as catalog:
            identifier = catalog.snapshot(name or "manual")
        return text(
            "snapshot_saved", locale, identifier=identifier, name=name or "manual"
        )

    def compare(a: float, b: float) -> dict[str, Any]:
        try:
            return compare_snapshots(catalog_path, int(a), int(b))
        except KeyError as exc:
            raise gr.Error(str(exc)) from exc

    locale = DEFAULT_LOCALE
    with gr.Blocks(title=text("app_title", locale)) as app:
        with gr.Row():
            header = gr.Markdown(text("header", locale))
            language = gr.Dropdown(
                LANGUAGE_CHOICES,
                value=locale,
                label=text("language", locale),
                min_width=160,
                scale=0,
            )
        intro = gr.Markdown(text("intro", locale))
        applied = gr.State({"view": initial, "filters": {}, "status": "committed"})
        overview = gr.HTML(_overview(initial, locale))
        category_controls = []
        range_controls = []
        with gr.Accordion(text("filters", locale), open=True) as filters_accordion:
            with gr.Row():
                for key in category_keys:
                    category_controls.append(
                        gr.Dropdown(
                            choices(key),
                            value=[],
                            multiselect=True,
                            label=dimension_label(key, locale),
                        )
                    )
            with gr.Accordion(text("ranges", locale), open=False) as ranges_accordion:
                for key in numeric_keys:
                    with gr.Row():
                        label = f"{dimension_label(key, locale)} / {DIMENSIONS[key]['unit']}"
                        range_controls.extend(
                            [
                                gr.Number(
                                    value=None,
                                    label=f"{label} · {text('minimum', locale)}",
                                ),
                                gr.Number(
                                    value=None,
                                    label=f"{label} · {text('maximum', locale)}",
                                ),
                            ]
                        )
            with gr.Row():
                availability_dimension = gr.Dropdown(
                    [(text("any_dimension", locale), "")] + dimension_choices(locale),
                    value="",
                    label=text("availability_dimension", locale),
                )
                availability = gr.Dropdown(
                    availability_choices(locale),
                    value="all",
                    label=text("availability_state", locale),
                )
                status = gr.Dropdown(
                    status_choices(locale),
                    value="committed",
                    label=text("collection_status", locale),
                )
                update = gr.Button(text("apply_filters", locale), variant="primary")
        selection_note = gr.Markdown(
            text("slice_count", locale, count=len(initial["ids"]))
        )
        with gr.Tabs():
            with gr.Tab(text("tab_distribution", locale)) as distribution_tab:
                with gr.Accordion(
                    text("definitions", locale), open=False
                ) as definitions_accordion:
                    completeness = gr.Dataframe(
                        headers=completeness_headers(locale),
                        value=_completeness_rows(initial, locale),
                        interactive=False,
                    )
                dimension = gr.Dropdown(
                    dimension_choices(locale),
                    value="affordance",
                    label=text("dimension_distribution", locale),
                )
                dimension_plot = gr.Plot(_dimension_plot(initial, "affordance", locale))
                with gr.Accordion(
                    text("dataset_overview", locale), open=False
                ) as overview_accordion:
                    plot = gr.Plot(
                        _plots(initial, locale),
                        label=text("slice_distribution", locale),
                    )
                distribution_note = gr.Markdown(text("distribution_note", locale))
                joint_kind = gr.Dropdown(
                    joint_choices(locale),
                    value="material_light",
                    label=text("joint_statistics", locale),
                )
                active_cells = gr.State(initial["joint_cells"]["material_light"])
                joint_plot = gr.Plot(
                    _joint_plot(initial["joint_cells"]["material_light"], locale)
                )
                drill_note = gr.Markdown(text("drill_note", locale))
                joint_table = gr.Dataframe(
                    headers=[
                        text("dimension_a", locale),
                        text("dimension_b", locale),
                        text("episode_count", locale),
                    ],
                    value=[
                        [c["x"], c["y"], c["count"]]
                        for c in initial["joint_cells"]["material_light"]
                    ],
                    interactive=False,
                    label=text("joint_cells", locale),
                )
                table = gr.Dataframe(
                    headers=episode_headers(locale),
                    value=initial["rows"],
                    interactive=False,
                    label=text("current_slice", locale),
                    max_height=320,
                )
                with gr.Row():
                    export = gr.Button(text("export_slice", locale))
                    exported = gr.File(label=text("slice_json", locale))
            with gr.Tab(text("tab_replay", locale)) as replay_tab:
                with gr.Row():
                    episode = gr.Dropdown(
                        initial["ids"],
                        value=initial["ids"][0] if initial["ids"] else None,
                        label="Episode",
                    )
                    comparison = gr.Dropdown(
                        comparison_choices(initial["ids"], locale),
                        value="",
                        label=text("comparison_episode", locale),
                    )
                    play = gr.Button(text("open_replay", locale), variant="primary")
                preview = gr.HTML(_replay_markup(None, locale))
                replay_url = gr.State(None)
                with gr.Accordion(
                    text("phase_section", locale), open=True
                ) as phase_accordion:
                    with gr.Row():
                        phase = gr.Dropdown(
                            ["whole"],
                            value="whole",
                            label=text("phase", locale),
                        )
                        analyze = gr.Button(text("analyze", locale), variant="primary")
                    trajectory_note = gr.Markdown(text("trajectory_note", locale))
                    trajectory_chart = gr.Plot(label=text("trajectory_chart", locale))
                    trajectory_metrics = gr.JSON(
                        label=text("trajectory_metrics", locale)
                    )
                    trajectory_result = gr.State(None)
                evidence = gr.Markdown()
                with gr.Accordion(
                    text("record_evidence", locale), open=False
                ) as evidence_accordion:
                    record_json = gr.JSON()
            with gr.Tab(text("tab_versions", locale)) as versions_tab:
                version_note = gr.Markdown(text("version_note", locale))
                with gr.Row():
                    name = gr.Textbox(
                        value="acceptance-review", label=text("snapshot_name", locale)
                    )
                    save = gr.Button(text("save_snapshot", locale))
                snapshot_status = gr.Markdown()
                with gr.Row():
                    before = gr.Number(
                        value=1, precision=0, label=text("baseline_snapshot", locale)
                    )
                    after = gr.Number(
                        value=2,
                        precision=0,
                        label=text("comparison_snapshot", locale),
                    )
                    compare_button = gr.Button(text("compare_versions", locale))
                diff = gr.JSON(label=text("version_diff", locale))
            with gr.Tab(text("tab_scope", locale)) as scope_tab:
                scope = gr.Markdown(text("scope", locale))

        def set_language(
            selected_locale: str,
            applied_state: dict[str, Any],
            dimension_key: str,
            joint_kind_value: str,
            episode_id: str | None,
            trajectory_state: dict[str, Any] | None,
            replay_state: str | None,
        ) -> tuple[Any, ...]:
            selected_locale = normalize_locale(selected_locale)
            view = applied_state["view"]
            cells = view["joint_cells"][joint_kind_value]
            trajectory_update = gr.update(
                label=text("trajectory_chart", selected_locale)
            )
            if trajectory_state:
                trajectory_update = gr.update(
                    value=_trajectory_figure(trajectory_state, selected_locale),
                    label=text("trajectory_chart", selected_locale),
                )
            detail_note = details(episode_id, selected_locale)[1]
            range_updates = []
            for key in numeric_keys:
                base = f"{dimension_label(key, selected_locale)} / {DIMENSIONS[key]['unit']}"
                range_updates.extend(
                    [
                        gr.update(label=f"{base} · {text('minimum', selected_locale)}"),
                        gr.update(label=f"{base} · {text('maximum', selected_locale)}"),
                    ]
                )
            return (
                text("header", selected_locale),
                text("intro", selected_locale),
                gr.update(label=text("language", selected_locale)),
                _overview(view, selected_locale),
                gr.update(label=text("filters", selected_locale)),
                *[
                    gr.update(label=dimension_label(key, selected_locale))
                    for key in category_keys
                ],
                gr.update(label=text("ranges", selected_locale)),
                *range_updates,
                gr.update(
                    choices=[(text("any_dimension", selected_locale), "")]
                    + dimension_choices(selected_locale),
                    label=text("availability_dimension", selected_locale),
                ),
                gr.update(
                    choices=availability_choices(selected_locale),
                    label=text("availability_state", selected_locale),
                ),
                gr.update(
                    choices=status_choices(selected_locale),
                    label=text("collection_status", selected_locale),
                ),
                gr.update(value=text("apply_filters", selected_locale)),
                text("slice_count", selected_locale, count=len(view["ids"]))
                + (
                    text("slice_drilled", selected_locale)
                    if "selected_cell" in view
                    else ""
                ),
                gr.update(label=text("tab_distribution", selected_locale)),
                gr.update(label=text("definitions", selected_locale)),
                gr.update(
                    headers=completeness_headers(selected_locale),
                    value=_completeness_rows(view, selected_locale),
                ),
                gr.update(
                    choices=dimension_choices(selected_locale),
                    label=text("dimension_distribution", selected_locale),
                ),
                _dimension_plot(view, dimension_key, selected_locale),
                gr.update(label=text("dataset_overview", selected_locale)),
                gr.update(
                    value=_plots(view, selected_locale),
                    label=text("slice_distribution", selected_locale),
                ),
                text("distribution_note", selected_locale),
                gr.update(
                    choices=joint_choices(selected_locale),
                    label=text("joint_statistics", selected_locale),
                ),
                _joint_plot(cells, selected_locale),
                text("drill_note", selected_locale),
                gr.update(
                    headers=[
                        text("dimension_a", selected_locale),
                        text("dimension_b", selected_locale),
                        text("episode_count", selected_locale),
                    ],
                    value=[[c["x"], c["y"], c["count"]] for c in cells],
                    label=text("joint_cells", selected_locale),
                ),
                gr.update(
                    headers=episode_headers(selected_locale),
                    value=view["rows"],
                    label=text("current_slice", selected_locale),
                ),
                gr.update(value=text("export_slice", selected_locale)),
                gr.update(label=text("slice_json", selected_locale)),
                gr.update(label=text("tab_replay", selected_locale)),
                gr.update(
                    choices=comparison_choices(view["ids"], selected_locale),
                    label=text("comparison_episode", selected_locale),
                ),
                gr.update(value=text("open_replay", selected_locale)),
                _replay_markup(replay_state, selected_locale),
                gr.update(label=text("phase_section", selected_locale)),
                gr.update(label=text("phase", selected_locale)),
                gr.update(value=text("analyze", selected_locale)),
                text("trajectory_note", selected_locale),
                trajectory_update,
                gr.update(label=text("trajectory_metrics", selected_locale)),
                detail_note,
                gr.update(label=text("record_evidence", selected_locale)),
                gr.update(label=text("tab_versions", selected_locale)),
                text("version_note", selected_locale),
                gr.update(label=text("snapshot_name", selected_locale)),
                gr.update(value=text("save_snapshot", selected_locale)),
                gr.update(label=text("baseline_snapshot", selected_locale)),
                gr.update(label=text("comparison_snapshot", selected_locale)),
                gr.update(value=text("compare_versions", selected_locale)),
                gr.update(label=text("version_diff", selected_locale)),
                gr.update(label=text("tab_scope", selected_locale)),
                text("scope", selected_locale),
            )

        inputs = (
            category_controls
            + range_controls
            + [
                availability_dimension,
                availability,
                status,
                dimension,
                joint_kind,
                language,
            ]
        )
        all_outputs = [
            overview,
            plot,
            table,
            episode,
            comparison,
            applied,
            completeness,
            dimension_plot,
            joint_plot,
            joint_table,
            active_cells,
            selection_note,
        ]
        update.click(refresh, inputs, all_outputs)
        dimension.change(
            lambda state, key, selected_locale: _dimension_plot(
                state["view"], key, selected_locale
            ),
            [applied, dimension, language],
            dimension_plot,
        )
        joint_kind.change(
            change_joint,
            [applied, joint_kind, language],
            [joint_plot, joint_table, active_cells],
        )
        joint_table.select(
            drill,
            [applied, active_cells, dimension, joint_kind, language],
            all_outputs,
        )
        episode.change(phases, episode, phase)
        app.load(phases, episode, phase)
        analyze.click(
            curves,
            [episode, comparison, phase, language],
            [trajectory_chart, trajectory_metrics, trajectory_result],
        )
        export.click(save_slice, applied, exported)
        episode.change(details, [episode, language], [record_json, evidence])
        app.load(details, [episode, language], [record_json, evidence])
        play.click(replay, [episode, comparison, language], [preview, replay_url])
        save.click(snapshot, [name, language], snapshot_status)
        compare_button.click(compare, [before, after], diff)
        language_outputs = [
            header,
            intro,
            language,
            overview,
            filters_accordion,
            *category_controls,
            ranges_accordion,
            *range_controls,
            availability_dimension,
            availability,
            status,
            update,
            selection_note,
            distribution_tab,
            definitions_accordion,
            completeness,
            dimension,
            dimension_plot,
            overview_accordion,
            plot,
            distribution_note,
            joint_kind,
            joint_plot,
            drill_note,
            joint_table,
            table,
            export,
            exported,
            replay_tab,
            comparison,
            play,
            preview,
            phase_accordion,
            phase,
            analyze,
            trajectory_note,
            trajectory_chart,
            trajectory_metrics,
            evidence,
            evidence_accordion,
            versions_tab,
            version_note,
            name,
            save,
            before,
            after,
            compare_button,
            diff,
            scope_tab,
            scope,
        ]
        language.change(
            set_language,
            [
                language,
                applied,
                dimension,
                joint_kind,
                episode,
                trajectory_result,
                replay_url,
            ],
            language_outputs,
        )
        app.unload(cleanup)
    return app
