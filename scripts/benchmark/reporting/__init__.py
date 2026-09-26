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

"""Offline, domain-neutral benchmark aggregation and comparison rules."""

from __future__ import annotations

from .aggregation import MetricDefinition, aggregate_attempts, aggregate_runs
from .comparison import ComparisonResult, compare_metric, comparison_reasons
from .compat import convert_legacy_rows, normalize_legacy_row
from .figures import FigureStyle, write_metric_csv, write_svg_bar_chart
from .report import rebuild_summary, write_technical_report
from .tables import render_markdown_table

__all__ = [
    "ComparisonResult",
    "FigureStyle",
    "MetricDefinition",
    "aggregate_attempts",
    "aggregate_runs",
    "compare_metric",
    "comparison_reasons",
    "convert_legacy_rows",
    "normalize_legacy_row",
    "rebuild_summary",
    "render_markdown_table",
    "write_metric_csv",
    "write_svg_bar_chart",
    "write_technical_report",
]
