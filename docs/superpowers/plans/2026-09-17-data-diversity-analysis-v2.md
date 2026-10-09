# Six-dimension analysis and trajectory comparison

> **For agentic workers:** Use superpowers:subagent-driven-development. User approved this continuation after the proposed next-step plan. Work in the existing feature worktree without merging or pushing.

**Goal:** Extend the runnable preview with six-family filters, definitions/completeness/source summaries, three joint views and exact cell drilldown, plus phase-aligned TCP/joint/speed plots.

**Architecture:** Pure offline analysis functions own definitions, typed filter construction, completeness, bins and cell membership. Existing Catalog and frozen applied-view exports remain authoritative. A separate trajectory module reads numeric recordings and matches task segments by name and occurrence, preserving original physical timestamps for velocity and duration.

**Tech Stack:** Existing NumPy, Matplotlib, Polars, SQLite, Gradio, Viser; no new dependencies.

**Spec:** ../specs/2026-09-17-data-diversity-p0-p1-acceptance.md, scoped to P0-01 / P1-01–03 / P1-05 improvements authorized in chat. Full geometric clustering and recipe generation remain deferred.

## Acceptance
- Every registered dimension has definition version, label, family, value kind and expected units. Completeness counts known, unknown, absent and incompatible values independently; sources remain explicit. Existing historical records are not rewritten.
- Asset/material/Affordance/approach/trajectory family categorical multiselect; numeric pose XYZ/yaw, light and duration bounds; unknown-only selection; no empty range silently accepted.
- Asset × binned pose X, Affordance × approach, material × light show counts and exact episode membership, including unknowns. Selecting a joint row refines the frozen applied population; export and charts match its IDs.
- Trajectory comparison matches named segments and occurrence, never silently substitutes an unmatched phase. Uses recorded times for speed; normalized phase time is only the horizontal comparison axis.
- Retimed identical geometric paths have zero geometric difference but different speed/duration; a bent path has nonzero difference. Compare joint values only for matching robot and joint dimensions; clearly label unavailability.
- Existing 12 real episodes support all UI paths; fixtures verify unknowns, bins including final edge, invalid bounds, phase offsets, shape/timing separation.

## Tasks
- [x] Root: add dimensions.py and focused tests first; keep storage contract compatible. Implement summaries, filters, joint membership and unknown handling.
- [x] Delegate trajectory.py + test_trajectory.py: pure phase extraction/comparison/plots, meaningful synthetic tests and real NPZ check. No simulator imports.
- [x] Root: extend ui.py with multi-select/range/unknown controls, definitions table, distribution selector and joint row drilldown; integrate trajectory functions after API handoff. Preserve snapshot/export behavior.
- [x] Run focused/regression tests, Gradio build and browser verification; update context/API docs and v2 acceptance report. Independent review and fix concrete findings.

Results and remaining full-P0/P1 scope: [v2 acceptance report](../reports/2026-09-17-data-diversity-analysis-v2.md).
