# Shared benchmark reporting

Reserved for offline analysis and technical-report building proposed in
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679): aggregation,
uncertainty, compatible paired comparisons, tables, shared figure styling,
provenance, plotting-data exports, and legacy result conversion.

The [DexUni legacy readers](dexuni/README.md) have been migrated here for
pilot diagnosis. They are not the future qualified report builder and retain
explicitly documented limitations. Future report builders consume
frozen run IDs and plotting configurations without a simulator or GPU. Export
SVG/PDF figures with their plotting data and source-run references. Reject
speedup calculations across incompatible conditions and preserve missing,
unsupported, failed and not-run cells with their reasons.

Keep current three-table/leaderboard reports working as a legacy preset when
domain-specific templates are implemented. New loading/rendering templates
and benchmark-skill changes belong to that implementation, not this scaffold.
See the [benchmark layout](../README.md#technical-report-comparison-layout).

## DexUni maintained tools

Run from the repository root with the project's Python environment:

```bash
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.build_overview
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.report_c02_performance
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.build_c02_iteration_comparison
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.build_c02_comparison
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.verify_c02_scene --run <retained-run>
```

These tools consume local retained results under
`outputs/benchmarks/tech_reports/dexuni/`; report/video builders rewrite derived
outputs there. They do not run physics. C02's retained 24×12 recorder is maintained
at `physics/platforms/dexuni/isaaclab/c02_record_24x12.py`; measurement wrappers
are `physics/experiments/dexuni/c02_measure_24x12.py` and `c02_probe.py`.
Run those platform commands only in their matching simulation environment.
Historical copies under results preserve original run provenance.
