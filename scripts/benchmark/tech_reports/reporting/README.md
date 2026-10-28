# Shared benchmark reporting

Reserved for offline analysis and technical-report building proposed in
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679): aggregation,
uncertainty, compatible paired comparisons, tables, shared figure styling,
provenance, plotting-data exports, and legacy result conversion.

The current suite contains experiment-specific offline tools and a local HTML
builder. See the [draft snapshot](releases/simulation-report-2026-10-28/README.md)
for data setup and validation; runtime portability is not yet complete.
The planned common report builders consume
frozen run IDs and plotting configurations without a simulator or GPU. Export
SVG/PDF figures with their plotting data and source-run references. Reject
speedup calculations across incompatible conditions and preserve missing,
unsupported, failed and not-run cells with their reasons.

Keep current three-table/leaderboard reports working as a legacy preset when
domain-specific templates are implemented. New loading/rendering templates
and benchmark-skill changes belong to that implementation, not this scaffold.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
