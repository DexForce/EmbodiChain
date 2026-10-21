# Shared benchmark reporting

Reserved for offline analysis and technical-report building proposed in
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679): aggregation,
uncertainty, compatible paired comparisons, tables, shared figure styling,
provenance, plotting-data exports, and legacy result conversion.

This directory contains documentation only. Future report builders consume
frozen run IDs and plotting configurations without a simulator or GPU. Export
SVG/PDF figures with their plotting data and source-run references. Reject
speedup calculations across incompatible conditions and preserve missing,
unsupported, failed and not-run cells with their reasons.

Keep current three-table/leaderboard reports working as a legacy preset when
domain-specific templates are implemented. New loading/rendering templates
and benchmark-skill changes belong to that implementation, not this scaffold.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
