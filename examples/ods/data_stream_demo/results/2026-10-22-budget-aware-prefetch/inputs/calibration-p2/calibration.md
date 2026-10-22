# Three-stage calibration method

Measure the accepted generator rate γ and the SDK episode write/encode busy
rate ρ independently. Preserve the all-generation rate for compatibility, but
use post-initial generations for effective supply and P selection: the initial
fill includes simulation startup. Report the later-generation service count;
the default 40 updates may observe only one later generation.
 SDK create/finalize and queue waits remain separate
costs. Effective data service is min(γ, ρ). Pure model-update rate is retained;
the hot learner rate ν instead includes batch preparation, transfer and real
synchronized updates, excluding data/quota waits. Merge overlapping intervals
within each service; never sum independent stage times into total wall time.

At fixed batch B and protected lifetime episode budget K, the supply ceiling
is min(γ,ρ) K/B updates/s. Demand/supply is ν divided by that ceiling. The
implied ceil(ν B/min(γ,ρ)) is a diagnostic, not permission or a recommendation
to raise K. A demand/supply ratio above one indicates expected data waiting.

Consume the initial cache and at least eight later-generation updates before
using a candidate for P selection; this is a short service-capacity estimate,
not a steady-state end-to-end throughput guarantee. Explicit shorter calibrations remain visible and are
excluded from selection. With the same hardware, source and recipe, select
the smallest measured P reaching 90% of the best effective data rate. Do not
compare different camera modes, hardware placements or training recipes.
