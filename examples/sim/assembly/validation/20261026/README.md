# Native assembly validation — 2026-10-26

These committed records distinguish geometry, physical behavior and reference
initialization. They do not qualify complete paper reproduction or RL learning.
The scripts and pinned dependency instructions are in the parent example README
and `scripts/patches/`. Large model/trajectory artifacts remain local outputs;
these summaries retain profile settings and measured failures. Full generated
records and media links resolve to the [immutable evidence archive](../ARCHIVE.md).

- [Many-pegs report](many_pegs.md) and [machine-readable results](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/20261026/many_pegs.json).
- [Gear report](gears.md) and [machine-readable results](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/20261026/gears.json).

The official 25-cell primitive grid retains all 50 bodies and 1050 collision
pieces. At 0.25 ms, all 25 pegs seat. For the controlled 200 µm comparison, matching
Newton's effective solver options and GPU-derived constant initialization yields
a maximum endpoint difference of 0.784861 µm; this is an endpoint diagnostic, not
a quasistatic gap oracle. Refined 20/50/100 µm geometries still expose a thin
convex-planarity rejection or nonfinite state, with failures preserved.

Three gear pairs pass 36 native short-run diagnostic cells plus independent
property checks. The original CPU-compiled-constant Warp reference has 0/3 passes;
public `set_const_0` initialization matching Newton produces 3/3 passing
independent references. Both protocols remain recorded, with different derived
inverse weights explicitly disclosed. PGS remains a separate CPU profile.

The GIFs use native DexSim Camera RGB matching the physical window view; the SDK
window itself lacks pixel readback. Recorded physics is never replaced by an
ideal gear relation or prescribed follower pose.

![Native many-pegs preview](https://raw.githubusercontent.com/DexForce/EmbodiChain/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/20261026/many_pegs.gif)

![Native contact-driven gears](https://raw.githubusercontent.com/DexForce/EmbodiChain/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/20261026/gears.gif)
