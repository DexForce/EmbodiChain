# Assembly experiment evidence

This directory retains the measured conclusions, primary scored episode ledgers
and independent reference inputs. The default-asset experiment and its current
preview are in the [learning example](../../../learning/blind_assembly/validation/default_retrain_20261026/RESULTS.md).
Generated logs, repeated health snapshots, intermediate selection records and
duplicate media belong under `outputs/`, rather than in the current source tree.
Reusable benchmark runners and their reference inputs remain in the repository.

The complete evidence published before cleanup remains available at immutable
commit `f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0`:

- [Physics probes, passive demos and historical nominal RL evidence](https://github.com/DexForce/EmbodiChain/tree/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation).
- [Default-asset transfer and retraining evidence](https://github.com/DexForce/EmbodiChain/tree/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation).

Links from the reports to archived artifacts use that commit. They include
failed experiments and numerical warnings; cleanup does not revise any measured
score or qualification. To inspect an archived file locally, use
`git show f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0:<repository-relative-path>`.

The local download `EmbodiChain_assembly_evidence_before_cleanup_20261027.tar.gz`
contains all 156 original files (19,964,731 uncompressed bytes) and a per-file
SHA-256 manifest. All archive entries were verified before cleanup. Its compressed
size is 16,031,564 bytes and SHA-256 is
`648eba94ccc4c575219d69cd3d20696b3872fe5033b1771be5c8ed552ee26293`.
The [machine-readable archive descriptor](archive.json) records these identities.
The archive is a local deliverable, not a hosted dataset release. The already
delivered model and video downloads remain unchanged.

Keep future run output in an ignored `outputs/` directory. Commit concise
results, necessary reference fixtures and primary evidence deliberately; the
local ignore rules prevent experiment scratch files being added accidentally.
