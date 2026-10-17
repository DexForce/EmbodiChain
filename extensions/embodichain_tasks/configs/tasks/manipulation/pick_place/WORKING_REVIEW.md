# Working showcase change review

Review target: the local working change based on `1c9897b4`, including the new
chunk policy and tests. The lead verified integration and experiment evidence;
three independent agents performed the selected read-only expert passes under
`.agents/skills/review-pr/SKILL.md`.

Two actionable findings were accepted and corrected before completion:

1. **P2 — Comparability did not include chunk settings and output normalization.**
   Action horizon and observation noise could change behavior without changing
   the old parameter/input-normalization fingerprints. The reporter now compares
   both horizons and noise, hashes input/target normalization and requires
   explicit evidence that all four buffers were included. Older incomplete
   chunk artifacts cannot support a comparability headline.
2. **P3 — TensorBoard throughput counted contexts instead of supervised targets.**
   The primary efficiency tracker already counted actual loss-mask targets. The
   TensorBoard window now sums those actual target counts instead of assuming
   one target per context. The resume smoke checked the scalar against the
   saved timing/target records: 129.53639 targets/s by both computations.

| Priority | Location | Defect | Impact | Action |
| --- | --- | --- | --- | --- |
| P2 | `_ods_pick_place_training.py`, `ods_pick_place_compare.py` | Incomplete effective-policy fingerprint | Could compare unlike policies | Resolved; negative comparison/hash tests pass |
| P3 | `_ods_pick_place_training.py` | Contexts mistaken for targets in one scalar | Underreported chunk throughput | Resolved; actual resume audit matches |
| N/A | N/A | No remaining actionable finding | N/A | N/A |

| Review item | Result |
| --- | --- |
| Target | Working changes from `1c9897b4` |
| Scope | ODS context sampling, private BC policy/runner/reporter, teacher yaw, tests, docs/context |
| Execution | Multi-agent expert review plus lead integration/API verification |
| Open findings | P0: 0; P1: 0; P2: 0; P3: 0 |
| Assessment | No remaining actionable findings in reviewed scope |
| Validation | 283 focused tests; API coverage 2604/2604; context check; actual generation/training/evaluation/resume/video |
| Residual limits | One/four-demo diagnostics fail; later checkpoint regresses; single training seed; no efficiency claim; native teardown warning retained |

| Role | Scope | Executor | Status / evidence | Candidate disposition |
| --- | --- | --- | --- | --- |
| engine | Locking, publication/provenance, causal spans, padding, errors, sample accounting; verification lens | `/root/v4_engine_review` | Complete; 28 sampling/history and 34 corpus tests; extra CPU boundary/failure probes | None |
| ai | Causal inputs/labels, target representation, masks, normalization, checkpoint/resume, comparison accounting; verification lens | `/root/v4_ai_review` | Complete; 38 policy/comparison checks after fixes; numerical rotation roundtrip | `ai-1` and `ai-2` accepted and resolved |
| robotics | Frames/scales/rotations, grasp symmetry, live relative controller, student execution; verification lens | `/root/v4_robotics_review` | Complete; policy tests and 1,000 nonidentity-rotation roundtrips, maximum error 5.96e-6 | None |
| api / integration | CLI/default recipe, compatibility, docs/context, actual subprocess execution and shutdown | Lead | Complete; source review, 283 combined checks and retained runtime artifacts | Offline small-data cardinality guard corrected and covered |

The independent reviewers did not run the simulation experiments. Runtime
qualification and held-out results were produced by the lead and are recorded
in [the experiment report](WORKING_SHOWCASE.md). No remote review or approval
was published. Production simulation/planning code beyond the task's grasp
orientation was unchanged; additional architecture/security passes were not
selected because no new dependency, external trust or remote execution boundary
was introduced.
