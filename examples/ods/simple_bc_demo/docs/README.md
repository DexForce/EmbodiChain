# Demo evidence and validation boundaries

[Run the local demo](../README.md). The records below were produced before the
task and learner moved into `examples/ods/simple_bc_demo/`. They retain the
executed implementation revisions, original filenames, absolute artifact paths,
SHA256 values, outcomes and limitations. The JSON payloads were moved verbatim;
their old `examples/data_pipeline/` and official task paths identify historical
execution rather than current entry points. Local `/tmp/` paths describe the
validation host and are not portable download links.

| Record | Executed source / scope |
| --- | --- |
| [Current local-package validation](PY311_VALIDATION.json) | `d54cfc8a`: conda py311, current public sources, online/offline/resume and independent student execution |
| [Working policy](WORKING_SHOWCASE.md), [JSON](WORKING_SHOWCASE_RESULTS.json) | `b683c4c699a71562341d4b213d06857f0fbdf744`: 40/40 expert qualification, selected checkpoint 99/100, later refresh checkpoint 65/100; both outcomes retained |
| [Quota tests](ODS_QUOTA_TESTS.md), [JSON](ODS_QUOTA_TEST_RESULTS.json) | `6d2ff529` learning trials and `69d65ecd` CPU-device fix: strict source caps, exploratory rate/learning regimes and separate repeat |
| [Configurator validation](ODS_CONFIGURATION_VALIDATION.json) | `8748aa22000377b7366d5709341eeeaec66cfad4`: CPU contracts, bounded two-generation smoke and scoped sampler costs |
| [Pilot protocol](OFFLINE_PILOT.md), [results](OFFLINE_PILOT_RESULTS.md), [JSON](OFFLINE_PILOT_RESULTS.json) | `3cd6379a092f8106f08ea6073f13bec9a4bd6f63`: one-seed, four-arm acquisition-inclusive comparison; no reserved tests opened |
| [Accounting audit](ODS_ACCOUNTING_AUDIT.md), [JSON](ODS_ACCOUNTING_AUDIT.json) | Original pilot baseline and subsequent counter implementation: exact correctness; final C32 median overhead 5.127% leaves the strict 5% gate incomplete |
| [Online/offline alignment](ODS_ALIGNMENT_VALIDATION.json) | Base `ecf47fc431234c2119281038e99367b2fec85861` plus the recorded uncommitted fix: exact CPU batches/optimizer updates and real expert-producer integration |
| [Public loading validation](DATASET_LOADING_VALIDATION.json) | Base `6d9e1708292131eb778d7f50b808682026d599d6` plus recorded changes; pre-extraction golden cases, numeric API/SDK checks and physical integrations; later consolidation and example-test removal are separately identified |

[CONFIGURATION.md](CONFIGURATION.md) describes the current local CLI and the
measurement/profile method. Its historical counts and measurements are labeled
separately. The shared [loading design](../../../../docs/architecture/dataset-loading-design.md)
owns the public numeric Dataset/source contract.

The historical checks establish the behavior of their recorded revisions.
They do not assert that the relocated implementation has rerun the same learning
campaigns. Earlier example-model/writer/optimizer unit tests were removed;
production ODS and dataset tests remain maintained. The exact historical test
counts remain visible in the records and are not current CI counts.

Policy success, source parity, bounded reuse and efficiency have separate
acceptance criteria. The pilot's offline32 baseline reached its first evaluated
qualifying checkpoint earlier than ODS. Sampled memory peaks, feature-conversion
placement, instrumentation scope and every reported regression remain part of
the evidence. Numerical parity and a working smoke do not resolve the full-path
telemetry gate or establish an ODS learning/throughput advantage.

Development scenes 810000–810019 and the working-policy reserved scenes
50000–50099 were exposed by the recorded campaigns. Their reuse cannot provide
fresh held-out confirmation for a later study.
