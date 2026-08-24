# Declarative Expert Program Rollout Report

This is a deterministic, static Phase 8 snapshot of checked-in framework and integration code. It does not run simulation, report physical acceptance, or certify production readiness for an embodiment.

## Framework Contract Matrix

`framework-tested` describes the reusable framework contract only. A task appears in the matrix below only when its integration/production code is checked in; that code status does not imply physical acceptance.

| Capability | Framework status | Integration gate | Scope |
| --- | --- | --- | --- |
| Pick + Place(at) | framework-tested | per-embodiment integration | Typed goals, compilation, execution, and terminal effects are covered. |
| Physical attach/release evidence | framework-tested | per-embodiment integration | Effects require live constraint and object-to-endpoint pose evidence. |
| Slide | framework-tested | per-embodiment integration | Typed handle geometry, grasping, and axis-constrained motion are covered. |
| Articulation joint validator | framework-tested | per-task integration | Measured joint-state application acceptance is covered. |
| Schema-v2 sequential | framework-tested | per-task integration | Ordered call execution and failure propagation are covered. |
| HandOver | framework-tested | integration-required | No landed task integration is claimed by this report. |
| Place relation (on/inside) | framework-tested | integration-required | Embodiment frames and relation validators must be supplied. |
| Registered call | framework-tested | integration-required | Production registration must declare and validate its concrete contract. |
| Schema-v2 parallel | framework-tested | integration-required | Fail-closed by default; production use requires an authoritative validator. |

Parallel execution remains fail-closed by default. Resource declarations alone do not authorize production concurrency; the selected embodiment must provide an authoritative validator.

## Checked-in Integration Matrix

Only the two checked-in vertical slices below are classified as integration/production code. Physical acceptance is tracked separately.

| Embodiment | Task | Skill contract | Terminal effect | Program schema | Code status | Physical acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| UR5 | Cube Pick + Place | Pick + Place(at) | two-finger constraint + pose relation | schema-v2 sequential | checked in | pending: one cycle passed; full three-cycle gate remains |
| CobotMagic | Open Drawer | Registered call -> Slide | articulation joint validator | schema-v2 sequential | checked in | pending: rerun the supported-simulation validator gate |

HandOver, Place relations (`on`/`inside`), and schema-v2 parallel are framework-tested but integration-required. They are intentionally not listed as checked-in integrations.

Both checked-in environment classes have zero task-local motion or demo-generation overrides; `test_task_classes_do_not_override_motion_or_demo_generation` keeps that structural metric at zero.

## Migration Size Snapshot

The baseline is a fixed, manually recorded pre-migration snapshot: Cube is 598 lines / 23912 bytes and Drawer is 245 lines / 8833 bytes. The tool does not inspect Git history. Current values are recomputed only from the four explicit files in the table.

Baseline identity: Cube uses Git blob `1965563b060d1fc889f03ad13d47655c2edcd99b` and Drawer uses Git blob `3b4cbdc09537098b4f109d46efb8785b88f31ce1` at each task's Python path listed in the current-source column. Blob IDs remain stable across stack rebases.

Counting rule: `lines` is the number of raw LF (`0x0A`) bytes; `bytes` is the raw on-disk byte length. Counts are summed per task without normalizing encoding or line endings.

| Task | Baseline lines | Current lines | Line delta | Baseline bytes | Current bytes | Byte delta | Current source files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cube | 598 | 237 | -361 (-60.4%) | 23912 | 7785 | -16127 (-67.4%) | `embodichain_tasks/embodichain_tasks/multi_segments/cube_pick_place.py`<br>`embodichain_tasks/configs/expert_program/multi_segments/repeated_cube_pick_place.yaml` |
| Drawer | 245 | 411 | +166 (+67.8%) | 8833 | 15204 | +6371 (+72.1%) | `embodichain_tasks/embodichain_tasks/tableware/open_drawer.py`<br>`embodichain_tasks/configs/expert_program/tableware/open_drawer.json` |
| Total | 843 | 648 | -195 (-23.1%) | 32745 | 22989 | -9756 (-29.8%) | the four files above |

## Demo Success Measurement

`scripts/benchmark/expert_program/demo_success.py` executes each fixed seed exactly once, always discards the episode buffer, and counts executor exceptions as failed rows. It writes raw JSON plus a three-table Markdown report. Its CLI supports offline raw-JSON re-aggregation and an explicit `--run-simulation` mode that constructs one standard Gym environment from Gym and Expert Program configurations.

No success-rate result or release gate is checked in yet. The main-API migration requires a fresh supported-simulation Open Drawer validator run, while repeated Cube still needs three-cycle physical acceptance before a fixed-seed rate is meaningful.

## Drift Check

Regenerate the checked-in report after an intentional source or capability snapshot change:

```bash
python scripts/tools/expert_program_rollout_report.py
```

CI and local validation can reject stale output without rewriting it:

```bash
python scripts/tools/expert_program_rollout_report.py --check
```
