# Reusable online and offline training-data loading

Status: proposed public interfaces; the current change implements PickPlace
alignment checks and fixes, not the public loader API. Updated 2026-10-19.

## Problem and acceptance criterion

Public `OnlineDataset` wraps `OnlineDataEngine.sample_batch()`. PickPlace calls
`sample_batch_with_context()` directly and implements resident offline context
sampling in its private `FrozenCorpus`. The LeRobot reader is also example-owned
and requires the PickPlace sidecar. Public offline inspection and depth readers
already exist, but neither provides a general training-sequence contract.

The required invariant is: the same source frames, schema, window and transform
produce the same training tensors, regardless of live or persisted storage.
This does not require a refreshing online experiment and a fixed offline corpus
to draw the same batches, or imply identical learning curves.

## Ownership and proposed interfaces

All shared loading components belong to `embodichain/data_pipeline/datasets/`.
Names below are proposed, not currently exported.

| Component | Responsibility |
| --- | --- |
| `SequenceSpecCfg` | Validated history length, prediction horizon, sampling mode and padding policy; use `@configclass`. |
| `BatchSource` protocol | Expose schema and `sample_context(batch_size, spec, generator=...)`; return an independently owned TensorDict. |
| ODS source adapter | Delegate publication, locks, provenance, errors and quotas to the existing engine. |
| LeRobot source adapter | Read through the official SDK, validate or explicitly map metadata, and provide bounded resident numeric storage. |
| Dataset wrappers | Share batch iteration/collation; preserve `OnlineDataset` compatibility and add an offline wrapper. |

Share pure window/mask calculations and canonical padding between source
adapters. Keep source ownership, synchronization and lifecycle separate. The
public contract must not depend on Franka, a 51-feature layout or ChunkBC.
Task features, absolute TCP-target construction, normalization fitting, noise,
training and efficiency reporting remain with their existing owners.

## Batch and sequence contract

- Use `obs`, `actions`, `valid`, `loss_mask` and source metadata in a TensorDict.
  `obs` may be nested named tensors; feature layout/dtype are schema-defined.
- For history H and prediction horizon F, return `[B, H + F - 1, ...]`.
  Index H-1 is the current pre-action observation, paired with its recorded
  action request or controller command as explicitly declared by the schema.
- History includes now; future supervision includes now. `loss_mask` is true
  only for real frames at or after the current index. Every current anchor is
  valid. Never cross an episode, invalid gap or continuity change.
- Preserve current episode-uniform selection followed by uniform eligible
  anchors. Any alternative weighting needs an explicit configuration and its
  own parity tests; it is not a storage-adapter default.
- Invalid numeric payloads are zero; invalid source/language IDs are -1.
  Apply canonical masking after any row-local feature conversion: quaternion
  conversion can map a zero input to a nonzero identity rotation. Stateful or
  stochastic transforms must declare temporal/RNG requirements separately.
- Schema validation includes ordered names/slices, dtype, observation timing,
  action representation, normalization, frame, rotation convention and step
  duration. A matching vector width alone does not establish compatibility.
- Persist portable dataset/episode/frame identities. Keep engine-local run ID,
  generation and acceptance time as optional origin metadata, not a replacement
  for portable identity. Resolve language using the source's own dictionary;
  unrelated integer language IDs are not equivalent text.
- Persist continuity and segment metadata when present. For a source that
  cannot represent a requested boundary/mode, reject it explicitly. The current
  PickPlace format supports continuous valid spans and rejects adjacent valid
  continuity changes or negative continuity IDs on valid frames.

## Loading, memory and concurrency

The first offline implementation supports numeric episodes and explicit
schema mappings. Reuse official SDK decoding instead of implementing another
LeRobot format reader. Make SDK imports optional/lazy; metadata inspection must
remain usable without loading simulation or video stacks. Do not silently infer
action semantics for external datasets lacking EmbodiChain metadata.

Read in bounded SDK batches. Require an explicit resident-memory allowance and
estimate tensor allocation before materialization; reject oversized datasets
before allocation. This initial resident backend does not promise scalable
out-of-core image/video training. Leave decoding, bounded lazy episode caches
and additional storage backends for a separately measured extension. Record
loading cost separately from resident minibatch sampling.

ODS must choose anchors, check validity/identity/quota and copy the selected
payload under its existing producer lock. Do not expose unlocked shared views
or copy the whole cache for every batch to make a common interface convenient.
Pure window calculation uses O(B*T) validity work and O(B*(H+F)*D) selected
payload storage, matching the existing path; avoid per-frame Python dispatch.
The padding fix writes into the newly constructed feature tensor in place.

Batch ownership must survive later refills or samples. A supplied generator
must isolate sampling from augmentation/model randomness; adding this option
to the engine must preserve its existing default RNG behavior. ODS owner-only
refill/stop operations remain outside Dataset workers. Budgeted offline
sampling must not multiply allowances through copied DataLoader workers; start
with explicit owner-process consumption and reject unsupported worker modes.

Finite stored episodes may be sampled repeatedly. State source cardinality,
iteration limits and quota exhaustion separately. Preserve immediate empty
window errors and distinguish quota exhaustion, permanent read/worker failure
and schema errors. Retry/wait deadlines belong to the caller. Do not catch a
permanent failure and silently continue with old data.

Use a common delivered-chunk unit for optional per-source limits, including
duplicate draws within a batch. Preserve atomic full-batch debit after copying;
failed optimization does not refund delivered chunks. Live replacement retires
unused credits, whereas an offline source does not refill. Offline protected
resume must restore its ledger or fail; ODS resume continues to declare a new
source run rather than exact shared-cache recovery.

## Alignment gate

The initial executable gate is
`tests/data_pipeline/test_ods_pick_place_alignment.py`. It uses the actual
engine sampler and quota ledger with a deterministic shared CPU buffer, actual
official SDK persistence/readback, actual feature conversion and actual learner.
It covers dirty tails, starts/ends, single-frame spans, invalid gaps, horizon
1/4/64, nonbinding/binding quotas, failed full-batch allocation, provenance/RNG
equality and three optimizer updates with identical augmentation randomness.
Unsupported continuity is rejected before offline collection.

Before declaring the public adapters interchangeable, extend this gate to:

1. Preserve the small real-producer check described below as an integration
   gate when extracting the public interfaces; freeze each source snapshot
   during its paired comparison.
2. Schema/order/dtype/action-contract mismatches, explicit external mappings,
   timestamp gaps, language resolution and supported continuity/segment modes.
3. Source replacement and copy ownership, worker/read errors, quota depletion,
   bounded loading and unsupported DataLoader configurations.
4. A lightweight offline import in a process without simulation dependencies.

Use exact tensor equality for deterministic CPU numeric paths and explicitly
defined tolerances for nondeterministic device operations. Report corpus
identity, selected frame IDs, schema, normalizer/model/RNG states and SDK
version. Numerical parity is not evidence of a throughput improvement.

### Completed PickPlace validation

The [recorded check](../../extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/ODS_ALIGNMENT_VALIDATION.json)
passed nine paired CPU cases and 146 related regression cases. A real expert
producer also supplied two generations/four accepted episodes (1,330 valid
frames); four paired batches/32 chunks and four optimizer updates matched
exactly, including sampling RNG, loss, gradients and optimizer state. All eight
observed owned processes exited. These checks validate the current numeric
showcase path, not the proposed public API or every batch of the previous pilot.
The record retains an earlier harness filename failure and the native shutdown
warning; neither is silently removed from the evidence.

## Implementation sequence and performance gate

1. Land canonical padding, continuity rejection and the paired tests. Retain
   previous experimental records with their original source revisions.
2. Extract schema/window contracts and private pure sampling helpers; preserve
   existing engine behavior and add an explicit generator without new defaults.
3. Add the bounded LeRobot numeric adapter and offline Dataset wrapper. Extend
   `OnlineDataset` with opt-in context sampling; keep old chunk APIs working.
4. Migrate the showcase and remove its duplicate loader/sampler implementation.
   Avoid a new registry, training framework or generic backend plugin system.
5. Qualify end-to-end overhead before repeating timed comparisons. Include B=1
   and B=256, F=1/64, C=32/256, loading/peak memory, lock-held sampling time and
   concurrent refill. Use equal payloads, instrumentation and workload settings;
   report per-batch and wall-clock costs separately. Disclose feature conversion
   placement and acquisition/persistence costs; numerical parity alone does not
   make differently staged pipelines a pure storage comparison. The existing 5% telemetry
   overhead gate remains unresolved and cannot be waived by parity tests.

Public API documentation and focused tests accompany each implementation step.
Image/video loading, mixed online/offline replay, distributed worker ledgers,
DAgger, RLInf integration and exact cache recovery are outside the first scope.
