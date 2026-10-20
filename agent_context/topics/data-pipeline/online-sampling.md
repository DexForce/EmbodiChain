# Online worker lifecycle and sampling

[Topic overview](data-pipeline.md). Read this for the selected detailed flow.

### Online engine lifecycle and sampling

1. Config controls episode-row capacity, max episode steps, buffer device, Gym/action configs,
   refill threshold, retry attempts, and startup timeout (`engine/data.py`). The
   implementation accepts any `buffer_device` value but calls `TensorDict.share_memory_()`
   only when the resulting device is CPU; the documented cross-process shared-buffer path
   therefore depends on CPU storage (`engine/data.py`).
2. Only the creator PID may `start()` or `stop()`. Start moves `CREATED -> STARTING`, spawns a
   non-daemon forkserver child, starts an error monitor, requests fill, and blocks until
   initial fill or timeout. Success is `READY`; `FAILED` and `STOPPED` are terminal and cannot
   restart.
3. The child builds the environment with dataset saving and initial rollout disabled. It
   clamps environment episode length to the shared row capacity, then repeatedly produces
   complete successful demonstrations up to the configured attempt limit (`_run_sim_worker`,
   `engine/data.py`).
4. Before reusing `[write_start, write_end)`, `set_rollout_buffer()` invalidates those rows.
   The child publishes that range in `lock_index` under the same process lock used by
   samplers, writes the rollout, then clears the range after a full fill.
5. `sample_batch()` is READY-only. While holding the lock it excludes the current write range,
   checks every row in a sampled window, enforces `continuity_id`, applies
   segment/boundary/episode mode rules, and clones the final batch before releasing the lock
   (`engine/data.py`).
   `sample_batch_with_context()` samples every valid anchor, including episode
   ends, with a fixed causal-history/future extent. The current frame is at
   `history_length - 1`; invalid/continuity boundaries pad both sides with
   zero payloads and `-1` language/provenance IDs. Consumers use `valid` for
   history padding and `loss_mask` for real current/future supervision. Optional
   context segment mode restricts spans to accepted segments; explicit generation
   filters require provenance. An optional generator isolates sampling RNG.
6. The default sample-threshold refill is requested after sample count reaches `refill_threshold * buffer_size`. Duplicate
   requests are suppressed only while `fill_signal` remains set; the worker clears it when it
   starts a fill, so sampling that reaches another threshold during an active fill may queue
   one subsequent full refill.
   With `refill_policy="manual"`, only the owner requests subsequent fills through
   `request_refill()`. Outstanding manual requests coalesce, including while the
   worker is writing. `wait_for_generation()` observes complete publication and
   propagates failure/timeout; readiness is still established by the initial fill.
7. Worker exceptions are serialized into a fixed 64 KiB envelope and reconstructed when safely
   picklable; otherwise consumers receive stable `OnlineDataWorkerError`. The error is shared
   so all consumers observe terminal failure.
8. Stop first signals close and wakes fill, then joins for the configured finite,
   positive `shutdown_timeout` before terminating and finally killing if needed.
   This controls only the graceful-close wait; forced-shutdown join bounds remain
   separate. A forced shutdown marks durability unconfirmed and ends in FAILED. Concurrent stop
   cancels a starting engine.
9. `OnlineDataset` is infinite. Item mode samples one row then removes the leading dimension
   so DataLoader can collate; batch mode re-samples a dynamic chunk per yield. Multi-worker
   readers sample shared memory independently.

## Explicit owner-side flow control

`runtime.py` owns opt-in `ODSFlowController` around a caller-managed manual
engine. Its creating process/thread calls the controller; the application still
starts/stops the engine and owns training, evaluation and resource accounting.
An optional `ODSConfiguration` must match engine cache, producer count, quotas
and batch size. A plan or production limit requires an explicit positive
`gym_config.num_envs` dividing cache size; unplanned/unlimited legacy engines
retain their existing layout support.
With a plan, `prepare_batch()` schedules quota-threshold refill, bounded quota
waits and batch-start pacing; a pause does not accumulate catch-up credits.
Without a plan it only polls, leaving round/frozen refill decisions explicit.
The optional accepted-episode limit covers initial fill and whole-cache refills;
partial-cache remainder is not produced, and direct engine requests bypass it.
Already pending generations are adopted without a new engine request or refill
callback; only requests made through the controller receive request timestamps.

Controller waits drain lifecycle events and check health/caller deadlines.
`sample_context()` retries only live no-window/quota exceptions within a total
bound; Source/Dataset behavior stays immediate. Generation waits can accept an
absolute monotonic deadline that includes earlier work. Callback interval/refill
times use absolute host monotonic seconds; the application aligns its own clock.
The controller does not restore source or pacing state. The
[configuration guide](../../../docs/source/features/ods/configuration.md#apply-the-plan)
owns the integration example and callback details.

`engine.check_health()` is READY-only and checks existing worker errors/liveness
without sampling/refill. `get_flow_status()` returns compact generation and
optional quota counters without per-row audit records or payload clones.
Producer counters and quota totals use separate lock snapshots. Unlocked quota
uses the same valid/continuity and write-window test as the full audit, but is a
coarse scheduling bound; exact context/filter eligibility stays with sampling.

## Optional publication accounting

`OnlineDataEngineCfg.lifecycle_event_capacity` enables a bounded shared log;
zero preserves the default. The owner calls `drain_lifecycle_events()` for
ordered publication/eviction identities, eligible anchors and host-monotonic
timestamps. Capacity must accommodate initial filling before `start()` returns.
Consumers must drain during active refill and explicit waits: a full event log
backpressures the producer outside the trajectory-copy lock, and
`wait_for_generation()` does not drain it. Stop interrupts that backpressure;
remaining records can be drained after stop/failure. Events are accounting
metadata, not saved trajectory payloads or replay recovery.

## Optional per-trajectory sampling budgets

Set both `episode_sample_budget` and `episode_sample_hard_limit` to positive
integers with budget <= hard limit. Omitting both retains historical sampling.
Quota mode uses a CPU shared ledger with O(cache rows) storage. The producer
assigns fresh monotonic episode identities at publication; replacing a slot or
changing generation cannot refill an old identity's budget. Under the existing
copy lock, selection excludes exhausted sources and accounts for repeated rows
inside a batch. Only a successful copied batch debits its full count. Delivered
chunks remain charged if downstream optimization fails.

`OnlineDataBudgetExhaustedError` distinguishes insufficient unlocked eligible
quota from no valid windows and terminal worker failure. It is immediate and
does not partially spend a batch. `get_sampling_budget_snapshot()` returns
retained identities, budgets, delivered/remaining counts and lifetime delivery;
clean stopped engines remain auditable. The owner decides when to refill, wait
with a bound, or stop, directly or through the optional flow controller. The
fixed publication budget is not an adaptive replay retention policy and does
not restore old sources after process restart.

`configuration.py` derives bounded budgets, cache/refill settings and pacing
targets from measured accepted episodes/s and training chunks/s. It reads an
explicit qualified/provisional reuse profile and runtime metadata; it does not
infer scientific bounds or change the measured batch size/concurrency. Memory
constraints can reduce active consumption rather than raise quotas. Measurement
helpers preserve units and common monotonic windows. The
[core configuration guide](../../../docs/source/features/ods/configuration.md)
owns the reusable method; the [local binding](../../../examples/ods/simple_bc_demo/docs/CONFIGURATION.md)
owns the demo CLI and artifact mapping.

## Recorded actions and provenance

`action_dim` separates stored action width from robot-state `state_dim`; omitting
it preserves the historical width for both. Default
`action_semantics="expert_controller_command"` records the live
`ExpertActionSpec`. `action_semantics="policy_request"` requires an explicit
width and ordered `action_contracts`. After attaching its buffer and before
executing a demonstration, the worker checks the resolved recording semantics,
observed proprioception width, ActionManager contracts, contiguous slices and
controlled-joint order. Matching policy and expert widths does not establish
matching semantics.

The worker selects `EmbodiedEnvCfg.rollout_action_semantics` from the ODS
configuration. Explicit policy recording uses the existing raw ActionManager
route without requiring a DatasetManager; floating layout, finite values and
action-space bounds are checked before processing. Direct `ControllerAction`
commands cannot enter a policy-request buffer. The expert schema and policy
request schema remain independent. Configuration examples live in the
[public online-data guide](../../../docs/source/features/online_data.md).

`include_provenance` adds engine-local run, accepted-episode and generation IDs,
`ods_frame_index` and `ods_accepted_at_ns`. Frame indices are zero-based positions
within the stored episode. The timestamp is Unix nanoseconds at successful
acceptance/publication, stamped while holding the publication lock before the
row window becomes readable. It is not a monotonic or simulation timestamp.
Samplers clone provenance with observations/actions under that same lock.
Invalid frames, including history padding, retain `-1` in every provenance field.
`get_stats()` returns detached counters and excludes the active write window
from valid-frame counts. These IDs do not provide shared-buffer checkpoint
recovery or portable offline content identity.

## Local BC demo integration

`examples/ods/simple_bc_demo/train.py` is the runnable entry point. The same
directory owns `environment.py`, `config.py`, local `configs/`, ChunkBC in
`models.py`, shared `trainer.py`, and the independent `evaluation.py` CLI.
The module-qualified Gym ID is
`examples.ods.simple_bc_demo.environment:SimpleBCPickPlace-Franka-v1`; the task
is outside the official task catalog. See the [demo README](../../../examples/ods/simple_bc_demo/README.md)
for commands and the [evidence index](../../../examples/ods/simple_bc_demo/docs/README.md)
for historical implementation hashes and validation limits.

The sole learner uses measured geometry and causal history to predict absolute
TCP/gripper targets. Input/target normalization is frozen from the initial
training cache. Student execution has no expert segment inputs. Checkpoints
restore ChunkBC/optimizer/RNG state; live resume creates a fresh ODS cache.

The runner consumes public `BatchSource` adapters for live supply, a frozen
initial cache and resident LeRobot data. Shared pure quota selection in
`datasets/_sampling.py` preserves the draw algorithm while source ledgers keep
their distinct process/lock ownership. `data.py` owns experiment corpus hashes
and accounting, not a separate sampler. `collection.py` and `lerobot.py` own
bounded single-writer persistence and sidecar/content auditing around the public
reader. Cold offline episode count is explicit and independent of training
rounds. Run-local SDK caches count toward final disk footprint; temporary write
peaks remain unmeasured. Writer/producer failures propagate before publishing
an incomplete corpus.

`TensorDataSource.configure_sampling_budget()` optionally protects resident
sampling with one nonrefillable per-source allowance. It is creator-process/
thread only, rejects duplicate source identities and cannot be enabled after
sampling. Protected offline resume is rejected until allowance restoration
exists. Conventional offline baselines retain unrestricted reuse.

`configure.py` binds public throughput planning to this fixed learner.
`--auto-config-input` enables protected per-source allowances and deferred
evaluation; defaults retain fixed-round scheduling. Source caps do not establish
learning sufficiency. `--deferred-evaluation-schedule updates` aligns saved
panels; producer shutdown precedes deferred evaluation, so results are post hoc
and cannot certify online time-to-target. Comparison fingerprints retain G1
content, initialization and frozen normalization separately. Missing save times
remain unknown, and drawdown covers only evaluated policies. Resource sampling
covers startup/waits independently of updates; process-tree PSS is a sampled
peak and learner Torch counters exclude native simulator allocations.

Unit tests target reusable data-pipeline contracts. The demo model, persistence
and orchestration use runnable experiments and recorded evidence without
dedicated example unit tests. Expert physical qualification, small-data
diagnostics and student learning are separate; smoke success and low loss do
not establish a learning or efficiency advantage. The local
[ODS configuration guide](../../../docs/source/features/ods/configuration.md)
owns reusable rate units, planner limits, scheduling and measurement checks.
The [demo binding](../../../examples/ods/simple_bc_demo/docs/CONFIGURATION.md)
records its local CLI and artifacts; the historical
[pilot protocol](../../../examples/ods/simple_bc_demo/docs/OFFLINE_PILOT.md)
separates acquisition-inclusive and data-ready budgets.

## Sampling failure boundary

`sample_batch()` checks the shared worker error, READY state and worker liveness
before selecting a window. Invalid arguments raise `ValueError`. If no unlocked
valid window satisfies `chunk_size` and `sampling_mode`, it raises `DataUnavailableError`
immediately; that eligibility failure does not itself change the engine state.
Refill accounting runs only after a successful clone. `OnlineDataset.__iter__()`
and `SequenceDataset` propagate these exceptions, so retry policy belongs to the
consumer. `DataUnavailableError` subclasses `RuntimeError`; quota exhaustion
and terminal worker/read failures remain distinct and must not be retried by
matching exception messages.

## Failure diagnosis

| Symptom | Inspect |
|---|---|
| Engine times out or all readers fail | Inspect the shared worker error envelope and demo retry limit; do not catch only in one DataLoader consumer. |
| Samples contain rows being replaced | Selection or clone moved outside the `lock_index` critical section; restore the shared lock boundary. |
| Sequence crosses a reset or fragment | Fix validity/`continuity_id`/segment eligibility in `sample_batch()`, not downstream training code. |
| Engine cannot restart | Expected after FAILED/STOPPED; create a new engine instance. |
| Non-CPU online buffer is not truly process-shared | `_create_buffer()` currently does not reject it but skips `share_memory_()`; keep `buffer_device="cpu"` unless the implementation adds another explicit interprocess transport. |

## Online language ownership

The successful demo result supplies overall and segment text. The worker fills
`task_index`/`subtask_index` while its row window remains excluded, then publishes
the rows. `SharedLanguageRegistry` appends immutable task/subtask dictionaries in
a bounded shared UTF-8 buffer. No manager server or simulation object is needed
by a consumer. Registry and row locks retain independent responsibilities.

`engine.resolve_language(batch)` and `OnlineDataset.resolve_language(batch)`
return nested strings matching sampled dimensions. History padding keeps
`task_index` and `subtask_index` at `-1`; the resolver deliberately rejects these
invalid IDs. Select `valid` frames (or supervised `loss_mask` frames) before
decoding padded histories. Tokenization belongs to the training transform;
shared TensorDicts retain numeric IDs. IDs are engine-local,
not portable offline dataset indices. They remain valid across refill/shutdown
while the engine/registry object is retained. Exhausting `language_buffer_bytes`
fails generation through the existing worker-error channel before publishing an
incomplete rollout; create a new engine with adequate capacity rather than
reusing IDs. See `tests/data_pipeline/test_online_language.py` for forkserver,
spawn DataLoader, frozen sample, failure, and half-open range coverage.
