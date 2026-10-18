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
   `sample_batch_with_history()` uses the same selection/copy boundary and
   returns the complete contiguous causal prefix plus a supervised window.
   Its `valid` mask excludes right padding; `loss_mask` selects the supervised
   frames. Recurrent consumers can reconstruct inference history and supervise
   episode starts. Prefixes never cross invalid frames or continuity changes;
   optional start-window bias leaves ordinary sampling unchanged.
   `sample_batch_with_context()` samples every valid anchor, including episode
   ends, with a fixed causal-history/future extent. The current frame is at
   `history_length - 1`; invalid/continuity boundaries pad both sides with
   zero payloads and `-1` language/provenance IDs. Consumers use `valid` for
   history padding and `loss_mask` for real current/future supervision.
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

## PickPlace experiment integration

`examples/data_pipeline/ods_pick_place_showcase.py` owns the entry point, policy
models, measured-state features and separate student evaluator. The private
`_ods_pick_place_training.py` orchestrates geometry policies;
`_ods_pick_place_policy.py` owns the short-history absolute-target chunk learner;
`_ods_pick_place_benchmark.py` owns validated numeric corpora, batch-identity
replay and efficiency accounting. The geometry feature contract uses measured
proprioception, TCP/cube transforms, goal-relative positions and cube velocity.
Normalization is frozen from initial training data. Policies do not consume
expert segment annotations during inference.

MLPs supervise individual frames using pose requests or joint residuals. The
geometry GRU uses pose requests, complete causal prefixes and masked supervised
windows; its recurrent state advances from episode reset during evaluation.
The older joint-command GRU remains a separate compatibility route with its
original observation/checkpoint contracts.
Chunk BC derives absolute TCP targets from pre-action observations and recorded
requests. Only causal history is a model input; future observations are used
solely to decode supervised commands. The student executes queued network
targets through the same bounded relative-pose controller, replans periodically,
and clears history/queue at reset. Input and target statistics are frozen from
initial training data and both are included in comparison fingerprints.

The runner supports live overlap, frozen initial data, offline full-corpus and
serial-refresh modes. Offline matched delivery replays an explicit MLP frame
manifest against the same validated corpus; recurrent/chunk exact-manifest replay is
not supported. Serial refresh is labeled an unmatched schedule and cannot
isolate overlap under identical supervision. Corpus manifests validate the
observation/action schema and tensor content; content identity is independent
of engine run IDs. Learner resume restores model/optimizer/RNG state, with a
fresh ODS source for live runs rather than a restored replay cache.

Small-data gates, expert physical qualification and student learning assessment
are separate. Smoke success and low training loss do not establish physical
student success or an efficiency advantage. See the task-owned
[experiment protocol](../../../extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/BENCHMARK.md)
for comparable acquisition-inclusive/data-ready budgets and
[task guidance](../../../extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/README.md)
for commands and measured results. Optional anchor-budget pacing and compact
source-use accounting live in `_ods_pick_place_usage.py`; defaults retain fixed
round scheduling. Deferred evaluation closes the producer before evaluating
saved policies, so its model-selection results are post hoc and cannot certify
online time-to-target. Evaluation scalars use each checkpoint's actual policy
step; save-completion events separately timestamp when its weights became
available. The study runner preserves missing save times in older runs and
reports drawdown only over evaluated policies. Background resource sampling
covers startup and waits independently of update counts; process-tree PSS is a
sampled peak and learner Torch counters exclude native simulator allocations.
See the task-owned
[tuning protocol](../../../extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/TUNING.md)
for rate units, protected scene splits and measurement limits.

## Sampling failure boundary

`sample_batch()` checks the shared worker error, READY state and worker liveness
before selecting a window. Invalid arguments raise `ValueError`. If no unlocked
valid window satisfies `chunk_size` and `sampling_mode`, it raises `RuntimeError`
immediately; that eligibility failure does not itself change the engine state.
Refill accounting runs only after a successful clone. `OnlineDataset.__iter__()`
does not catch these exceptions, so retry policy belongs to the consumer.

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
