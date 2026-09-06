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
6. Refill is requested after sample count reaches `refill_threshold * buffer_size`. Duplicate
   requests are suppressed only while `fill_signal` remains set; the worker clears it when it
   starts a fill, so sampling that reaches another threshold during an active fill may queue
   one subsequent full refill.
7. Worker exceptions are serialized into a fixed 64 KiB envelope and reconstructed when safely
   picklable; otherwise consumers receive stable `OnlineDataWorkerError`. The error is shared
   so all consumers observe terminal failure.
8. Stop first signals close and wakes fill, then joins, terminates, and finally kills if
   needed. A forced shutdown marks durability unconfirmed and ends in FAILED. Concurrent stop
   cancels a starting engine.
9. `OnlineDataset` is infinite. Item mode samples one row then removes the leading dimension
   so DataLoader can collate; batch mode re-samples a dynamic chunk per yield. Multi-worker
   readers sample shared memory independently.

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
