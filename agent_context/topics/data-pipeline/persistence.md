# Recorder and depth persistence

[Topic overview](data-pipeline.md). Read this for the selected detailed flow.

### Persistence transaction and concurrency boundaries

1. `env.step` appends observations/actions to the rollout buffer. `reset(options={"save_data":
   True})` invokes `DatasetManager.apply("save", env_ids)` before the relevant rows are
   cleared. `finalize()` is a storage barrier only; it never commits a live rollout
   implicitly.
   Expert actions follow the environment's stored-action schema: active qpos by
   default, or flat `[qpos, qvel]` with matching feature names and metadata when
   position-velocity mode is enabled. The policy action space is not resized.
2. Recorder construction requires `1 / env.step_dt` to be an exact integer FPS. Non-integral
   simulation rates fail early.
3. `_save_episodes()` slices only valid lengths. Segment-fragment mode creates independent
   append-only payloads; accepted fragments save by default and failed fragments require
   opt-in. If an explicit commit produces no eligible fragment, it raises rather than silently
   succeeding.
4. `_save_single_episode()` first normalizes causal observation/action pairs, writes subtask
   metadata by temp-file replacement, begins depth output, and submits non-depth frames to
   LeRobot.
5. `dataset.save_episode()` is the primary commit point. Immediately after it returns, the
   recorder marks the dataset committed and advances `curr_episode`; sidecar filenames
   therefore cannot reuse that index.
6. Failure before the LeRobot commit rolls back recorder time and aborts depth temp files.
   Failure after commit cannot roll back the episode: it retains the episode, surfaces the
   error, and a fragment records sticky partial-commit state so retry cannot duplicate it.
7. Fragment IDs provide deterministic same-process deduplication. They are not a cross-process
   recovery journal.
8. Async enqueue clones all tensor payloads to CPU and deep-copies metadata in the caller
   before reset reuses the buffer. One FIFO worker is the sole LeRobot accessor and assigns
   deterministic episode order.
9. The async worker records per-payload errors but continues draining later items. Finalize
   rejects new work, queues a sentinel after all existing payloads, joins the worker, calls
   base finalization, and aggregates background plus storage errors. Its queue is unbounded,
   so sustained writer lag can grow RAM.
10. When sidecars are enabled, depth uses one HEVC MP4 per sensor per episode. Each writer
    creates a same-directory temp file and replaces the final file on close; failed/empty
    episodes clean the temp. The manager attempts every sensor, records only successfully
    finalized videos, atomically replaces `depth_meta.json`, then raises aggregated failures.
11. Sidecar encoding is active only when `DepthVideoCfg.enable` is true and
    `detect_depth_encoder()` finds the requested HEVC encoder. If it is disabled or
    unavailable, depth stays as ordinary numeric LeRobot features regardless of
    `keep_numeric_fallback`. When sidecars are active, `keep_numeric_fallback=True` duplicates
    exact numeric depth in LeRobot; otherwise depth is removed from LeRobot frames and read by
    composing the dataset with `load_depth_dataset()` (`datasets.py`).
12. `meta/embodichain_episodes.jsonl` is appended under the recorder's metadata lock after the
    LeRobot and depth commits; unlike subtask parquet and `depth_meta.json`, it is not
    replaced as one atomic file. A JSONL append failure is therefore a surfaced post-commit
    failure, not a rollback point.

### Offline episode receipts

The opt-in Affordance collector uses `DatasetManager.commit_episode_rows()`
with exactly one save sink whose exact type is synchronous `LeRobotRecorder`.
Async recorders, additional save sinks, deferred video encoding, fragment mode,
and saving failed episodes are unsupported. Preflight rejects unsupported
sinks before the first collection reset; row payload validation rejects empty
or duplicate rows and missing episode/commit identity.

`DemoCommitReceipt` identifies the physical row, episode, commit, LeRobot index,
and confirmed modalities. The recorder returns it only after primary dataset,
metadata, and configured depth outputs finish. If enabled, trajectory auto-save
adds `trajectory` confirmation at the environment boundary; only that complete
host receipt advances the collection quota. Dataset receipts remain inspectable
when a later row or trajectory save fails, so a failed run cannot be mistaken
for a rollback of earlier writes. A receipt says nothing about an absent modality.

LeRobot v3 keeps Parquet writers open after `save_episode()`. The receipt path
finalizes and locally reloads the dataset for each accepted episode, then uses
the public resumed-recording path for the next append. This ensures readable
footers and avoids reopening an existing file for truncating writes, at the
cost of a dataset reload per episode.

Commit IDs deduplicate within one recorder process. Once a LeRobot save starts,
any persistence failure is treated as uncertain and blocks further writes to
the sink; it cannot safely be retried under a different commit ID. This provides no
cross-process recovery. Optional trajectory files and LeRobot are not one
atomic transaction. The host records accepted rows, completed host receipts,
and underlying dataset receipts separately in the run manifest. See
[offline collection](../env-framework/execution.md#offline-affordance-collection)
for reset, quota, provenance, and run status ownership. The online and ordinary
async collection contracts are unchanged.

### Common failures and recommended change sites

| Symptom | Likely cause / change site |
|---|---|
| Saved images/actions mutate after reset | Async caller enqueued live rollout views; clone-to-CPU and deep-copy must happen before enqueue. |
| Episode indices reorder or LeRobot races | More than one persistence worker or non-FIFO access; retain single-worker ownership. |
| Retry writes a duplicate episode | A post-commit failure bypassed sticky fragment state or fragment ID changed. Keep `save_episode()` as the explicit irreversible point. |
| Depth file exists without matching episode, or vice versa | Determine whether failure was before or after LeRobot commit; only pre-commit work can be aborted transactionally. |
| Async collection consumes growing RAM | Writer cannot match producer throughput; queue is intentionally unbounded. Throttle production or select sync persistence. |
| Finalize succeeds while work remains | Sentinel/join ordering or error aggregation changed; queued items must precede sentinel and base finalization. |
