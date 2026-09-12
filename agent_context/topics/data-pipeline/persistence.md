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

### Common failures and recommended change sites

| Symptom | Likely cause / change site |
|---|---|
| Saved images/actions mutate after reset | Async caller enqueued live rollout views; clone-to-CPU and deep-copy must happen before enqueue. |
| Episode indices reorder or LeRobot races | More than one persistence worker or non-FIFO access; retain single-worker ownership. |
| Retry writes a duplicate episode | A post-commit failure bypassed sticky fragment state or fragment ID changed. Keep `save_episode()` as the explicit irreversible point. |
| Depth file exists without matching episode, or vice versa | Determine whether failure was before or after LeRobot commit; only pre-commit work can be aborted transactionally. |
| Async collection consumes growing RAM | Writer cannot match producer throughput; queue is intentionally unbounded. Throttle production or select sync persistence. |
| Finalize succeeds while work remains | Sentinel/join ordering or error aggregation changed; queued items must precede sentinel and base finalization. |
