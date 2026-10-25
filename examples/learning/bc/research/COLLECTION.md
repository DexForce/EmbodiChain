# Fixed offline PickPlace corpus

`collect_pick_place.py` prepares the visual BC acceptance data directly from
the physical task in PR #828. Use the pinned reference worktree at
`51f1f0cd5a7602f23358c142997e9045323f547a` and the matching native runtime.
The producer does not construct an ODS engine or learner.

```bash
CUDA_VISIBLE_DEVICES=4 python examples/learning/bc/_support/_collect.py \
  --reference-source /path/to/pr828-worktree \
  --output /path/to/fixed-corpus --episodes 1024 \
  --num-envs 4 --gpu-ids 4 --batches-per-worker 8
```

GPU indices are examples. Admit only available devices. The collector queries
the assigned physical UUID immediately before each new worker, requires at least
32 GiB free and at most 20% sampled utilization, and otherwise waits within the
same worker deadline. One worker owns each selected GPU. Requests, UUIDs,
admission observations and cleanup records are preserved per worker group.

## Reference acceptance and collection

Expert qualification uses one environment and reference seeds `10000..10039`.
At least 38 of 40 scenes must satisfy both the measured physical success and
the demonstration executor's successful completion. Qualification scenes do not
enter training. The task's 600-control-step limit, 30 Hz control, physics,
expert planning, relative action decoder and physical success predicate are
unchanged. The camera calibration is `pick-place-v2`.

Collection defaults to four independent physical environments. Increasing the
batch size can alter the expert planner's shared retiming, so an alternative
parallel count needs its own qualification. `--batches-per-worker` controls
native-world reuse only, with a bound of 16 batches per process. Each reset uses
its explicit candidate seed. Buffers hold one physical batch, and each accepted
episode is written before its storage can be reused. Grouping batches avoids
repeating process imports and native initialization; it does not promise
byte-identical trajectories to fresh worlds. The manifest records each
episode's world group and batch index.

Candidates are assigned deterministically and published in candidate order.
The collector attempts at most twice the requested episode count. Rejected
attempts, segment reasons and measured outcomes remain in worker reports. Only
accepted episodes enter the fixed training manifest. Unused accepted tails
remain on disk with their provenance.

## Files and freezing

Each committed HDF5 episode contains:

- `geometry`: float32 `[T,51]` measured state features.
- `actions`: float32 `[T,7]` normalized relative-pose and gripper requests.
- `images/cam_high`, `images/cam_wrist`: uint8 `[T,256,256,3]` RGB frames.

`environment.json` binds the task and camera configuration through its SHA256
in `manifest.json`. Episode records bind length, candidate identity, source
device and file SHA256. A separate `qualification_environment.json` preserves
the scalar qualification deployment. Every worker preserves its exact
environment, request, demo metadata, outcomes and cleanup confirmation.

Training readers must reject `complete=false`. Completion publishes exactly the
requested accepted count after every scheduled worker has exited cleanly.
Repeating the command against a completed corpus is a no-op; never edit its
manifest or episode files after training starts. Partial clean workers can be
reused on restart; incomplete workers remain failed evidence and require an
explicitly reviewed recovery rather than being treated as accepted data.

`collection_seconds` measures wall time from the first collection request;
`preparation_seconds` starts at the canonical qualification request. Both include
restarts and waits. `last_invocation_seconds` describes only the final launcher
invocation. Worker reports separately record time after Python imports, so these
scopes must not be interchanged when reporting throughput.

Use `--diagnose-reuse` for a bounded three-batch lifecycle inspection. Diagnostic
outputs are separate from formal qualification and the frozen training corpus.
