# RL training, evaluation and extension

Read this when the request needs these details. [Topic overview](rl-learning.md).

## Training and Evaluation Lifecycle

The standard trainer repeats:

1. start and collect a rollout;
2. update the algorithm;
3. log train metrics;
4. evaluate when the configured step boundary is reached;
5. save periodic and best-evaluation checkpoints.

The training entry owns its acquired environments, writer, tracking run and
process group from construction onward, including initialization failures.
`runtime.py` protects environments until a `PolicyRuntime` is returned; its
caller then owns `close()`. Cleanup runs in reverse acquisition order and keeps
an existing error intact. A writer flush failure on otherwise successful exit
still propagates. Existing W&B runs and distributed process groups are borrowed:
the entry neither reinitializes nor closes them. Simulator cleanup does not exit
the host process.

Evaluation uses an independent environment and
`evaluate_episodes()`. It counts completed asynchronous episodes, reports
terminal metrics, temporarily switches the policy to evaluation mode, and
restores each submodule's prior mode, including frozen normalization layers.
The viewer adapter captures these modes at setup and restores them once on close.

`Trainer(event_cfg=..., eval_event_cfg=...)` attaches each phase to its own
native environment through `EmbodiedEnv.add_events` before the collector's
initial reset. The host owns startup, interval and reset execution; the trainer
does not execute these terms a second time. Plain vector environments retain
post-step interval callbacks, but cannot host episode camera recorders or other
lifecycle modes. Their class functors initialize through a collector reset
callback at construction, full-rollout resets and checkpoint resume; row-level
auto-reset lifecycle remains the native host's responsibility.
Before and after evaluation (including failure), the trainer
uses `flush_recordings()` to discard unfinished frames and wait for committed
videos, leaving recorders open for the next evaluation. See the
[camera lifecycle contract](../manager-functor/manager-functor.md#camera-recording).
External interval functors retain their post-evaluation `flush` / `finalize`
hooks, including the legacy output-path argument; these hooks are not invoked
for environment-owned functors.
The launcher resolves train and active eval camera demand before creating the
first World. Direct Python callers must likewise select rendering before adding
camera events. Phase event names must not collide with task event names.

### Checkpoints and resume

Checkpoints include policy parameters, trainer and complete-rollout counters,
best-evaluation state, observation-normalizer state when enabled, and optimizer
or LR-scheduler state when present. The loader requires schema v2 with complete
compatibility metadata and rank-local RNG/normalization state; GRPO includes its
fixed KL reference. Older schemas and incomplete required metadata are rejected
before model mutation; missing fields are not filled from defaults.
Both trainers expose `load_checkpoint(path, resume_scheduler="restore")`.
CLI `--resume` selects a file or `latest/best` from `--run`; the source and new
output run remain separate. Selectors follow the exact manifest entry, fail when
the entry or its file is missing, and never fall back to another checkpoint.
Training selection defers unrelated file existence checks through
`RunManifest.load(check_files=False)`; default manifest loading still validates
every referenced file for evaluation. Schema-v2 files restore independently of
source-run snapshots and historical checkpoints. Resume keeps the historical
best score but does not copy its file: the new run gets a best entry only after
an evaluation improves that score.
Budgets are absolute targets and environments start
fresh episodes. `restore` keeps the saved LR horizon; `restart` preserves
optimizer momentum and binds the configured curve to remaining updates.
Zero-ending linear curves cannot resume beyond their horizon in restore mode.
The loader checks semantic and structural compatibility before updating state,
including effective APG rollout, gradient-accumulation and action settings.
Native CLI training-event declarations also participate in this check; generated
run output paths and evaluation-only event settings do not.
CLI and Python API checkpoints interoperate; absent shared CLI context produces
a warning while available runtime metadata is checked.
Variable-horizon APG transition budgets use the actual update count to check
an explicit restored LR horizon.
`checkpoint.py` owns shared validation and atomic persistence.
Optional environment training state uses the same v2 file and is collected per
rank. Environments implement `get_training_state`, side-effect-free
`validate_training_state`, and `load_training_state`; validation runs before
learning state changes, and restoration precedes the fresh episode reset.
Every checkpoint includes a per-rank environment-state list; stateless
environments store `None`. G1 Rough uses these hooks for terrain assignments and
its task generator,
requiring the original terrain map and per-rank environment count. Its first
reset preserves the restored assignments; physical episodes still restart.
Configuration snapshots are frozen from loaded data on the first save; later
saves update the run manifest without rewriting those snapshots. Final saves
reuse the current update's periodic or best checkpoint. Subsequent saves and final
reuse verify the previous saved file and active run manifest references; missing
artifacts raise errors instead of being recreated. Unreferenced historical files
are not required. Interrupted updates retain the previous complete checkpoint.

### Update semantics

`trainer.seed` seeds Python, NumPy, Torch, CUDA, and Warp before environment
construction. Set `trainer.torch_deterministic: true` when a reference run
also requires deterministic PyTorch algorithms.

Complete APG mode resets before every independent microbatch and preserves one
graph across the entire environment-provided horizon. It masks rewards after
the first done, freezes observation statistics through backward, applies the
rollout's objective scale, and averages `gradient_accumulation_steps` full
trajectories before one optimizer step. It
never shortens a scheduled horizon to satisfy `total_timesteps`; without an
explicit timestep budget, CLI `iterations` maps to an exact optimizer-update
budget.

Programmatic `APG.accumulate_complete_rollout()` also accepts a graph-connected
`terminal_value` per environment. It contributes with the full-horizon discount
only to rows without a hard termination or truncation. The caller owns value
fitting, freezing estimator parameters while retaining state gradients, and any
soft survival multiplier. The common trainer does not fit a value function
automatically.

Complete mode can clamp actions to the environment space and install a
per-environment action-adjoint norm hook. Non-finite adjoint rows are zeroed;
finite rows are clipped independently with an overflow-safe norm. APG also has
a pre-clip policy-gradient safety limit that skips unsafe updates.
PPO supports adaptive learning rates from Gaussian KL, clipped value loss,
configurable minibatch counts, and optional action-mean bounds. Adaptive KL
scheduling cannot be combined with `algorithm.cfg.lr_scheduler`. The actor and
critic can maintain independent observation statistics in
`models/normalizer.py`; these stay frozen through collection and PPO epochs,
then update for the next rollout. Evaluation keeps them frozen. A restored
policy must use the original normalization settings.
Time-limit truncations bootstrap rewards with the value stored for the
transition before GAE masks the episode boundary.
See the [PPO configuration options](../../../docs/source/overview/rl/config.md)
for the public configuration fields.

`Trainer` increments `num_updates` after each algorithm update.
`trainer.save_frequency_updates` enables saving at update intervals. An
explicit positive `save_freq` additionally enables saving by environment steps.

On the simulator path, distributed mode initializes NCCL, assigns one CUDA
device per local rank, wraps the policy in
`DistributedDataParallel`, aggregates step and episode statistics, and
keeps logging, evaluation, and checkpoint ownership on rank zero.
Because action evaluation calls the underlying policy, algorithms explicitly
average gradients before clipping and stepping; adaptive PPO uses global KL.
GRPO removes post-terminal transitions before shuffling when
`truncate_at_first_done` is enabled. Distributed GRPO weights losses and metrics
by valid transition counts and keeps all ranks in the same optimizer steps;
a rank with an empty local tail still joins gradient reduction for other ranks.
All ranks participate in checkpoint operations and error synchronization.
Resume requires the same world size and restores each rank's RNG and local
normalization statistics alongside common policy/optimizer state.
Differentiable algorithms and lightweight learning environments do not
currently support this distributed path.

## Execution and storage boundaries

`SyncCollector` owns standard transition writes. Native `EmbodiedEnv` lends only
previous action/done views through a private collection scope; it does not bind
the learner TensorDict or advance a training cursor. The scope covers reset,
steps and bootstrap, releases borrowed views on failure, and restarts reward
history each rollout. Legacy environments and explicitly bound recording
buffers retain `set_rollout_buffer` and their environment-side writer. Expert
episode storage remains separate. The [Gym execution contract](../env-framework/execution.md)
owns pre-reset return-value preservation and episode metadata.

When a finite terminal value is unavailable, timeout bootstrapping retains the
current-state value fallback. Collection clears stale terminal values even when
callers reuse a raw TensorDict without `Buffer.start_rollout`.

Policies own an optional private `_training_input_spec()` contract declaring
consumed transition fields and whether action evaluation mutates inputs.
`algo/common.py` resolves it once per update, merges algorithm-owned fields and
selects before flattening/minibatch gathers without identifying concrete policy
classes. GRPO includes its KL reference policy's fields; any undeclared consumer
retains the full payload. Field selection and reference-target copying are
independent guarantees. Built-in policy declarations guard subclass/instance
and auxiliary overrides; custom policies default to full payload and defensive
target copies unless explicitly opting in. Declarations are read afresh each
update, and transition-derived `next_value` keeps its existing precedence.
The same declaration identifies observation fields consumed by post-update
normalization. Disabled built-in normalizers skip preparation; custom updater
overrides retain the complete flattened observation payload.
Actor/critic observations, auxiliary labels and adaptive-KL parameters remain
aligned. This contract does not make arbitrary auxiliary input mutation safe.
PPO and GRPO accumulate reporting metrics on device until the update ends.
Adaptive learning-rate decisions and non-finite-gradient checks remain at their
existing optimizer boundaries.
PPO reads parameter groups afresh and deduplicates shared parameters each call.
Each group uses sequential native gradient clipping, and non-finite gradients
reject the optimizer step.

`Trainer` snapshots per-step metric means and episode completions on device,
then transfers them at the rollout boundary. TensorBoard/W&B still receive each
step's tags and recorded global step; this is deferred reporting, not temporal
averaging. Completed episodes retain step-then-row order and bounded history.
Distributed ranks exchange only new completions at each rollout boundary;
rank zero merges them in step, rank, then row order into one bounded window.
Other ranks retain local histories. Metric steps use the summed environment
count across ranks rather than the pre-rollout global counter.
`DifferentiableTrainer` batches built-in CUDA episode statistics in bounded
chunks, flushing by the collection boundary and on collection failure. Completed
episodes retain step-then-row order and integer-length precision; CPU and custom
step callbacks keep immediate reporting. Action-adjoint diagnostics transfer
together after backward without moving gradient-safety decisions.
Both trainers share scalar sink writes and evaluation score comparison in
`embodichain/learning/rl/_trainer_reporting.py`; their callers retain history,
rank filtering, distributed reduction, scheduling and checkpoint ownership.
Native TensorBoard combines a step's prepared float scalars into one event:
tag/value/step/order remain intact, while walltime and partial-I/O failure
granularity follow that event. Custom scalar writers retain individual calls.
Collection suppresses sink errors; update/evaluation reporting propagates them.
A failed native submission is never replayed because it may already be queued.

`utils/helper.py` owns observation-group resolution for both public flattening
helpers and standard collection's snapshots into owned rollout slots. Both paths
preserve configured group ordering, sorted nested leaves and grouped dtype
promotion. Flattening without a destination may return borrowed storage for one
tensor leaf.
Differentiable collection clones retained initial/current/successor observations
before environment buffer reuse. Inference preparation also owns a snapshot
because custom transforms and
normalizers may modify their inputs in place.

The differentiable collector uses a private built-in sampling path that retains
actions and entropy without building unused likelihood/value graphs. Public
`get_differentiable_action` keeps full outputs, and custom sampling overrides use
that public path. APG actor-critic evaluates values at its own update boundary.
APG batches CUDA reporting scalars at each segment's existing pre-backward
boundary and combines ordinary same-device gradient finite checks into one
host read. Every segment still backpropagates and validates before the next;
an invalid accumulation window discards subsequent gradients. Other gradient
layouts/devices retain sequential checks.

## Official Examples

| Example | Environment path | Config location |
|---------|------------------|-----------------|
| CartPole | registered simulator Gym env | `embodichain/extensions/tasks/configs/tasks/classic_control/cart_pole/agents/` |
| PushCube | registered simulator Gym env | `embodichain/extensions/tasks/configs/tasks/manipulation/push_cube/agents/` |
| PointMass PPO | registered lightweight env, standard rollout | `embodichain/extensions/tasks/configs/tasks/classic_control/point_mass/agents/ppo.yaml` |
| PointMass APG | differentiable lightweight env | `embodichain/extensions/tasks/configs/tasks/classic_control/point_mass/agents/apg.yaml` |
| Newton planar reach | experimental differentiable FK reference | `embodichain/learning/rl/experimental/newton/` |

`PointMassRL` is the reference environment for comparing standard and
differentiable training over the same task dynamics. The Newton planar-reach
example is an experimental gradient reference, not a general simulator task.

## Extension Points

### Add an Algorithm

1. Implement a `BaseAlgorithm` subclass and config under `algo/`.
2. Declare the correct `RolloutKind`.
3. Register the config/class pair in `algo/__init__.py`.
4. Add focused algorithm, routing, and rollout-contract tests.

### Add a Policy

1. Implement the `Policy` contract under `policies/`.
2. Register it with `register_policy` from `embodichain.learning.rl.policies`.
3. Ensure its outputs satisfy every intended algorithm.
4. Provide graph-preserving sampling if used with differentiable rollouts.

### Add a Lightweight Environment

1. Implement `LearningEnvProtocol`, or `DifferentiableEnvProtocol` for APG.
2. Register the factory with `@register_learning_env`.
3. Ensure finished rows auto-reset while returning terminal reward/done with
   the next initial observation.
4. For variable complete APG rollouts, implement
   `ScheduledDifferentiableEnvProtocol` and return the full non-truncated horizon.
5. Expose `observation_normalize_mask` when semantic dimensions must remain
   raw during normalization.
6. Add an official config under
   `embodichain/extensions/tasks/configs/tasks/<domain>/<task>/agents/` when it is a
   bundled task.

Use `add-task-env` for simulator-backed task environments and
`manager-functor` for their observation, reward, event, and action
components.

## Actor-critic gradients

APG Actor-Critic retains derivatives through critic inputs while freezing critic
parameters/statistics during actor collection and backward. Terminal-value
gradients, detached lambda-return regression and a separate critic optimizer
have disjoint parameter groups. Task-data masks stay detached.

Optional `update_critic_normalization` runs after actor backward and before
lambda targets; default leaves statistics unchanged. Weighted current states
and valid bootstrap successors contribute, while complete-rollout post-terminal
suffixes do not. Critic minibatches contain only positive-weight samples. An
empty valid set leaves parameters and optimizer momentum unchanged.

## Experimental host replay

`learning/rl/experimental/_host_replay.py` owns a private synchronous transport
boundary around CPU replay storage. The wrapped sampler retains window, mask
and history rules; only transition/sample transfers move data. Returned device
samples own storage independently of subsequent host writes, with dtypes
preserved and gradients detached. Host sampling uses CPU RNG; it does not
preserve a prior CUDA sampling stream.

Its private ring-checkpoint helpers borrow CPU tensors for synchronous saving
and validate capacity, fields, dtypes and cursor metadata before restoring owned
storage. The caller serializes writers, pairs replay with model/RNG receipts,
and closes sampler sequences across simulator restarts. Restoring replay alone
does not restore the physical trajectory or justify exact-resume claims.

The [Sharpa example](../../../examples/learning/sharpa_inhand/README.md) owns
resource profiles and resume requirements. Its `_native_checkpoint.py` owns
synchronous persistence, model/replay pairing checks and resume provenance.
