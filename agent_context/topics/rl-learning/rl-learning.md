# Reinforcement learning

RL orchestration lives in `embodichain/learning/rl/`; shared MLPs, normalization
and layers live in `learning/models` and `learning/layers`. Package exports stay
lazy so lightweight APG imports do not initialize the simulator-backed Trainer.

## Entry Points

| Request | Owner |
|---|---|
| Unified training CLI/config assembly | `embodichain/learning/rl/train.py`; `embodichain train-rl` |
| Environment protocols and registration | `embodichain/learning/rl/env.py` |
| Algorithm type, rollout kind and registry | `embodichain/learning/rl/algo/` |
| Policy API, registry and observation groups | `embodichain/learning/rl/policies/` |
| Standard collection/storage | `embodichain/learning/rl/collector/`, `buffer/` |
| Standard trainer and optimizer scheduling | `embodichain/learning/rl/utils/trainer.py`, `utils/optimizer.py` |
| Differentiable collection/training | `embodichain/learning/rl/differentiable_trainer.py`, `collector/` |
| Completed-episode evaluation | `embodichain/learning/rl/evaluation.py` |
| Checkpoint, manifest and policy Viewer | `embodichain/learning/rl/checkpoint.py`, `policy_evaluation/` |
| Official deployment/agent settings | `extensions/embodichain_tasks/configs/tasks/<domain>/<task>/agents/` |

The module training entry remains a compatibility path. Concrete settings come
from the selected config/classes, not example recipes. [Training details](training.md)
own checkpoint/resume, distributed behavior, APG safety and extension steps.
[Data assets](../data-assets/data-assets.md#pretrained-policy-bundles) own downloaded
RUN/cache/revision identity before evaluation uses the same runtime/loader.

## Configuration Resolution

JSON/YAML `trainer`, `policy` and `algorithm` blocks resolve environment first,
then policy/networks and algorithm config, and finally trainer by `rollout_kind`.
Dimensions are checked against built batched spaces. Policies live under
`learning.rl.policies`; generic model builders are not RL policy registries.

## Environment Paths

`trainer.gym_config` discovers installed task packages/init hooks, decodes the
Gym config, applies explicit runtime overrides and builds a registered env.
Omitted renderer preserves file selection. Recording/eval camera demand is
resolved before the first World. Simulator tasks use standard rollouts and
reject differentiable algorithms. Supported task IDs declare RL capability or
use a configuration-owned `rl_task` host; [Gym configuration](../env-framework/configuration.md)
owns that registration. Direct Python callers arrange required discovery.

`trainer.learning_env` selects a registered factory/name/config. Optional online
resources pass by reference and cannot duplicate scalar config keys. Custom
policies may own `from_config` validation. Lightweight environments implement
`LearningEnvProtocol`; differentiable ones also implement `detach_state`, and
scheduled APG supplies the next complete horizon/objective scale. Legacy vector
environment names remain aliases. This path currently rejects distributed mode
and environment profiling; see [differentiable environments](../differentiable-env/differentiable-env.md).

The [Sharpa in-hand example](../../../examples/learning/sharpa_inhand/README.md)
trains cylinder rotation in native EmbodiChain physics. `reference.py` first
trains the pinned upstream teacher, whose saved training config parameterizes
the native run. `native_full.py` / `_full_native_task.py` collect eight-scale,
domain-randomized transitions through `SimulationManager`, physics-owned
parameter writes and exact raw-cache replay. `_apg_learner.py` keeps the
upstream HORA teacher, world model and replay while routing actor updates
through EmbodiChain APG. Native hand self-contact dimensionality and metal
contact parameters still differ from upstream, so results are not strict
upstream physics parity or an algorithm comparison.
`native_full_evaluate.py` checks frozen source receipts before qualifying a
newly trained checkpoint; reference/implementation transfer modes are
diagnostics only. Qualify a freshly trained checkpoint with physical traces.
The sibling [Wuji example](../../../examples/learning/wuji_inhand/README.md)
adapts the same learner to the 20-DOF Wuji hand in run-owned source snapshots.
Its native task requires a palm-up root pose (local palmar +X faces world +Z),
verified initial opposing-pad contacts, authored actuator effort limits and
fixed physical parameters across eight sizes. Differentiable FK gates rewards
by pad geometry and forces, while native termination/evaluation checks actual
opposing-pad participation. Neither route uses the simulator's FK-only
`DifferentiableEnv` or differentiates contact physics.

## Rollout and Trainer Routing

| Kind | Trainer / payload | Algorithms |
|---|---|---|
| STANDARD | Trainer + SyncCollector + TensorDict RolloutBuffer | PPO, GRPO |
| DIFFERENTIABLE | DifferentiableTrainer + graph-connected segments | APG, APGActorCritic |

The standard store reserves a bootstrap slot beyond its transition count and
admits one unconsumed rollout. Actor/critic groups may differ; collection and
evaluation retain the declared ordering and statistics. PPO preserves old
Gaussian parameters for KL. Throughput uses transitions collected by the current
train call, separately from restored cumulative counters.

Differentiable collection does not detach into the standard buffer. Segmented
TBPTT and complete rollouts have different graph lifetimes. Complete mode resets
each independent microbatch, retains the full scheduled horizon, excludes
post-terminal rewards and freezes observation statistics until backward.
`detach_state()` neither resets nor resamples a task. Semantic mask/type fields
remain unnormalized. Actor-critic parameters are disjoint; time truncation needs
the full pre-reset terminal-observation batch, while hard termination suppresses
bootstrap. [Gradient details](training.md#actor-critic-gradients) own critic
normalization and empty-sample behavior.

`DifferentiableTrainer.update_once` exposes one collection/update boundary for
online workflows; its caller owns evaluation and persistence. Native locomotion
keeps task-owned actions/rewards and catalog-owned assets/defaults. Contact
accumulation and selected reset remain [sensor-owned](../sensor-system/contact-history.md).

## Native policy Viewer

`eval-policy --viewer` adapts the original task's reset/step/action/observation
path and owns its camera lifetime. Headless evaluation may use NoRender; Viewer
switches a saved NoRender config to Hybrid unless an explicit native renderer
is selected. Viewer with explicit NoRender rejects before simulation.

Native headless evaluation records the host's terminal outcomes for selected
episodes beside run/checkpoint identities. Viewer summaries include that
snapshot when available; both use host step results and retain scalar metrics.

Task camera presets/target poses stay task-owned. `_viewer_camera.py` handles
framing and tracking/free-view transitions through queued native operations, so
immediate readback may still be the previous rendered pose. Preserve manual
orbit/zoom and recorder view. Use
`tests/learning/rl/policy_evaluation/test_viewer.py` plus native qualification for
camera changes; the [evaluation guide](../../../docs/source/guides/policy_evaluation.md)
owns user controls.

## Invariants and validation

Rollout kind, collector, payload and trainer must agree. APG requires graph-
preserving actions/rewards and row-local adjoint handling. GRPO env counts must
satisfy grouping. Evaluation counts completed asynchronous episodes in an
independent env. Distributed simulator training assigns one device per rank;
only rank zero owns external logging/checkpoints.

Use `tests/learning/` for routing, algorithms, buffers, gradients and checkpoint
contracts; run native/distributed suites only for those resource boundaries.
Missing names point to registries/discovery, disappearing gradients to early
detachment, and reset/terminal issues to the selected environment protocol.
