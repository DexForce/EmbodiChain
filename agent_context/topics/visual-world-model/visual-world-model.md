# Visual world model

`embodichain_models.dynamics.lewm` owns the image encoder, action-conditioned
latent predictor, SIGReg objective, HDF5 window source, weights-only export and
CEM planning. It has no simulator dependency. The installable
`build_lewm_trainer` factory uses common `learning.sl` orchestration and
rejects multi-process training: rank-local SIGReg statistics do not implement a
global batch objective. `examples/learning/lewm/` owns pinned reference
comparison, official OGBench-Cube migration checks, historical UR push-cube
collection and paired simulator evaluation. `official-reference` compiles a
pinned MuJoCo reference and exports the robot/physical contract;
`official-kinematics` checks that robot in DexSim without stepping dynamics.
The example's [official alignment plan](../../../examples/learning/lewm/OFFICIAL_CUBE.md)
owns the task parameters and remaining physical/evaluation gates. Historical
UR reachable-query success does not establish official-task alignment.
Its validation diagnostics compare fixed physical candidates; frozen-visual
predictor ablations use the common SL trainer and source-bound latent caches.
CEM sampling and sequence-selection options are recorded evaluation contracts.
`official-eval-reference` uses the pinned original evaluator;
`official-eval-dexsim` keeps native worlds alive across planning barriers while
one planner preserves global query/RNG order. Physical replay can reuse the
reference's raw actions. Planning observation history is independent of the
network's maximum trained context. Small pointwise neural differences can change
CEM elite selection, so numerical tolerances alone do not prove identical plans.
Training-statistics sampling still supplies raw controller actions to the model;
final-mean selection reports the recomputed cost of that complete sequence.
Standardized-space sampling, mean-candidate insertion, optional clipping and
zero minimum sampling deviation express the upstream CEM contract explicitly;
the official recipe selects these options without changing legacy defaults.
Historical reports without these options retain the original defaults, while
aggregation rejects mixed planning configurations.

Episodes contain T+1 RGB frames and T raw actions; each action leaves the
corresponding frame. Windows stay within episodes. The source identity binds
the file content, split, window shape and optional action-block stride. Blocked
windows sample strided frames and concatenate every intervening raw action;
the portable model configuration binds that block length. Optional bounded
uint8 device caches preserve source identity, RNG and independent batch ownership;
their resident-memory limit excludes output batches and model activations.
Training/validation seeds must be
disjoint. Action statistics are computed only from training episodes and saved
with model weights; exported metadata binds the camera, control period and
action and rendering contracts. Evaluation rejects missing or mismatched render
contracts unless explicitly permitted and reported as unverified. The example
uses the project's default offscreen DLSS rendering; disabling denoising is an
explicit diagnostic mode. Optional
simulator traces retain per-step observations, physical states, actions and
planner RNG; `check-trace` rechecks saved decisions without claiming identical
simulator replay. Rollout receives H observed latents, H-1 executed actions and
strictly future candidate actions. The model does not produce decoded pixels
or physical state.

`LeWMCEMPlanner.plan_sequence` exposes complete model-action sequences for host
execution. The example's measured-future goal route owns reference collection,
replay reachability checks and raw/blocked execution. It restores snapshots owned
by the current world, preserves real prefix observations/actions in traces and
compares random and zero-action controls. It never imports serialized snapshot
owner tokens. These reachable-query reports remain separate from rendered-goal
acceptance and privileged geometry never enters model inputs.

Goal-image rendering restores physical state without stepping simulation and
warms renderer history before evaluation observations. Candidate re-scoring
requires unchanged encoder/projector tensors and action statistics so the saved
observed latents remain comparable. Diagnostic geometry is never a policy input.

The example keeps reference weights and simulator datasets outside the wheel.
Upstream numerical equivalence, held-out latent error and paired physical
success are different validation claims; see its README for commands and
resource revisions. Focused model tests live in `tests/models/dynamics/lewm/`.
