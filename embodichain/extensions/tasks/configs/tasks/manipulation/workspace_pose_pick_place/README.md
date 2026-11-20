# Workspace-conditioned task pose coverage

This acceptance task picks a 6×4×4 cm cuboid, lifts it, places it at a live
scene-relative XY/yaw target, releases it and waits for stability. Default CPU
physics, the UR5/DH gripper embodiment and the original controller/validation
owners are retained. The nominal robot installation is fixed. The non-colliding
goal reference is below the support plane; its declared +0.12 m offset identifies
the intended object centre. The two opposing physical gripper fingers have
explicit contact permissions, including the mimic finger.

`expansion.yaml` selects two independent dimensions:

```yaml
scene_sampling:
  strategy: bounded_pose  # bounded_pose | workspace_joint
scheduling:
  policy: fifo           # fifo | task_pose_coverage
```

Existing configurations without these settings retain the original sampler and
sequence. `workspace_joint` requires explicit object-to-TCP target matrices and
selects `joint_uniform_fk` or `cache` as its prior source. It does not silently
substitute a cache when none is configured. Both sources retain the same task
execution, measured validation and sink receipt path. Coverage is optional for
FIFO; feedback scheduling requires explicit `pose_coverage` grids.

The task uses one declared fixed object-to-EEF grasp for all four comparisons.
The joint source verifies that its grasp transform matches the executed policy.
Workspace samples are only proposals: projected XY/yaw is checked again at the
actual height and orientation using IK, joint limits and FK residuals. Runtime
planning and the measured collision/effect gates remain authoritative. A prior
miss retains a labelled uniform fallback; it is not an infeasibility certificate.

The joint candidate remains immutable for its logical ordinal. Feedback affects
only future candidates. Candidate RNG is independent of physical slot assignment;
adaptive selection also depends on the recorded feedback order, so different
parallel schedules need not produce identical future candidate sequences.

## Coverage and accounting

Each fully accepted and persisted episode contributes the measured initial and
terminal cuboid XY/yaw to separate pick/place grids. The cuboid's pi yaw symmetry
is canonicalized. Pick has 144 declared cells; place has 54. These are declared
domain cells, not a certified continuous feasible-volume denominator. Trajectory
variants of the same measured task poses do not increase unique task-pose coverage.

A measured transport guard latches a failure if, after the first 5 cm lift,
the object falls below 2.5 cm above its initial height while more than 3 cm
from the placement goal in XY. Reaching the target later cannot erase that
failure. This control-boundary guard complements collision/effect validation;
it is not a continuous-contact or substep retention certificate.

The domain fingerprint includes robot/installation, physical object configuration,
validation settings, ranges and grids. It excludes seed, sampling strategy and
individual candidate poses. SceneCase identities remain distinct. Existing session
receipts own acceptance: pending or failed writes do not increase confirmed counts.
Success-rate feedback records actual executions; preflight failures remain a
separate class. Observed TCP poses are auxiliary measurements, not extra points in
the primary score.

`pose-sampling.json` retains raw proposals, provenance, selections, outcome order,
resource counts, confirmed coverage and a source checkpoint. Checkpoint loading
reproduces proposal state but is **not** a full live-World or job-resume facility;
ordinary session/sink reconciliation must resolve pending writes before continuing.
The collector still requires a fresh output directory.

The source scores 75% of eligible choices by Beta(1,1) success estimate times
scarcity, with coarse paired pick/place regions; 25% use the underlying proposal
stream. The workspace prior independently reserves 25% uniform proposals. These
are different probabilities and are recorded separately. Pools are bounded at 64;
all generated candidates and decorated IK/planning calls consume budgets, including
unselected candidates. Physical rollouts have a separate cap from preparations.
Calls are cooperatively bounded, not forcibly preempted inside native code.

## Acceptance runs

From the repository root:

```bash
python -m scripts.benchmark.expansion.task_pose_coverage --suite pilot --output outputs/pose-pilot
python -m scripts.benchmark.expansion.task_pose_coverage --suite smoke --output outputs/pose-smoke
python -m scripts.benchmark.expansion.task_pose_coverage --suite formal --output outputs/pose-formal
python -m scripts.benchmark.expansion.task_pose_coverage --suite installation --output outputs/pose-installations
python -m scripts.benchmark.expansion.task_pose_coverage --suite parallel --output outputs/pose-b4
python -m scripts.benchmark.expansion.task_pose_coverage --suite trajectory --output outputs/pose-trajectories
python -m scripts.benchmark.expansion.task_pose_coverage --output outputs/pose-formal --rebuild
```

A/B/C/D are bounded+FIFO, bounded+feedback, workspace+FIFO, workspace+feedback.
Pilot uses fixed nominal poses and must pass physical acceptance before broad-domain
claims. Smoke uses 12 rollouts per group. Formal runs use both open/obstacle scenes,
five paired seeds, 4096 raw candidates, 128 rollouts, 32768 IK calls, 4096 planning
calls and 1800 seconds per cell. Initialization and prior preparation are included
in end-to-end wall time; pose budgets begin with source construction. Process
startup/import is recorded by the orchestrator log but excluded from worker wall
windows. No warmup is subtracted from these cold measurements.

The target is a median D/C unique-cell gain >=15%, positive in >=4/5 seeds,
without reduced median pick or place coverage or more than 5 percentage points
acceptance-rate loss. A failed target remains failed; no thresholds are relaxed.
Installation experiments are separate domains; they cannot inflate the nominal
installation score. Obstacle placement and physical criteria are fixed before
formal runs. Run `--media` only for separate illustrative replay; numerical T+1
evidence and confirmed artifact/LeRobot export are always retained.

`--suite parallel` verifies real B=4 operation but is not an assertion of identical
adaptive sample order to B=1. Deterministic source checkpoint tests cover feedback
replay and owned candidate reuse. Report wall time, physics outcomes, failures and
coverage separately; never infer physical success from program completion alone.
