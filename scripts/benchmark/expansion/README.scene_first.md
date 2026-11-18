# Conditional layout sampling acceptance

Run the three native task fixtures with:

```bash
python -m scripts.benchmark.expansion.scene_first --pilot --seeds 7 --rollouts 2 --candidates 8 --output outputs/layout-pilot
python -m scripts.benchmark.expansion.scene_first --seeds 7,18,29,40 --rollouts 6 --candidates 48 --seconds 300 --output outputs/layout-comparison
python -m scripts.benchmark.expansion.scene_first --output outputs/layout-comparison --rebuild
```

The collector accepts four strategies for layout comparisons.

| `scene_sampling.strategy` | Proposal |
| --- | --- |
| `bounded_pose` | Draw each declared independent pose in its bounds. A comparison opts into the same layout and task-path screening by providing `layout` and `task_mode`. |
| `scene_first` | Generate the declared layout before checking the robot's task path. |
| `robot_first` | Use a workspace sample to anchor one support group, then generate the remaining layout. |
| `bidirectional` | Alternate scene-first and robot-first by raw ordinal under the same candidate budget. Seed parity chooses the first direction. |

The legacy default stays `bounded_pose`. Use `--cases` and `--strategies` to
select subsets. The four-seed run balances the order within each task and uses
one raw draw per attempt. The report merges coverage cells across seeds and
retains per-seed timings, rejected proposals and source counts in `results.json`.

`layout.support_uid` and `parent_by_uid` select real prepared rigid entities and
an acyclic support tree. The shared scene generator prepares scaled geometry and
conservative support/clearance data once. A proposal fixes optional arena-space
`layout.conditions`, then samples the rest of the allowed layout. It does not
step physics, resize a World or mutate runtime asset topology. It preserves
complete support groups rather than independently mixing their coordinates.

Current group roots support XY translation with authored rotation/height;
member XY/yaw requires an explicitly declared intra-group policy. Root yaw or
Z expansion is rejected rather than silently ignored. Exact member anchors
require their moving ancestors to be anchored. Anchors restrict an existing
domain, not create a new placement permission. No-condition layouts retain the
old addressed RNG stream and continuation behavior. `SceneLayoutCondition`
is also available directly in `embodichain.lab.sim.scene.expansion`.

Robot-first and bidirectional jobs declare `scene_sampling.robot_anchors`. Each
entry selects a `phase_id` and `control_part` from `task_mode`. The target must
refer to an existing, movable layout root or to a member rigidly attached to one.
Earlier task phases must leave that relation intact. For example, the sorting
fixture alternates the right-arm red pickup and left-arm blue pickup as anchors.

The proposal source prepares `sample_count` FK samples per selected control
part, from the declared `workspace_source` (`cache` or `joint_uniform_fk`).
It projects each TCP's XY through the target's authored offset to the initial
group, intersects task and layout bounds, and balances selection across
occupied XY cells. The room occupied by the anchor is reserved during layout
completion. An empty prior spends one raw proposal and rejects that draw. The
next draw in the bidirectional stream still follows its assigned direction.
Height and orientation come from the authored root. The projected pose goes
through full-pose IK and task-path collision screening.

`task_mode` uses existing TaskOperationMode/TaskPhase declarations. Targets name
control parts explicitly and can describe multiple simultaneous arms. Held-object
transforms belong to each incoming phase and must match the intended task; they
are never inferred from a name. The existing feasibility backend selects current
or cached workspace seeds, checks IK, current joint limits and full-pose FK
residuals, and searches collision-checked full-state phase connections. A cache
miss is not a reachability rejection; bounded direct seeds remain available.
A no-solution result means this attempt exhausted its declared search, not proof
that the complete region is unreachable. Every actual Task Program still grounds
and plans again from the measured prepared scene.

`screening` bounds beam_width, ik_seeds, max_rrt_nodes and planning_timeout_s;
it changes preflight effort, not acceptance thresholds. All raw proposals,
discarded candidates and IK/planning calls consume the declared budgets. Failed
layout or task screening remains preparation rejection and never earns physical
success or a sink receipt. Homogeneous W times B replicas share one prepared
prototype and retain a frozen nominal planning start; eight-row collection is
covered by `tests/lab/expansion/test_layout_sampling_sim.py`. Each proposal still
uses the complete task-path screen before normal measured execution. Collision queries own no physical World and are released on
normal completion and startup failures. A topology revision invalidates the
prepared source instead of silently retaining stale geometry.

The fixtures reuse native Task Program execution, physical geometry, controllers,
segment validators and measured terminal/event acceptance. The grab/place and
stack fixtures use a rectangular primitive support with the existing UR5
embodiment. The dual-arm RGB sorting fixture reuses its original program and
dual-UR5 profile: right arm moves red, left arm moves blue, with the green
reference fixed. The arms act sequentially; this does not measure simultaneous
bimanual execution. Its original 5 cm lift and 3.5 / 3 cm XY placement bounds
remain unchanged. Independent acceptance additionally checks both release
states, terminal heights/speeds and the stationary green reference. All
movable topology and physical parameters stay fixed. Only declared XY layouts
vary in the controlled run. Both source strategies share the same task ranges,
path checks, physical thresholds and persistence gate. An initial geometry/path
pass never substitutes physical task completion.

Reports keep proposed/prepared/executed/accepted/persisted counts separately,
record full failed attempts and numerical T+1 evidence, and audit read-back from
confirmed episode artifacts. Declared scene-grid coverage is not a certified
continuous feasible-volume fraction. Task pose diversity is counted separately;
each fixed sorting placement contributes one category rather than novelty from
tracking noise. Both manipulated blocks contribute initial/placement coverage.
For each strategy the report shows cold wall time per seed and per receipt,
including World preparation, screening, execution and export. The separate
screening figure helps identify where time was spent.

The acceptance path rejects unsupported phases or geometry and uses the original
task validation thresholds. RGB sorting exercises both arms in the comparison.
