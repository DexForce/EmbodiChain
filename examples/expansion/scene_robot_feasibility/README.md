# Joint scene and robot installation search

Run from the repository root with the installed simulation runtime:

```bash
pip install -e '.[feasibility]'
embodichain search-scene-feasibility --config examples/expansion/scene_robot_feasibility/search.ur5.yaml --output outputs/scene_feasibility
```

The authored task picks, lifts, places and releases one cube. The job changes the
cube layout and UR5 installation while preserving the robot/tool, goal and
measured acceptance criteria. It checks an explicit phase graph before invoking
the existing Task Program runtime; actual runtime replanning and physical outcome
still decide acceptance. Structural contact permissions name physical URDF
frames and must be re-authored when the embodiment changes.

`atlas.json` contains exact proposals, independent held-out trials and conditional
joint regions. `session.json` retains shared attempt/receipt accounting. Accepted
`episodes/*.npz` contain T controller targets with T+1 measured state/timestamp
samples. Failures stay diagnostic evidence. `report.html` is an offline slice
inspector; the remaining dimensions are explicit conditions, and outlined boxes
are proposal regions rather than continuously certified sets.

The example allows 600 seconds, eight search candidates and two held-out trials
per selected region. These small counts are a smoke configuration, not evidence
of a high population success rate. Search and validation are bounded; finding no
successful region is a valid result. The general API defaults to eight independent
validation trials per region and reports an exact 95% interval only for a complete
fixed-count allocation.

For composed dual arms, define both simultaneous TCPs in a phase and move the
whole robot installation as one group. A Python `TaskProgramFeasibilityHost` can
bind multiple operation modes through `configure_mode` and supply richer semantic
or event validators. The configured adapter accepts one runtime mode, preserves
control-part names, and requires explicit runtime provenance-to-phase mappings.

The UR5 job enables a workspace prior for the declared grasp target. A bounded
joint-uniform FK population is prepared when no workspace cache is configured;
valid cached qpos/FK data can be reused instead. Half of prior requests retain
uniform exploration, and every proposed layout still runs fresh IK/path/physical
checks. The report records prior identity and guided/fallback counts.

`search.dual_ur5.yaml` exercises the existing verified tabletop HandOver program,
with paired preflight reach requirements and explicit measured transfer/release
events. The canonical runtime owns grasp choice and the full handover/placement
motion; native qualification must be assessed from its resulting atlas, rather
than inferred from the preflight endpoints alone.
