# Real2Sim reconstruction and trajectory adaptation

This workflow starts with a real robot video and its recorded joint trajectory.
It reconstructs a tabletop scene with Scene Engine, registers that scene against
the calibrated observation, then adapts the recorded motion through a bounded
agent/replay loop. The original robot is retained.

The executable commands, service configuration, prepared-recording schema, and
output layout are in the
[Real2Sim example: How to run](https://github.com/DexForce/EmbodiChain/blob/main/examples/real2sim/README.md#how-to-run).
In a source checkout, open `examples/real2sim/README.md`. Public interfaces are
documented in the {ref}`Real2Sim API reference <real2sim-video-adaptation>`.

## Processing stages

| Stage | Inputs and behavior | Owner |
| --- | --- | --- |
| Prepare reconstruction image | Frame zero, optional caller-selected same-camera reference frames, robot occlusion recovery, image editing to reduce robot/background/lighting clutter | `gen_sim.real2sim.scene.prepare_scene_input()` |
| Reconstruct | The cleaned image goes through the complete native Scene Engine | `gen_sim.real2sim.scene.reconstruct_video()` |
| Register | Original RGB-D and calibration establish one uniform scene transform; generated shapes and relative object placement are retained | `gen_sim.real2sim.scene.register_scene_export()` |
| Annotate and replay | Prepared joints and fixed scene; semantic annotation, native physics replay, failure localization | `gen_sim.real2sim.execution.run_prepared_real2sim()` |
| Correct | Small structured Cartesian/gripper residuals become numerical trajectories on the source time grid | `data_pipeline.real2sim.adaptation.trajectory_corrections.apply_trajectory_corrections()` |

Reconstruction only requires video. Registration additionally needs calibrated
depth, and trajectory adaptation needs the matching robot model, joints, and
sample-to-video mapping. A raw video is not a complete replay input.

The reconstruction image keeps the edited resolution after provider padding is
removed; it is not reduced to the original low-resolution frame. Registration
maps masks back to the original calibration grid. Recovered observation pixels
and model-generated hidden surfaces retain distinct provenance.

## Agent and execution boundary

The agent supplies semantic annotations, failed phase IDs, and bounded correction
parameters. It cannot supply executable code, new joint arrays, or scene edits.
The numerical module spreads residuals over the recorded sample grid and applies
the feature-specific approximate IK in `adaptation/_real2sim_ik.py`. IK error and
joint-limit excess are diagnostics. Native replay clips commands to actual URDF
limits and records physical states, contacts, and command differences.

The default budget is one baseline and at most two corrected replays. Objects
move through physical contacts after initialization. Source recordings and
registered assets remain fixed during the loop. Low-cost replay cameras are
independent of reconstruction image resolution.

Annotations follow canonical Task Program JSON with separate half-open source
bindings. They remain observational (`execution_ready=false`); executable runtime
bindings and expansion are separate work. A successful agent review is not an
independent physical qualification report.

## Entry point ownership

Reusable inputs, registration, numerical correction, video evidence, and media
encoding belong to `data_pipeline.real2sim.adaptation`. Service calls and workflow
composition belong to `gen_sim.real2sim`. `examples/real2sim` contains thin command
wrappers and the service profile example.

`execution.run_prepared_real2sim()` is the current bounded replay composition.
`pipeline.run_real2sim()` is the earlier contact-fit path with optional source
graph annotation. Both are part of this feature; the latter does not invoke the
agent replay loop. These paths are also distinct from the core normalized motion
prototype and Atomic Skill source-binding interfaces described earlier in the
Real2Sim API reference.
