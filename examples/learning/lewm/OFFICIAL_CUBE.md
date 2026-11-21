# OGBench single-Cube contract

The [reproduction example](cube/README.md) prepares the fixed data and source
revisions. Resources and runtime requirements are in
[DEPLOYMENT.md](cube/DEPLOYMENT.md). Measured outcomes and experiment limits are
in [VALIDATION.md](cube/VALIDATION.md).

## Scene and controller

The example's DexUni environment uses the compiled UR5e and Robotiq 2F85
asset export and five-dimensional controller.

| Boundary | Setting |
| --- | --- |
| Robot | Six arm joints and eight gripper joints, with two point CONNECT constraints, one joint equality and a split finger tendon |
| Policy action | Relative XYZ, yaw and normalized gripper opening |
| Action scales | `[0.05, 0.05, 0.05, 0.3, 1.0]` |
| Cartesian workspace | Lower `[0.25, -0.35, 0.02]`, upper `[0.6, 0.35, 0.35]` with a down-facing pinch frame |
| Timing | 2 ms per physics step, 25 physics steps per control, 200 controls per expert episode |
| Rigid-body solver | DexUni CPU, Newton iteration, implicit-fast integration, elliptic cone |
| Cube | 4 cm side, 79.36 g mass |
| Camera | `main_cam`, calibrated from the upstream `front_pixels` camera at 224×224 |
| Success | Cube center within 4 cm in 3D after an executed control |

The asset exporter records the compiled actuator, inertial, contact and
equality values in `physical-contract.json`. Passive forces and the split
gripper tendon use the source control law. The native camera uses the
compiled URDF colors and exported floor texture.

## Training and evaluation

The prediction model accepts five raw commands per model step and has a maximum
history of three observed frames. The official CEM run begins with a single
frame, samples 300 candidates across 30 iterations with 30 elites, and plans
five blocks of five controls. Each initial state and its 25-step future goal
are restored and rendered in the native scene. The first planned 25 controls
execute before replanning from the live camera. Each query has a maximum
budget of 50 controls.

The example reconstructs the official seed-42 query selection and checks the
source archive and query digests. Source episodes used for the fixed queries
and the development set are excluded from the new training and validation
splits. The world model uses RGB and standardized actions during planning.
Object position is used for the physical success check.

Use `embodichain run-example learning/lewm --help` for the upstream MuJoCo
asset and evaluation commands. The [Cube example](cube/README.md) runs the
corresponding DexUni evaluation.
