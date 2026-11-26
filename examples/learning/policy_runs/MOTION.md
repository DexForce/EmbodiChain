# UR5 motion MCP session

This demo uses the same UR5 + DH PGI 140-80, cube, platform, Task Program,
and validators as the registered UR5 pick-and-place. It runs one manually
stepped DexSim Native World with one Arena and a dedicated writer. The World
owner hosts the job cursor, calibrated grasp state, and persisted job receipts.
A DexPilot Run can mount the EmbodiChain motion service and the DexSim native
observations for the same Session.

| Interface | Contract |
| --- | --- |
| `embodichain_list_capabilities` | Scoped robot, control groups, frame, planner profile, registered skill goals, and writer capability |
| `embodichain_fk`, `embodichain_ik` | Arena-frame TCP with metres and XYZW quaternion; arm joints identified in configured order and radians |
| `embodichain_plan_motion`, `embodichain_execute_motion` | Plan from named joint/TCP waypoints using MotionGenerator and the trapezoidal planner. Retiming uses the control grid; checks cover joint position and velocity limits. Collision checking is outside this planner profile's qualification. Execution rechecks the measured robot, drive targets, moving bodies, topology, and physics step. |
| `embodichain_start_skill`, `embodichain_get_grasp` | `pick_up(cube)` yields one verified grasp. `place(cube, grasp_ref, platform_top)` uses the same host and checks the measured TCP-relative relation and closed gripper again. |
| `embodichain_prepare_program`, `embodichain_start_run` | Resolve and execute the registered `cube_platform_pick_place` Task Program as one job. |
| `embodichain_get_run`, `embodichain_cancel_run` | Durable receipt with phase, control/physics counts, terminal status and physical verdict; confirmation occurs at the control boundary. |

Use launch-time MCP profile arguments `--session ID --session-directory DIR
--motion-output OUTPUT --writer-key-file FILE`. The operator installs a
dedicated DexSim command table. Scene inspection uses native DexSim getters.
EmbodiChain coordinates actions and advances at most one configured control
interval per owner-thread callback. Keep the writer grant and receipt root
owner-controlled. Operator assembly and lifecycle use the supplied host.

A cancelled job preserves the last applied drive target on the paused World.
An interrupted skill after execution begins fences the partial episode;
start a fresh fixture to retry. Cancellation requested before Place begins
keeps the original verified grasp. A detected physical loss revokes the grip
credential for that World. Repeated command IDs retrieve the original receipt
and conflicting reuse raises `COMMAND_ID_CONFLICT`. An offline stdio service
can reopen persisted receipts scoped to the original Session ID.

## Mount the two MCP servers

For the dedicated motion World, the DexSim MCP server provides status,
articulation and rigid-body observations, and RGB capture from the bound
demo camera. EmbodiChain owns planning, skill and Task Program requests,
exclusive control and stepping, verified grasp references, and run receipts.
The DexSim World owner assembles the fixture, controls its lifetime, and
installs the dedicated command table.

Configure DexPilot with its per-Run stdio server list. The existing
`agent_python` and `agent_sdk` settings expose DexSim MCP. Add the following
entry to `mcp_servers` in the operator's DexPilot settings. Use absolute
paths for the Python environment, checkout, output root, and private writer
grant. DexPilot expands the two braced values for the selected Session.

```json
{
  "name": "embodichain",
  "command": "/absolute/path/to/python",
  "args": [
    "-m", "embodichain", "mcp",
    "--session", "{session_id}", "--session-directory", "{session_directory}",
    "--motion-output", "/absolute/path/to/session-output/receipts",
    "--writer-key-file", "/absolute/path/to/session-output/sessions/writer.key"
  ],
  "env": [{"name": "PYTHONPATH", "value": "/absolute/path/to/EmbodiChain"}]
}
```

The two-turn launcher below provisions that configuration in memory for its
own DexPilot Run.

## Run the two-turn DexPilot demo

Use a Python 3.11 environment with the EmbodiChain `[mcp]` extra,
`dexsim-engine==0.6.1rc1`, and `imageio[ffmpeg]`, plus a host with GPU
rendering and Xvfb. Use a configured DexPilot DSH route with per-Run stdio
mounting. The recorded run used Node.js 26 and DexVerse `50001247a`
(MR !1687). Make the UR5 and DH PGI 140-80 assets from the shipped cube-platform
deployment available to the host. Run from the checkout with a fresh output
directory and an available GPU.

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a node examples/learning/policy_runs/dexpilot_ur5_motion.mjs \
  /absolute/path/to/dexpilot/src/index.ts /absolute/path/to/python /absolute/path/to/new-output
```

The first Agent turn lifts the cube and holds it. The launcher closes the
DexPilot connection, then opens a second turn on the same World. The Agent
recovers the Pick receipt, checks the grasp, and commits Place. The launcher
confirms matching Session IDs, distinct MCP client processes, unchanged
physics steps between Pick completion and the reconnect check, and all
physical placement criteria. `summary.json` references the trajectory,
video, per-run receipts, and native World journal. Read `summary.json`
and substitute the value of `pick.run_id` below to score the trace again.

```bash
/absolute/path/to/python -m examples.learning.policy_runs.verify_ur5 \
  '/absolute/path/to/new-output/receipts/<pick.run_id>'
```

The first version qualifies only the declared UR5 grasp and support target
in the bundled Default-backend fixture. A different robot, scene, or task
can implement the same host job protocol with its own measured evidence.
