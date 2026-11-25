# Run a trained policy in a DexSim World

The optional EmbodiChain MCP policy service accepts four versioned IDs for a
robot, a trained policy, a scene and a task. The server advertises the registered
combinations from its catalog. The two examples use the official PartyMimic
RP1 sparse5 ONNX actor and the public ANYmal-C velocity TorchScript actor.

## Prepare the catalog

Run these commands from the EmbodiChain checkout in the same environment that
will run the World. Install the MCP extra and the dependencies for the linked
[PartyMimic example](../partymimic/README.md) and
[ANYmal-C example](../policy_evaluation/README.md).

```bash
catalog_dir=/absolute/path/to/policy-run-assets
mkdir -p "$catalog_dir/assets/rp1"
cp examples/learning/policy_runs/catalog.json "$catalog_dir/catalog.json"
python -m examples.learning.partymimic.prepare --output-dir "$catalog_dir/assets/rp1/model"
ln -s "$HOME/.cache/embodichain/partymimic" "$catalog_dir/assets/rp1/cache"
python examples/learning/policy_evaluation/prepare_resources.py --output "$catalog_dir/assets/anymal"
```

Each profile binds the four IDs to World requirements, local resources and a
Python runner imported by the host. Resource paths are relative to the catalog.
Use a catalog you trust and install the same EmbodiChain version in the stdio
server and the host. Put the checkout on the host's Python import path for
these examples. Keep the discovery and output directories under the operator's
control.

RP1 builds a step-up scene from the prepared robot description and binds the
imported RP1 and crate through `SimulationManager.attach`. A preconverted copy
of that same scene can be registered as `scene_mjcf` and `scene_usd` in the
RP1 profile. The runner applies one 5 ms Newton substep and invalidates any
capture made with a different schedule before running the shared 50 Hz
PartyMimic loop. The result includes the metrics, the control trajectory and,
when enabled, a 20 fps preview. ANYmal-C loads the official motion profile,
builds the flat-ground scene in the supplied World and runs the SDK evaluator
with that borrowed environment. It records displacement, height, tilt and
step counts.

## Mount on DexPilot

Start with a fresh manual DexSim World per task. The service can preflight
before World creation when launched with an explicit per-run discovery
directory. Register the stdio command with an MCP client or configure DexPilot
with an operator-supplied `mcp_servers` entry:

```json
{
  "name": "embodichain",
  "command": "/absolute/path/to/embodichain",
  "args": [
    "mcp", "--session", "{session_id}",
    "--session-directory", "{session_directory}",
    "--policy-catalog", "/absolute/path/to/policy-run-assets/catalog.json",
    "--policy-output", "/absolute/path/to/policy-run-output"
  ],
  "env": []
}
```

Use the following tool sequence in the client's reusable instructions:

1. `embodichain_prepare_run` validates the four IDs and returns a plan and the
   required DexSim World configuration.
2. `embodichain_start_run` takes that plan ID and the fresh World Session ID.
   The actual build and closed loop run on the World's owner thread.
3. Poll `embodichain_get_run` until `done` and present the acceptance metrics and
   artifact paths. The optional wait is bounded to twenty seconds.

For `dexpilot chat TASK_DIR`, copy the included `pack/` directory next to
`TASK_DIR/task.md` to supply these instructions once.

A repeated start for the same plan and Session returns the stored receipt.
A different task uses a new World. Keep the same output directory to recover
the receipt after a stdio reconnect. An interrupted host can leave a queued or
running receipt with unknown effects. Start a fresh World for recovery. The
ordinary `embodichain mcp --session` invocation still exposes its existing
catalog. The dedicated writer host is a separate mode.

An RP1 task can be as short as:

> 让 RP1 用 PartyMimic 搬箱策略在搬箱上台阶场景持箱登上平台并站稳。返回验收和预览视频。
