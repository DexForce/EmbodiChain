# Attach to an external World

The attachment fixture exercises a single existing Arena with two robots, one moving cube and a static platform. It uses the installed DexSim wheel and EmbodiChain's UR5 asset and UR5 + DH PGI 140-80 task configuration. Run from the repository root in a Python 3.11 simulator environment. A headless renderer may require Xvfb on Linux.

```bash
xvfb-run -a python -m pytest tests/mcp/test_attached_world.py --run-gpu -v
```

Each case snapshots the host inventory, joint measurements and drive targets, cube state and topology before binding B. It also exercises a partially rejected attachment, a live host edit, planner lookup, topology invalidation, repeated detach and deferred cleanup. The fixture submits scene operations through DexSim's existing dispatcher.

## Dedicated writer acceptance

The fixture owner launches a separate manually stepped World per qualification
run. Choose `dexsim` or `newton` (MuJoCo-Warp) and an unused output directory:

```bash
xvfb-run -a python -m examples.sim.attached_world.control_host --backend dexsim --directory /tmp/ur5-control
```

The host builds both UR5s, cube and platform using the existing DexSim scene
operations before it publishes its session record. The owner retains shutdown
via Ctrl+C or SIGTERM. Ready means the `dsws_*.json` record is present. A
dedicated queue accepts native diagnostic reads and authenticated EmbodiChain
commands. It rejects SDK scripts, unfiltered adapters and other native writes.
Use the 32-byte owner-only grant file with `WriterClient`; the host journal
records a signature for each request. Restrict the host directory to its owner.

```bash
xvfb-run -a python -m pytest tests/mcp/test_world_control.py --run-gpu -v
```

The acceptance test reads A/B via native DexSim, commands the arm on B, disconnects,
reconciles the same command on a new connection, confirms the stop boundary
and reopens native target/measurement data. Each interval is four 10 ms host
updates. Native physics has no absolute step/time readback on this SDK; its
receipt explicitly identifies the dedicated host counter. Newton also supplies
native step/time and the adapter checks the clock delta. A transport failure
leaves a pending file next to the grant. Query the original receipt before new
writes; confirmed partial or unobservable effects require a fresh host. Keep
the discovery record, journal and JSON report for reproducibility.

## MCP and DexPilot

Install `embodichain[mcp]` in the Python environment. Start a fresh dedicated
host for **each** of the following runs. The host framebuffer can use Xvfb;
the MCP client and DexPilot run in a separate terminal. From the checkout root:

```bash
python -m examples.sim.attached_world.verify_control --directory /tmp/ur5-control --interface python --output /tmp/ur5-python.json
# Restart with a fresh host and a new directory before this run.
python -m examples.sim.attached_world.verify_control --directory /tmp/ur5-control-mcp --interface mcp --output /tmp/ur5-mcp.json
```

For interactive use, launch the MCP server with the exact ID and discovery
directory from that host:

```bash
embodichain mcp --session SESSION_ID --session-directory HOST_DIR --writer-key-file HOST_DIR/writer.key
```

The writer option probes the host grant before advertising status, receipt,
target-and-step and stop tools. With no grant the server exposes the existing
read-only robot tool. Set both `--run-id` and `--turn` to label host journal
entries when a Run manages the MCP process.

For the real Agent acceptance install/configure DexPilot, its DSH runtime and
model route, then run the following from the checkout root. Supply the absolute
path to the installed DexPilot library entry if Node cannot resolve the package.

```bash
node examples/sim/attached_world/dexpilot.mjs HOST_DIR /absolute/path/to/python /tmp/pilot-report.json /absolute/path/to/dexpilot/src/index.ts
```

The example targets DexPilot's Node API at DexVerse revision `29d5dd6f5`.
That version builds its ACP MCP list per Run. The example registers EmbodiChain
on that Run instance using its current server-factory hook; this hook is
version-specific. DSH invokes the registered MCP tools in two real turns.
The second turn reconnects through a fresh server, retries the original
command ID and stops. Independent native readbacks bracket the run.
The output includes both Run/turn IDs, client PIDs, the complete host receipt
and journal location. The SDK may truncate large responses in the journal;
the verifier retrieves the authoritative complete receipt by command ID.
Keep the two readbacks and journal with the resulting report. Keep the writer
grant private. Terminate only the host created for the example when finished.
