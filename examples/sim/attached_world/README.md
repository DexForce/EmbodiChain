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
