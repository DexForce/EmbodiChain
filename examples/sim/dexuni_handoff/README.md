# Use EmbodiChain on a live DexUni World

DexPilot connects to a standard headless, manually stepped DexSim Session.
An operator opts the host into the installed `dexsim.runtime_handoff`
provider at startup. The client describes a finalized World, binds existing
objects by role and topology revision, calls the profile and releases the
same World. The host serializes all changes on its owner thread. Reconnecting
a client preserves the host-side binding and command receipts.

| Demo | Existing objects | Physical acceptance |
| --- | --- | --- |
| UR5 | PGI gripper, selected cube, distractor, two destinations | Lift the selected cube, validate the grasp on a later turn, release on the newly selected platform and measure sustained stability. |
| RP1 | RP1 with carrying constraints, 1.5 kg crate, 0.48 m step, camera, floor | Record crate tracking, two-foot support and a 400-frame camera video. |
| ANYmal-C | Free-base quadruped at 30° yaw, floor | Command 0.4 m/s forward, measure displacement in the initial heading frame, base height and tilt over 100 samples. |

Use a matching DexSim SDK with headless host handoff support, the EmbodiChain
wheel with its MCP dependencies and a configured DexPilot DSH account.
The qualification drivers run on Linux, CPython 3.11, Node 24+ and CUDA.
Start from the examples parent directory. In a wheel installation this is
`$VIRTUAL_ENV/share/embodichain`. Install the pinned model resources following
the [policy catalog instructions](../../learning/policy_runs/README.md).
The preparation uses the RP1 converter from that cache, plus trimesh and
pycollada for the ANYmal visual meshes.

```bash
root=/absolute/path/to/policy-run-assets
entry=/absolute/path/to/dexpilot/src/index.ts
cp examples/sim/dexuni_handoff/live_catalog.json "$root/live_catalog.json"
python -m examples.sim.dexuni_handoff.prepare --catalog "$root/live_catalog.json" \
  --converter-python "$root/assets/rp1/cache/converter/bin/python" --output "$root/scene-assets"
out=$(mktemp -d)
for source in external dexpilot; do
  xvfb-run -a python -m examples.sim.dexuni_handoff.ur5_qualification "$out/ur5-$source" "$source" "$entry" || exit 1
  for profile in rp1 anymal-c; do
    xvfb-run -a python -m examples.sim.dexuni_handoff.policy_qualification \
      --assets "$root/scene-assets" --catalog "$root/live_catalog.json" --profile "$profile" \
      --source "$source" --output "$out/${profile%-c}-$source" --dexpilot-entry "$entry" || exit 1
  done
done
python -m examples.sim.dexuni_handoff.compare "$out"
```

Both sources start with one floor. For `external`, the harness stages the task
objects before DexPilot connects. For `dexpilot`, the agent uses native MCP
tools to add each robot, body, camera or light and configure its initial state.
RP1 assembly imports separate USD components. Each pair uses the same model
files, physics parameters and native initial states. The comparator reads
the ready snapshots, owner journals, motion receipts and sample traces. Keep
each output directory private because it also holds the local writer grant.

To check the lower-level control API on two independent UR5s, run
`python -m examples.sim.dexuni_handoff.qualify --output NEW_DIRECTORY --dexpilot-entry "$entry"`
under the same display environment. This checks joint order, a four-step
command and its replay after a reconnect.
