# Image-to-execution GenSim workflow

Configure the required [model services](../../embodichain/gen_sim/services/README.md),
install the GenSim optional dependencies, and supply a real tabletop photograph:

```bash
embodichain gensim run --config examples/gensim/run.yaml \
  --image /absolute/path/to/table.png \
  --instruction "Put the cup on the tray" \
  --output outputs/gensim-run-001
```

For an existing scene export, add `--scene /absolute/path/to/scene-workspace`
and supply its corresponding original photograph. The scene stays read-only;
the new run receives its own task, native assets, deployment and execution report.

`runtime.dual_ur5.yaml` supplies the environment timing and an existing robot
component with declared resources, gripper commands, grasp generation and measured
grip evidence. Its installation pose is an example: adjust the component's allowed
`init_pos`/`init_rot` overrides to put the generated scene in reach. The selected
execution policy must have a matching embodiment contract and
`effect_assurance: verified`. The workflow keeps the file-owned physics backend.

The photograph, service outputs and robot installation determine whether native
planning and physical execution succeed. A failed grasp, unavailable capability,
unsupported target or exhausted episode remains a failed run with diagnostics.
These example coordinates do not constitute physical qualification of arbitrary
generated scenes.

See the [command and output guide](../../embodichain/gen_sim/README.md) for the
supported automatic bindings, optional task-owned binding declarations and the
distinction between execution results and normative TaskSpec certificates.
