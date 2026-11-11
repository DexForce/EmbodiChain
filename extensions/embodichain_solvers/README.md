# Official solver integrations

`embodichain_solvers` ships in the main `embodichain` distribution with the same
version and dependencies. It owns concrete physics deployment adapters;
`embodichain_models` owns networks, codecs and model training recipes. Generic
`DifferentiableEnv`, functional Scene forwarding and the Newton FK gradient
bridge remain in `embodichain.lab`. Core modules do not import this extension.

The NeRD provider lives in `embodichain_solvers.physics.nerd.provider` and is
registered as `nerd` through the existing `dexsim.newton_solvers` entry point.
Its private `_binding.py` owns the model-specific joint/contact ordering and
physical signature. Package imports are inert; selecting the provider imports
its model and SDK dependencies, and constructing a solver loads the export.
The public module exports `NeRDProvider` and `NeuralRobotDynamicSolver`.

Use the existing Newton configuration, for example:

```yaml
solver_cfg:
  solver_type: nerd
  options:
    checkpoint: /path/to/scene-matched/export
```

The adapter supports the recorded Cartpole/Ant domains and validates the export
against the Scene before stepping. Physics signatures and model qualification
are unchanged by relocating source. Collection, APG factories and research
workflows stay in [the NeRD showcase](../../examples/learning/nerd/README.md).
Refresh editable installation mappings from the repository root after moving
or adding extensions. No separate distribution or solver registry is needed.
