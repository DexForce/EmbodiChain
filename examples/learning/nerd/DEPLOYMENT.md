# NeRD native solver quickstart

This source-checkout example explicitly selects the NeRD solver and runs the
published Cartpole/Ant models. `train-policy` uses a lightweight learning-env;
it does not exercise native solver deployment. See the [showcase](README.md)
for supervised training and the [validation record](VALIDATION.md) for the
runtime and execution checks behind this example.

## Prepare the environment and exports

Run from the repository root in an environment with the project's dependencies
installed. Refresh bundled extension mappings if needed:

```bash
python -m pip install -e . --no-deps
```

This command registers the bundled packages and `nerd` entry point; it does not
install missing dependencies. The checked CUDA runtime used Python 3.12,
Torch 2.11.0+cu128 and SDK 0.6.0rc2. Native NeRD also supports CPU tensors;
the complete CUDA execution described below was checked separately from CPU
unit coverage. Physical state and model weights use float32.

Download only the reference assets and portable exports, without SL datasets:

```python
from pathlib import Path
import json
import shutil
from huggingface_hub import hf_hub_download
from examples.learning.nerd._resources import (
    _MODEL_REPO, _MODEL_REVISION, _PAYLOAD_MANIFEST_SHA256,
    _require_hash, prepare_reference_assets,
)

prepare_reference_assets()
for robot in ("cartpole", "ant"):
    target = Path("outputs/nerd-native/models") / robot
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "weights.pt"):
        source = hf_hub_download(
            repo_id=_MODEL_REPO,
            revision=_MODEL_REVISION,
            filename=f"neural_dynamics/nerd/{robot}/v1/{name}",
        )
        shutil.copy2(source, target / name)
    _require_hash(target / "manifest.json", _PAYLOAD_MANIFEST_SHA256["model"][robot])
    manifest = json.loads((target / "manifest.json").read_text())
    _require_hash(target / "weights.pt", manifest["weights_sha256"])
```

These imports are example-owned resource helpers, not new public library APIs.
They pin model revision `78e581526570069410bf269b38c0c52c800b24a7` and reference
assets `97e202e41fe7fb504656a6b163291305e400780d`. Assets use the showcase cache
or `EMBODICHAIN_NERD_RESOURCE_ROOT`; the following scene helper uses that same
asset resolver. For an already prepared offline cache, pass
`local_files_only=True` to both download helpers. Never bypass the hash checks
or relabel an export to admit different geometry.

## Run, read back, reset and close

Run this block with `python` from the same repository root. It reuses the
example-owned `DynamicsSimulation` helper, which declares the reference scene
through SDK SceneBuilder and constructs a direct `NeRDProvider`. It creates
no learning environment and selects no analytical solver.

```python
from pathlib import Path
import json
import torch
from examples.learning.nerd._support._scene import DynamicsSimulation

for robot in ("cartpole", "ant"):
    checkpoint = Path("outputs/nerd-native/models") / robot
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    with DynamicsSimulation(
        robot,
        device="cuda:0",
        num_envs=2,
        solver={"solver_type": "nerd", "options": {"checkpoint": str(checkpoint)}},
        num_substeps=1,
        physics_profile=manifest["physics_profile"],
        use_cuda_graph=False,
    ) as simulation:
        initial, _ = simulation.snapshot()
        q0, qd0 = initial["q"], initial["qd"]
        effort = torch.zeros(2, manifest["qd_dim"], device="cuda:0")
        if robot == "cartpole":
            effort[:, 0] = 0.02
        else:
            effort[:, 6:] = 0.02
        first = None
        for step in range(60):
            simulation.step(effort)
            state, contacts = simulation.snapshot()
            assert all(torch.isfinite(value).all() for value in state.values())
            if step == 0:
                first = {key: value.clone() for key, value in state.items()}
        assert (state["q"] - q0).abs().max() > 0
        simulation.reset(q0, qd0)
        reset, _ = simulation.snapshot()
        torch.testing.assert_close(reset["q"], q0, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(reset["qd"], qd0, atol=2e-6, rtol=1e-5)
        simulation.step(effort)
        replay, _ = simulation.snapshot()
        for key in ("q", "qd"):
            torch.testing.assert_close(replay[key], first[key], atol=2e-5, rtol=2e-5)
        print(robot, state["q"].shape, state["qd"].shape, "step/reset passed")
```

Expected state shapes are `(2, 2)` / `(2, 2)` for Cartpole and `(2, 15)` /
`(2, 14)` for Ant. `step` holds a generalized effort for one export interval;
Ant's first six effort columns must be zero. `snapshot` returns owned canonical
state/contact tensors. `reset` clears native inference history and installs the
complete canonical batch. The context manager closes the batch/Scene/World
owners even if a check fails. This helper is NoRender; it is not a viewer demo.

## Provider configuration in another host

The scene construction above is part of the example, not a generic robot
importer. Applications can use either existing SDK configuration form:

```python
from dexsim.engine.newton_physics import (
    ProviderNewtonSolverCfg, RegisteredNewtonSolverCfg,
)
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain_solvers.physics.nerd.provider import NeRDProvider

options = {"checkpoint": "outputs/nerd-native/models/cartpole"}
direct = ProviderNewtonSolverCfg(provider_factory=NeRDProvider, options=options)
registered = RegisteredNewtonSolverCfg(solver_type="nerd", options=options)
physics = NewtonPhysicsCfg(
    device="cuda:0",
    physics_dt=1 / 60,
    num_substeps=1,
    solver_cfg=direct,  # Or registered after installing the entry point.
    collision_cfg=None,
    use_cuda_graph=False,
)
```

This last block configures a backend; it does not build a scene. The host still
owns World/Scene construction, the matching reference articulation, state and
control buffers, stepping and cleanup. Use the reference helper for an entire
runnable example. Provider options are `checkpoint` and optional Boolean
`trainable` (default false); `trainable` does not enable native inference graphs.

## Contracts and common admission errors

| Requirement | Meaning |
| --- | --- |
| Export directory | `manifest.json` plus its hash-matching `weights.pt`; not an SL optimizer checkpoint |
| Physics identity | Recorded joint order, asset/physical signature, COM/velocity and contact schema must match the finalized model |
| Timing | One prediction per step; `physics_dt` equals export `dt`, `num_substeps=1` |
| Geometry | Published exports use `warp_reference`; Ant has its recorded 25 plane-contact slots and friction mapping |
| Controls | Generalized effort / native `joint_f`; no free-base force or external wrench |
| Native execution | No CUDA Graph capture, runtime drive tuning or general native contact-force reports |

A signature/profile error is an incompatible model/scene pairing, not permission
to edit the manifest. A trained-dt error requires matching the export interval.
An unknown registered name usually needs refreshed package metadata; a direct
Provider configuration does not use installed-name discovery.

Learning owns explicit state/history and `detach_state()` boundaries. Native
inference runs under `no_grad`; the optional SDK functional service calls the
same transition with a graph and detaches on explicit publication. Fixed geometry
is cached, graph-connected FK/contact results are not. The private Newton FK
gradient adapter preserves per-call storage and first-order input derivatives;
tensor FK/IK APIs remain in `compute.kinematics` and `learning.layers`.
