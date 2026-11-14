# Running examples

Use the path of any Python script under this directory, with or without `.py`:

```bash
embodichain run-example --list
embodichain run-example sim/motion/planners/curobo_planner --help
embodichain run-example ods/simple_bc_demo/train --help
embodichain run-example learning/nerd --help
embodichain run-example learning/rwm --help
embodichain run-example learning/rigidformer demo --help
embodichain run-example sim/atomic_skill --skill place --variants 8
embodichain run-example sim/atomic_skill/coordinated_pickment --help
```

The DexUni pouring example plans a fresh Franka grasp, pour and place sequence
with live fluid physics. It requires CUDA and a DexSim build with DFSPH support:

```bash
embodichain run-example sim/demo/franka_dexuni_pour_demo --help
```

A directory with `__main__.py` can be selected by its directory path. All arguments
after the path go to the script, including `--help`; an optional `--` separator is
also accepted. `--list` discovers Python files from the directory without importing
them. Private files and `__init__.py` are omitted from the list.

Examples keep their existing command-line options and dependency requirements.
The launcher preserves the caller's working directory, so relative input and
output arguments resolve there. Package imports such as `examples.ods...`,
relative imports, sibling imports, exit codes and Ctrl+C reach the selected script.

Add new examples and their resources anywhere in this tree. There is no catalog,
registration entry, special example namespace, or required wrapper function.
Executable scripts should use the usual `if __name__ == "__main__":` entry point.

Source and editable installs run the checkout's examples. Wheels and source
distributions preserve this same tree and its resources; wheels install it under
`share/embodichain/examples/`. The CLI locates the installed data through the
distribution metadata and works outside the repository. Example code is not
installed inside the `embodichain` Python package.

The [NeRD showcase](learning/nerd/README.md) prepares pinned external resources,
trains through the shared SL recipe, exports models and evaluates saved
transitions. Its private support code owns teacher collection, historical
training and reference experiments. Reusable models and codecs are installed
as `embodichain_models`; downloaded datasets, weights and raw reference assets
are stored outside the repository and wheel.

The [SL data-source example](learning/sl_data_source/README.md) connects resident
causal windows to explicit action or state-transition encoders, and describes
caller-owned online sampling and the limits of exact recovery.

The [differentiable FK example](learning/differentiable_fk/README.md) trains a local
Franka joint predictor through fixed FK using TCP pose supervision and the shared
SL trainer. Its browser preview compares initialized and trained predictions on
held-out targets, with frozen inference for interactive target edits.

The [RWM fitting example](learning/rwm/README.md) qualifies controlled supervised
state-transition fitting and state/action gradients. Its generated recipes use
the installable `embodichain_models.dynamics.rwm` model family and shared SL
trainer; its corpus generation and acceptance helpers remain private to the example.

The [visual ACT example](learning/act/README.md) collects Franka PushCube
demonstrations, trains an RGB and proprioception action-chunking policy, and
evaluates checkpoints on independently seeded simulator scenes.

The [RigidFormer showcase](learning/rigidformer/README.md) prepares official
weights and geometry, trains and evaluates the model, and runs an engine demo
with Newton spheres beside a free model rollout. The demo opens a native window
or records an engine-camera video with `--headless`.

The [Atomic Skill example catalog](sim/atomic_skill/README.md) owns all per-skill,
coordinated-manipulation and recovery demonstrations. The two source-checkout
walkthroughs under `scripts/tutorials/atomic_skill/` teach single-arm and dual-arm
API composition; they reuse the canonical example setup.

The [scene and robot feasibility preview](expansion/scene_robot_feasibility/README.md)
compares an authored installation with task-conditioned joint layout/deployment
search in an offline 3D viewer. It replays measured successful episodes, exposes
failed candidates and keeps independent region validation separate.

The [GenSim workflow configuration](gensim/README.md) runs image task planning,
scene generation, native binding and configured Task Program execution through
`embodichain gensim run`.
The [scripted GenSim reproduction](gensim/scripted_pick_place/README.md) rebuilds
fixed table/cube inputs and executes the same binding/deployment workflow,
preserving the measured pickup/placement verdict, graph frontend and wall time.

The [differentiable retiming showcase](learning/differentiable_toppra/README.md)
compares a stopped duration objective with autograd optimization of free path
nodes. Its CPU browser preview displays gradients, finite-difference checks and
trajectories on a shared physical clock, with portable export and replay.

The [camera ground visibility example](sim/camera_ground_visibility.py) uses one
physical ground across four Arenas. Training and debug cameras independently
include it, then change selected environment policies and reset one camera row.
It saves RGB mosaics, depth arrays and capture-policy metadata. This requires the
DexSim camera render-category API; the default camera configuration continues to
work with older SDKs. Ground hidden in a camera still supports the falling box.

The [Expansion Engine showcase](expansion/engine/showcase/README.md) demonstrates
native and real-demonstration-driven experts through the shared engine. Its offline
viewer pairs measured 3D playback with RGB-D, recipes, source clocks and explicit
capability/acceptance results. Full and quick modes preserve unrun and failed cases;
recovery and portability diagnostics keep their evidence separate from expert data.


## Newton cable runtime

The [create-cable tutorial](../scripts/tutorials/sim/create_cable.py) creates a
procedural cable with one fixed segment, steps it under gravity, reads world
segment state and restores the episode creation state. Run it with the native
viewer or with two headless environments:

```bash
python scripts/tutorials/sim/create_cable.py
python scripts/tutorials/sim/create_cable.py --headless --renderer no-render --num-envs 2
```

The cable examples use CUDA and DexSim `0.6.0rc2` (engine commit
`dbaca660471d5aaa7e750093e7fe1c207465c4af`). The DexForce package index
provides Linux x86_64 wheels for Python 3.11 and 3.12. To install the pinned
engine in your EmbodiChain environment and check its build, run:

```bash
python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site \
  dexsim-engine==0.6.0rc2
python -c "import dexsim; from dexsim.scene import SpawnedCable; print(dexsim.__version__, dexsim.__commit_id__)"
```

EmbodiChain defaults to DexUni when no solver is selected;
legacy `auto`/`AutoSolverCfg` selections must migrate to `dexuni` or the concrete
solver needed by the scene. Explicit `semi_implicit` remains required for
Newton gradient mode.

The [cable Viser demo](sim/demo/cable_viser.py) displays replicated segment
capsules in the browser:

```bash
python -m embodichain run-example sim/demo/cable_viser --num-envs 2 --port 8080 --duration 12
```
