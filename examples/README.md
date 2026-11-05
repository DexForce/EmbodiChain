# Running examples

Use the path of any Python script under this directory, with or without `.py`:

```bash
embodichain run-example --list
embodichain run-example sim/motion/planners/curobo_planner --help
embodichain run-example ods/simple_bc_demo/train --help
embodichain run-example sim/atomic_action --skill place --variants 8
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

## Newton cable demo

Use CUDA and a cable-capable DexSim build. The package version alone does not
prove cable support: the published `0.5.1rc1`, `0.5.1rc2` and `0.5.1` release
snapshots lack the public cable interfaces used by this example. The dependency
range therefore selects the 0.5.1 development series, before its alpha/RC/final
releases. An available Python 3.12 development baseline is `0.5.1.dev10194`, built from
`6dfb756f151d0d704d8a0583afa2c16d245e0ea5`:

```bash
python -m pip install --extra-index-url http://192.168.3.43:8080/simple \
  --trusted-host 192.168.3.43 'dexsim-engine==0.5.1.dev10194'
python -c "import dexsim; from dexsim.scene import SpawnedCable; from dexsim.engine.newton_physics import DexUniSolverCfg; print(dexsim.__version__, dexsim.__commit_id__)"
python -m embodichain run-example sim/demo/cable_viser \
  --num-envs 2 --port 8080 --duration 12
```

The internal package index requires access to the DexForce network. A newer
complete development wheel can also be installed with its own declared
dependencies; record its build SHA. Switching a source branch does not update
an installed wheel. EmbodiChain defaults to DexUni when no solver is selected;
legacy `auto`/`AutoSolverCfg` selections must migrate to `dexuni` or the concrete
solver needed by the scene. Explicit `semi_implicit` remains required for
Newton gradient mode.

## Franka rope wrapping

Run a single-arm floor pickup, post wrap, tightening and two-second hold with
real finger/rope contact. The example includes its default configuration,
reproducible asset generation and physical acceptance checks:

```bash
python -m embodichain run-example sim/demo/franka_rope_wrap \
  --output /tmp/franka-rope-wrap-run
```

See the [Franka rope example](sim/demo/franka_rope_wrap/README.md) for asset
preparation, CUDA/SDK requirements, plan-only execution and result semantics.
