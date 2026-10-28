# Running examples

Use the path of any Python script under this directory, with or without `.py`:

```bash
embodichain run-example --list
embodichain run-example sim/motion/planners/curobo_planner --help
embodichain run-example ods/simple_bc_demo/train --help
embodichain run-example learning/nerd --help
embodichain run-example learning/rwm --help
embodichain run-example sim/atomic_action --skill place --variants 8
embodichain run-example sim/atomic_action/coordinated_pickment --help
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

The [RWM fitting example](learning/rwm/README.md) qualifies controlled supervised
state-transition fitting and state/action gradients. Its generated recipes use
the installable `embodichain_models.dynamics.rwm` model family and shared SL
trainer; its corpus generation and acceptance helpers remain private to the example.

The [Atomic Skill example catalog](sim/atomic_action/README.md) owns all per-skill,
coordinated-manipulation and recovery demonstrations. The two source-checkout
walkthroughs under `scripts/tutorials/atomic_action/` teach single-arm and dual-arm
API composition; they reuse the canonical example setup.
