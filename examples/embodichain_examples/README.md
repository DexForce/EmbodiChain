# Runnable examples

The examples in this directory are installed as `embodichain_examples` by the
main wheel. Their scenes, robot fixtures, orchestration and artifact handling
remain outside the `embodichain` core package. They can run from any working
directory after installation.

```bash
embodichain run-example --list
embodichain run-example atomic-action --help
embodichain run-example atomic-action --skill place --variants 8
```

The core command in `embodichain/cli/run_example.py` reads the lightweight
`embodichain.examples` entry-point catalogs and imports only the selected
example. Listing examples does not import simulation, scenes or model code.
Arguments after the example name belong to that example, including `--help`.

To add a bundled example, place its Python package here and declare a name,
description and `module:callable` target in `_catalog.py`. The callable accepts
an argument sequence and owns its parser and resource lifecycle. New examples
need no change to the core launcher. Other distributions can provide the same
catalog contract through an `embodichain.examples` entry point.

The [Atomic Skill example](atomic_action/README.md) provides twelve fixed-scene
skills with optional affordance and trajectory expansion. The existing
`run-atomic-action` command forwards to this example for compatibility.
