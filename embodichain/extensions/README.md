# EmbodiChain extensions

The `embodichain` wheel includes the official extensions in
`embodichain.extensions`. Each package has the same layout in source checkouts,
editable installs, and wheel installs.

| Package | Responsibility |
| --- | --- |
| [`embodichain.extensions.tasks`](tasks/README.md) | Official tasks, deployments, and reusable task components |
| [`embodichain.extensions.robots`](robots/README.md) | Concrete robot presets and the official robot catalog |
| [`embodichain.extensions.models`](models/README.md) | Bundled model implementations |
| [`embodichain.extensions.solvers`](solvers/README.md) | Native-physics deployment adapters |
| [`embodichain.extensions.benchmarks`](benchmarks/motion_generation/README.md) | Evaluation suites, planner adapters, scoring policies, and reports |

Add a bundled extension as `embodichain/extensions/<name>/__init__.py`. The
build discovers it as `embodichain.extensions.<name>` and packages JSON, YAML,
URDF, README, license, attribution, and `preview.png` resources under
`embodichain/extensions/`. Declare any additional resource types in
`setup.py` and any plugin entry points in `pyproject.toml`.

Official task configuration references beginning with
`embodichain/extensions/tasks/configs/` resolve from the checkout or installed
wheel. Use the complete Python import path in configuration values that name a
factory or class, such as
`embodichain.extensions.models.dynamics.nerd.sl_recipe:build_supervised_trainer`.

Install from the repository root. Reinstall an existing editable checkout to
refresh its entry-point metadata after upgrading.

```bash
pip install -e . --no-deps
```
