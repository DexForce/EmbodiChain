# EmbodiChain extensions

Official extension modules live here, separate from the core `embodichain/`
package. The directory name describes their role and leaves room for future
extension types.

| Source directory | Python import | Responsibility |
| --- | --- | --- |
| `embodichain_tasks/` | `embodichain_tasks` | Official tasks, deployments, and reusable task components |
| `embodichain_robots/` | `embodichain_robots` | Concrete robot presets and the official robot catalog |
| `embodichain_benchmarks/` | `embodichain_benchmarks` | Evaluation suites, planner adapters, scoring policies, and reports |
| `embodichain_models/` | `embodichain_models` | Bundled model implementations |

Keep the `embodichain_` prefix on extension modules and Python packages. It
identifies their ecosystem outside this repository and avoids ambiguous
top-level imports such as `tasks` or `robots`. The parent `extensions/`
directory is only a source container, not a Python package or an import prefix.

These modules currently ship in the main `embodichain` distribution, sharing
its version and dependencies. Install or refresh the editable mappings from
the repository root after moving or adding a module:

```bash
pip install -e . --no-deps
```

New bundled extensions should use the nested layout
`extensions/embodichain_<name>/embodichain_<name>/__init__.py`. The root build
automatically includes these packages in the main distribution. The existing
flat layout, `extensions/embodichain_<name>/__init__.py`, is also supported;
each extension must use exactly one layout. Package names must be valid Python
identifiers. The build rejects ambiguous and missing package roots.

Non-Python resources still need explicit `package_data` rules in `setup.py`;
runtime discovery entry points, when needed, are declared in `pyproject.toml`.
Including a package does not register it as a plugin. Extensions currently have
no separate build metadata.

Official task references beginning with
`extensions/embodichain_tasks/configs/` work from a source checkout and an
installed wheel. The earlier `embodichain_tasks/configs/` prefix remains
supported by the configuration loader.
