# EmbodiChain extensions

Official extension modules live here, separate from the core `embodichain/`
package. The directory name describes their role and leaves room for future
extension types.

| Source directory | Python import | Responsibility |
| --- | --- | --- |
| `embodichain_tasks/` | `embodichain_tasks` | Official tasks, deployments, and reusable task components |
| `embodichain_robots/` | `embodichain_robots` | Concrete robot presets and the official robot catalog |
| `embodichain_benchmarks/` | `embodichain_benchmarks` | Evaluation suites, planner adapters, scoring policies, and reports |

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

New bundled extensions should use `extensions/embodichain_<name>/`, declare
their package mapping in the root `setup.py`, and use the relevant discovery
entry point when needed. Placing a directory here does not automatically
register or install it; extensions currently have no separate build metadata.

Official task references beginning with
`extensions/embodichain_tasks/configs/` work from a source checkout and an
installed wheel. The earlier `embodichain_tasks/configs/` prefix remains
supported by the configuration loader.
