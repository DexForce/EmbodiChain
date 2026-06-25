# EmbodiChain Release Patterns

## Observed release history

### v0.1.0

- Initial public release.
- Focus: project structure, simulation components, gym components, basic RL, examples, and documentation.

### v0.1.1

- Stability and usability release.
- Focus: observation space correctness, camera pose reporting, launcher cleanup, and documentation updates.

### v0.1.2

- Major platform expansion.
- Focus: RL training refactor, action manager, EmbodiEnv transition, USD import/export, online data streaming, gripper support, and docs for new workflows.

### v0.1.3

- Larger feature release.
- Focus: multi-GPU RL, cloth objects, grasp annotator, action functor modes, mass randomization, CLI entry points, and improved docs.

### v0.2.0

- Platform upgrade release.
- Focus: DexSim v0.4.0 migration, atomic action abstraction, benchmarking, multi-version docs, PyPI release workflow fixes, and dependency bumps.

### v0.2.1

- Incremental platform upgrade.
- Focus: DexSim v0.4.1 migration, sim-ready data generation, emissive light randomization, full-mesh grasp annotation, and GitHub Pages/docs deployment fixes.

## Style patterns

- Releases are tag-based and follow semantic versioning with a `v` prefix.
- GitHub release bodies are structured, not free-form.
- The first paragraph usually summarizes the main theme of the release.
- Dependency upgrades and compatibility breaks are called out explicitly.
- Feature bullets cite the contributor and PR number.
- Documentation and CI/CD fixes are treated as first-class release content.

## Release workflow cues

- Tag pushes trigger the release build path in `.github/workflows/`.
- Release builds produce package distributions.
- Docs builds preserve older versioned documentation when a new tag is published.

