# EmbodiChain Agent Instructions for GitHub Copilot

EmbodiChain keeps canonical agent skills under `.agents/skills/`.

## Project Context Routing

Use `.github/copilot/project-dev-context.md` as the local entry adapter, then
follow the canonical routing rules in `.agents/skills/project-dev-context/`.

## Common Skills

- Add task environments: `.github/copilot/add-task-env.md`
- Add functors: `.github/copilot/add-functor.md`
- Add atomic actions: `.github/copilot/add-atomic-action.md`
- Add robots: `.github/copilot/add-robot.md`
- Add tests: `.github/copilot/add-test.md`
- Run pre-commit checks: `.github/copilot/pre-commit-check.md`
- Draft or create pull requests: `.github/copilot/pr.md`
- Write benchmarks: `.github/copilot/benchmark.md`

