# Contributing to EmbodiChain

Thank you for your interest in contributing to EmbodiChain! We welcome contributions from the community to help make this project better.

## Bug report and feature requests

### Bug Report

If you encounter a bug, please use the **Bug Report** template to submit an issue.
*   Check if the issue has already been reported.
*   Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug.md) when creating a new issue.
*   Provide a clear and concise description of the bug.
*   Include steps to reproduce the bug, along with error messages and stack traces if applicable.

### Feature Requests

If you have an idea for a new feature or improvement, please use the **Proposal** template.
*   Use the [Proposal Template](.github/ISSUE_TEMPLATE/proposal.md).
*   Describe the feature and its core capabilities.
*   Explain the motivation behind the proposal and the problem it solves.

## Pull requests

We welcome pull requests for bug fixes, new features, and documentation improvements.

1.  **Fork the repository** and create a new branch for your changes.
2.  **Make your changes**. Please ensure your code is clean and readable.
3.  **Run formatters**. We use `black` for code formatting. Please run it before submitting your PR.
    ```bash
    black .
    ```
    > Currently, we use black==24.3.0 for formatting. Make sure to use the same version to avoid inconsistencies.
4.  **Submit a Pull Request**.
    *   Use the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
    *   Keep PRs small and focused.
    *   Include a summary of the changes and link to any relevant issues (e.g., `Fixes #123`).
    *   Ensure all checks pass.


## Contribute specific robots

To contribute a new robot, please check the documentation on [Adding a New Robot](https://dexforce.github.io/EmbodiChain/guides/add_robot.html).

## Contribute specific environments

To contribute a new environment, please check the documentation on [Embodied Environments](https://dexforce.github.io/EmbodiChain/overview/gym/env.html) and see the tutorial below:
- [Creating a Basic Environment](https://dexforce.github.io/EmbodiChain/tutorial/basic_env.html) 
- [Creating a Modular Environment](https://dexforce.github.io/EmbodiChain/tutorial/modular_env.html)

If you want to implement your tasks in a new repo and with some customized functors and utilities, you can also use the [Task Template Repo](https://github.com/DexForce/embodichain_task_template).

## Using Claude Code for Contributions

<details>
<summary>Setup, skills, and tips for using Claude Code</summary>

[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) is an AI-powered CLI that can assist you throughout the contribution workflow — from understanding the codebase to writing, reviewing, and debugging code.

### Setup

Install Claude Code and authenticate:

```bash
npm install -g @anthropic-ai/claude-code
claude
```

A `CLAUDE.md` file is present at the root of this repository. Claude Code reads it automatically at startup to load project conventions, structure, and style rules, so it is context-aware from the first prompt.

### Skills

Claude Code skills are built-in slash commands that automate common development tasks. They scaffold code, run checks, and enforce project conventions so you can focus on your logic instead of boilerplate. Invoke any skill by typing its command in the Claude Code prompt.

| Skill | Command | Purpose |
|-------|---------|---------|
| Add Functor | `/add-functor` | Scaffold a new observation, reward, event, action, dataset, or randomization functor with the correct signature, style, and module placement |
| Add Task Env | `/add-task-env` | Scaffold a new task environment with the correct file structure, `@register_env` decorator, base class, and test stub |
| Add Test | `/add-test` | Scaffold tests with the correct file placement, style (pytest vs class), mock patterns, and project conventions |
| Pre-Commit Check | `/pre-commit-check` | Run all local CI checks — code style, headers, annotations, exports, and docstrings — before committing |
| Create PR | `/pr` | Create a pull request following the project template and label conventions |
| Benchmark | `/benchmark` | Write benchmark scripts for measuring performance of solvers, samplers, and other computationally intensive components |

#### When to use each skill

**`/add-functor`** — Use when adding a new observation, event, reward, action, dataset, or randomization functor to an EmbodiChain environment. The skill will ask for the functor type and name, then generate the function- or class-style implementation with proper docstrings, type hints, and `__all__` exports.

**`/add-task-env`** — Use when creating a new task environment, including expert demonstration tasks, RL tasks, or any `EmbodiedEnv` subclass. The skill scaffolds the task file with `_setup_scene`, `_reset_idx`, and evaluation logic, plus a test stub.

**`/add-test`** — Use when writing tests for any EmbodiChain module — functors, solvers, sensors, environments, or utilities. The skill determines the correct test file location, style (pytest function vs class), and generates tests with the standard Apache 2.0 header and named constants.

**`/pre-commit-check`** — Run this before committing or creating a PR. It verifies code formatting (`black`), file headers, type annotations, `__all__` exports, and docstring completeness — the same checks the CI pipeline enforces.

**`/pr`** — Use after committing your changes to create a pull request. The skill checks git state, determines the PR type, drafts a description following the project template, runs formatting, creates a feature branch, and opens the PR via `gh` CLI with the correct labels.

**`/benchmark`** — Use when you need to measure the performance of a module (IK solvers, grasp samplers, metrics, etc.). The skill generates a well-structured benchmark script following project conventions.

### Suggested workflows

**Explore the codebase before making changes**

```
> Explain how the Functor/Manager pattern works in embodichain/lab/gym/envs/managers/
> How does the Action Manager work with EmbodiedEnv for RL tasks?
> Show me an example of how a randomization functor is registered in a task config.
```

**Implement a new feature**

```
> I want to add a new observation functor that returns the end-effector velocity.
  Which existing functor should I model it after?
> /add-functor
```

**Validate style and formatting before submitting**

```
> Review my changes in embodichain/lab/gym/envs/managers/randomization/visual.py
  for style issues, missing type hints, and docstring completeness.
> /pre-commit-check
```

**Write or update tests**

```
> /add-test
```

**Understand a bug**

```
> I'm getting a KeyError in observation_manager.py at line 42 when env_ids is None.
  What could cause this and how should it be fixed?
```

**Create a pull request**

After you've made your changes and committed them:

```
> /pr
```

The `/pr` skill will guide you through checking git state, determining the PR type, drafting a description, running formatting, and creating the PR with proper labels.

### Tips

*   Always run `/pre-commit-check` after making changes — it catches the same issues the CI pipeline checks.
*   Claude Code respects the `CLAUDE.md` conventions. If you notice it deviating (wrong docstring style, missing `__all__`, etc.), point it out and it will correct the output.
*   For large features, break the work into small, focused tasks and handle them one at a time using the appropriate skill for each step.
*   If you add a new skill to `.claude/skills/`, make sure to also add it to the Skills table and "When to use each skill" list in this document so contributors can discover it.