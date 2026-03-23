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

## Using Claude Code for Contributions

[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) is an AI-powered CLI that can assist you throughout the contribution workflow — from understanding the codebase to writing, reviewing, and debugging code.

### Setup

Install Claude Code and authenticate:

```bash
npm install -g @anthropic-ai/claude-code
claude
```

A `CLAUDE.md` file is present at the root of this repository. Claude Code reads it automatically at startup to load project conventions, structure, and style rules, so it is context-aware from the first prompt.

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
> Generate the functor following the project style, with a proper docstring and type hints.
```

**Validate style and formatting before submitting**

```
> Review my changes in embodichain/lab/gym/envs/managers/randomization/visual.py
  for style issues, missing type hints, and docstring completeness.
```

**Write or update tests**

```
> Write a pytest test for the randomize_emission_light function in
  embodichain/lab/gym/envs/managers/randomization/visual.py.
```

**Understand a bug**

```
> I'm getting a KeyError in observation_manager.py at line 42 when env_ids is None.
  What could cause this and how should it be fixed?
```

### Tips

*   Always run `black .` after Claude Code generates or edits Python files — Claude Code can do this for you if you ask.
*   Claude Code respects the `CLAUDE.md` conventions. If you notice it deviating (wrong docstring style, missing `__all__`, etc.), point it out and it will correct the output.
*   For large features, break the work into small, focused tasks and handle them one at a time.
*   Claude Code can help draft your PR description and populate the PR checklist once your changes are ready.

## Contribute specific robots

To contribute a new robot, please check the documentation on [Adding a New Robot](https://dexforce.github.io/EmbodiChain/guides/add_robot.html).

## Contribute specific environments

To contribute a new environment, please check the documentation on [Embodied Environments](https://dexforce.github.io/EmbodiChain/overview/gym/env.html) and see the tutorial below:
- [Creating a Basic Environment](https://dexforce.github.io/EmbodiChain/tutorial/basic_env.html) 
- [Creating a Modular Environment](https://dexforce.github.io/EmbodiChain/tutorial/modular_env.html)

If you want to implement your tasks in a new repo and with some customized functors and utilities, you can also use the [Task Template Repo](https://github.com/DexForce/embodichain_task_template).