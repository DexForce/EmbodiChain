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
    > Currently, we use black==26.3.1 for formatting. Make sure to use the same version to avoid inconsistencies.
4.  **Check public API documentation coverage**. The checker is read-only and
    verifies that exports declared through `__all__` appear in the Sphinx API
    reference.
    ```bash
    python docs/scripts/check_api_docs.py
    ```
    If it reports missing exports, update the appropriate API-reference page
    and public docstrings. Agent users can invoke `/update-api-docs` to generate
    these changes.
5.  **Submit a Pull Request**.
    *   Use the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
    *   Keep PRs small and focused.
    *   Include a summary of the changes and link to any relevant issues (e.g., `Fixes #123`).
    *   Ensure all checks pass.


## Contribute specific robots

To contribute a new robot, please check the documentation on [Adding a New Robot](https://dexforce.github.io/EmbodiChain/main/guides/add_robot.html).

## Contribute specific environments

To contribute a new environment, please check the documentation on [Embodied Environments](https://dexforce.github.io/EmbodiChain/main/overview/gym/env.html) and see the tutorial below:
- [Creating a Basic Environment](https://dexforce.github.io/EmbodiChain/main/tutorial/basic_env.html)
- [Creating a Modular Environment](https://dexforce.github.io/EmbodiChain/main/tutorial/modular_env.html)

If you want to implement your tasks in a new repo and with some customized functors and utilities, you can also use the [Task Template Repo](https://github.com/DexForce/embodichain_task_template).

## Using AI Coding Agents for Contributions

EmbodiChain supports both [OpenAI Codex](https://developers.openai.com/codex/cli) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code/getting-started). Either agent can help explore the codebase, implement focused changes, write tests, review diffs, and prepare pull requests.

### Setup

Follow the official setup guide for your preferred agent, then start it from the repository root:

| Agent | Setup guide | Start command |
|-------|-------------|---------------|
| OpenAI Codex | [Codex CLI](https://developers.openai.com/codex/cli) | `codex` |
| Claude Code | [Claude Code setup](https://docs.anthropic.com/en/docs/claude-code/getting-started) | `claude` |

The repository uses [`AGENTS.md`](AGENTS.md) as the canonical source for project structure, conventions, and contribution instructions. Codex reads it directly; Claude Code reads [`CLAUDE.md`](CLAUDE.md), which imports the same instructions. Run either agent from the repository root so it can discover these files and the project skills.

### Agent development context

Agent-facing development context lives in [`agent_context/`](agent_context/). The registry at [`agent_context/MAP.yaml`](agent_context/MAP.yaml) maps topic IDs, aliases, and keywords to focused Markdown files. When a task depends on project internals, ask the agent to reference the relevant project context before making changes, for example:

```text
Reference the project context for manager-functor before implementing this change.
```

Both agents are instructed to read `agent_context/MAP.yaml` first, resolve the requested topic, and load only the matching context files. For codebase-navigation questions they also verify mapped paths against the current source tree and fall back to live search when no topic matches. The files under `docs/source/` remain the human-facing Sphinx documentation and should be consulted only when explicitly requested.

### Shared project skills

Canonical skills live in [`.agents/skills/`](.agents/skills/). Claude Code uses thin adapters under [`.claude/skills/`](.claude/skills/) that point back to the same instructions, so both agents follow a consistent workflow. Ask the agent to use the relevant skill by name; common examples include:

| Skill | Purpose |
|-------|---------|
| `/project-dev-context` | Navigate, resolve, or update agent development context |
| `/add-functor`, `/add-task-env`, `/add-robot`, `/add-solver`, `/add-atomic-action` | Scaffold project components following repository conventions |
| `/add-test`, `/benchmark` | Add validation or performance benchmarks |
| `/update-api-docs` | Generate API-reference entries and descriptions for missing public exports |
| `/pre-commit-check` | Run proportional checks before committing |
| `/pr`, `/release` | Prepare a pull request or release |

Review all agent-generated changes, run the relevant tests, and use `/pre-commit-check` before submitting a pull request.
