---
name: pr
description: Create single or stacked pull requests for EmbodiChain following the project's PR template and conventions, including proportional validation, dependency ordering, and proper GitHub repository labels
---

# EmbodiChain Pull Request Creator

This skill guides you through creating a pull request that follows the EmbodiChain project's PR template and contribution guidelines.

## Usage

Invoke this skill when:
- You have completed a feature, bug fix, or other change and want to create a PR
- You want to split a large, dependency-ordered change into stacked PRs
- You want to ensure the PR follows the project's conventions
- You need help drafting a proper PR description

## Steps

### 1. Check Current State

First, check the current git status and changes:

```bash
git status
git diff HEAD
```

### 2. Determine Change Type

Based on the changes made, select one of these PR types:

- **Bug fix** - Non-breaking change which fixes an issue
- **Enhancement** - Non-breaking change which improves an existing functionality
- **New feature** - Non-breaking change which adds functionality
- **Breaking change** - Existing functionality will not work without user modification
- **Documentation update**

### 3. Draft the PR Description

Write a description that includes:

- **Summary**: A clear, concise summary of the change
- **Issue reference**: Which issue is fixed (e.g., "Fixes #123")
- **Motivation and context**: Why this change is needed
- **Dependencies**: List any dependencies required for this change

### 4. Select Proportional Validation

Inspect the changed files and validate only the affected behavior. Do not run the
full test suite automatically.

- Workflow-only changes: run `actionlint` on the changed workflows and any
  directly related workflow-script tests.
- Documentation-only changes: run the relevant docs checks or build; do not run
  Python tests unless executable examples or docs tooling changed.
- Isolated Python changes: run the nearest matching test module or package.
- Shared infrastructure or cross-package changes: run affected package tests and
  focused integration tests.
- Run the full suite only for broad cross-cutting changes, global dependency or
  test-configuration changes, release-critical changes that cannot be validated
  narrowly, or when the user explicitly requests it.

If a validation command is likely to take more than two minutes, state why it is
needed before starting it. Honor an explicit user request to skip or narrow tests,
and record skipped checks honestly in the PR description.

Use the `pre-commit-check` skill for the detailed selection and reporting rules.

### 5. Run Code Formatting

Before creating the PR, ensure code is formatted:

```bash
black .
```

If formatting changes were made, commit them first:

```bash
git add -A
git commit -m "Format code with black"
```

### 6. Create or Update Branch

For a single PR, create a feature branch if needed:

```bash
git checkout -b <branch-name>
```

Recommended branch naming:
- `fix/<description>` - for bug fixes
- `feat/<description>` - for new features
- `enhance/<description>` - for enhancements
- `docs/<description>` - for documentation changes

For a large change that can be split into small, dependency-ordered layers,
follow the [Stacked Pull Request Workflow](#stacked-pull-request-workflow)
instead of steps 6 through 9. Keep unrelated or independent changes in separate
PRs rather than forcing them into a stack.

### 7. Commit Changes

Commit with a clear message following conventional commits format:

```bash
git commit -m "type(scope): brief description

Detailed description of the change."
```

### 8. Push to Remote

```bash
git push -u origin <branch-name>
```

### 9. Create the PR

Use the gh CLI with the proper PR template:

```bash
gh pr create --title "<PR Title>" --body "<PR Body>"
```

### Stacked Pull Request Workflow

Use a stack when later changes depend on earlier changes but each layer can be
reviewed as a focused PR. Put shared types, schemas, and other foundations in
lower layers; put their dependents in higher layers. Keep every branch and PR in
the same repository because GitHub does not support cross-fork stacks. If the
contribution must come from a fork, use a single PR or separate non-stacked PRs.

> [!NOTE]
> GitHub stacked pull requests are in public preview and may change. They are
> available on GitHub, GitHub Mobile, and through the `gh stack` extension,
> but not in GitHub Desktop.

#### 1. Install and Authenticate the CLI

Use GitHub CLI 2.90.0 or later and Git 2.20 or later. Authenticate, then install
the official extension if it is not already installed:

```bash
gh auth login
gh extension install github/gh-stack
```

#### 2. Design the Stack

Define the trunk and order the layers from foundational to dependent. The
bottom PR targets the trunk, usually `main`, and each PR above it targets the
branch immediately below it:

```text
feat/ui          -> PR #3 (base: feat/api)   <- top
feat/api         -> PR #2 (base: feat/core)
feat/core        -> PR #1 (base: main)       <- bottom
main             <- trunk
```

Ensure each PR contains only the diff for its layer. If code in one layer
depends on another, place the dependency in the same layer or a lower layer.

#### 3. Create and Commit the Layers

Initialize the bottom branch, commit its focused change, then add branches from
the bottom upward:

```bash
gh stack init --base main feat/core
# Make, stage, validate, and commit the foundational change.

gh stack add feat/api
# Make, stage, validate, and commit the dependent API change.

gh stack add feat/ui
# Make, stage, validate, and commit the dependent UI change.

gh stack view
```

Run the proportional validation selected in step 4 for every affected layer.
Keep the commits and working tree for each layer complete before adding the next
branch.

#### 4. Publish or Update the Stack

Use the interactive submit flow so each PR can receive an EmbodiChain title and
body. This command pushes the branches, creates or updates their PRs, and links
them as a stack:

```bash
gh stack submit
```

Use `gh stack push` only when the branches should be pushed without creating or
updating PRs. Use `gh stack submit --auto` only when generated titles and draft
PRs are acceptable; add `--open` to make them ready for review.

After the remote stack or trunk changes, synchronize the entire chain:

```bash
gh stack sync
```

After amending a lower layer, cascade the new commit history through its
dependents, then update the remote PRs:

```bash
gh stack rebase --no-trunk
gh stack submit
```

If a rebase stops on conflicts, resolve and stage them, then run
`gh stack rebase --continue`; use `gh stack rebase --abort` to restore the
pre-rebase state. Preserve the dependency order and rerun validation for every
affected layer before submitting again.

#### 5. Describe Every Layer

Apply the standard PR template to every PR and add this stack context near the
top of each description:

```markdown
## Stack

- Layer: <position>/<total>
- Base: `<base-branch>`
- Depends on: <lower PR URL or "none">
- Followed by: <upper PR URL or "none">
```

Explain what can be reviewed in that layer and report its validation separately.
Use `Fixes #<issue>` only on the layer whose independent merge resolves the
issue; use `Refs #<issue>` on prerequisite layers. Apply appropriate repository
labels to every PR in the stack.

#### 6. Review and Merge

Review every layer independently. GitHub applies the trunk's merge requirements,
branch protections, CODEOWNER approvals, and applicable PR-triggered CI checks
to every layer, including PRs that do not directly target the trunk.

Merge from the bottom upward:

- Merge the top PR to land the entire stack.
- Merge a middle PR to land it and every lower layer; higher PRs remain open and
  GitHub automatically rebases and retargets them.
- Merge the bottom PR to land only the first layer.

Use `gh stack merge` to select and merge the desired portion. The operation is
all-or-nothing for the selected layers and cannot bypass merge requirements:

```bash
gh stack merge
```

### 10. Select and Apply Labels

After creating the PR, select proper labels from the repository label list and apply them.

First, list available labels:

```bash
gh label list
```

Then choose labels based on change type and scope. Typical mapping:

- Bug fix: `bug`
- Enhancement: `enhancement`
- New feature: `feature`
- Documentation update: `docs`
- Affected area labels when available (for example): `physics`, `robot`, `agent`, `dataset`, `dexsim`

Apply labels to the PR:

```bash
gh pr edit <pr-number> --add-label "bug" --add-label "env"
```

If needed, remove incorrect labels:

```bash
gh pr edit <pr-number> --remove-label "<label-name>"
```

## PR Template

Use this template for the PR body:

```markdown
## Description

<!-- Clear summary of the change -->

This PR [briefly describe what the PR does].

<!-- Include motivation and context if needed -->
[Add any relevant motivation and context here].

<!-- List dependencies if applicable -->
Dependencies: [list any dependencies required]

<!-- Reference the issue -->
Fixes #<issue-number>

## Type of change

<!-- Select one and delete the others -->
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] Enhancement (non-breaking change which improves an existing functionality)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (existing functionality will not work without user modification)
- [ ] Documentation update

## Screenshots

<!-- Attach before/after screenshots if applicable -->

## Checklist

- [x] I have run the `black .` command to format the code base.
- [ ] I have made corresponding changes to the documentation
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] Dependencies have been updated, if applicable.
```

## PR Title Guidelines

- Keep titles short (under 70 characters)
- Use present tense and imperative mood
- Examples:
  - "Fix KeyError when 'add' mode not present in observation_manager"
  - "Add support for XYZ sensor"
  - "Improve contact sensor data buffer"

## Quick Reference

| Command | Purpose |
|---------|---------|
| `git status` | Check current state |
| `git diff HEAD` | Show changes |
| `black .` | Format code |
| `git checkout -b branch-name` | Create branch |
| `git push -u origin branch` | Push to remote |
| `gh pr create` | Create PR |
| `gh stack init --base main branch` | Initialize a stack and its bottom branch |
| `gh stack add branch` | Add a branch above the current top layer |
| `gh stack view` | Show stack order and PR state |
| `gh stack submit` | Push branches and create or update stacked PRs |
| `gh stack sync` | Fetch, cascade rebase, push, and synchronize the stack |
| `gh stack rebase --no-trunk` | Cascade a lower-layer update through the stack |
| `gh stack merge` | Merge a selected portion or the entire stack |
| `gh label list` | List repository labels |
| `gh pr edit <pr-number> --add-label ...` | Apply labels to PR |

## Notes

- Keep PRs small and focused. Large PRs are harder to review and merge.
- It's recommended to open an issue and discuss the design before opening a large PR.
- The checklist in the PR template should be completed honestly.
- For current stacked PR behavior, consult the [public preview announcement](https://github.blog/changelog/2026-07-30-stacked-pull-requests-are-now-in-public-preview/),
  [GitHub stacked PR overview](https://docs.github.com/en/pull-requests/get-started/about-stacked-prs),
  and [CLI command reference](https://docs.github.com/en/pull-requests/reference/stacked-prs-cli-commands).
