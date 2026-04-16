---
name: pr
description: Create a pull request for EmbodiChain following the project's PR template and conventions, including selecting proper GitHub repository labels
---

# EmbodiChain Pull Request Creator

This skill guides you through creating a pull request that follows the EmbodiChain project's PR template and contribution guidelines.

## Usage

Invoke this skill when:
- You have completed a feature, bug fix, or other change and want to create a PR
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

### 4. Run Code Formatting

Before creating the PR, ensure code is formatted:

```bash
black .
```

If formatting changes were made, commit them first:

```bash
git add -A
git commit -m "Format code with black"
```

### 5. Create or Update Branch

If not already on a feature branch:

```bash
git checkout -b <branch-name>
```

Recommended branch naming:
- `fix/<description>` - for bug fixes
- `feat/<description>` - for new features
- `enhance/<description>` - for enhancements
- `docs/<description>` - for documentation changes

### 6. Commit Changes

Commit with a clear message following conventional commits format:

```bash
git commit -m "type(scope): brief description

Detailed description of the change.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### 7. Push to Remote

```bash
git push -u origin <branch-name>
```

### 8. Create the PR

Use the gh CLI with the proper PR template:

```bash
gh pr create --title "<PR Title>" --body "<PR Body>"
```

### 9. Select and Apply Labels

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
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
| `gh label list` | List repository labels |
| `gh pr edit <pr-number> --add-label ...` | Apply labels to PR |

## Notes

- Keep PRs small and focused. Large PRs are harder to review and merge.
- It's recommended to open an issue and discuss the design before opening a large PR.
- The checklist in the PR template should be completed honestly.
- Remember to include the Claude Code attribution at the bottom of the PR body.
