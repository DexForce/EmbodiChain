---
name: pre-commit-check
description: Use before committing or creating a PR for EmbodiChain to select proportional validation and verify affected code style, tests, headers, annotations, exports, and docstrings
---

# Pre-Commit Check

Run proportional local checks for the files being changed, catching relevant
issues before pushing without defaulting to the full test suite.

## When to Use

- Before creating a commit or PR
- User says "check my changes", "pre-commit", "verify before commit", "ready to push"
- After making any code changes to `.py` files

## Steps

### 1. Identify Changed Files

```bash
git diff --name-only HEAD
git diff --name-only --cached
git status --short
```

Collect all changed/added `.py` files.

Classify the change by affected area: workflow, docs, packaging, isolated Python
module, package-wide behavior, or cross-cutting infrastructure.

### 2. Run Black Formatting Check

This is the **first CI gate** and will cause immediate failure:

```bash
black --check --diff --color ./
```

If it fails, run `black .` and review the formatting changes.

### 3. Check Public API Documentation Coverage

Run the same read-only gate used by CI:

```bash
python docs/scripts/check_api_docs.py
```

If it reports missing exports, use `$update-api-docs` to add useful Sphinx
entries and descriptions. Do not change `__all__` solely to make this check
pass.

### 4. Check Apache 2.0 Copyright Header

Every `.py` file must begin with the 15-line copyright block. For each changed/new `.py` file, verify the first line is:

```
# ----------------------------------------------------------------------------
```

The full header template:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
```

### 5. Check `from __future__ import annotations`

Every `.py` file must have this import (after the header, before other imports). This enables `A | B` syntax and forward references.

### 6. Check `__all__` in Public Modules

For any new or modified module under `embodichain/`, verify it defines `__all__` listing all public symbols. Example:

```python
__all__ = ["MyClass", "my_function"]
```

Skip this check for `__init__.py` files that only re-export via `from . import *`.

### 7. Check Docstrings on Public APIs

For any new public function, class, or method:
- Must have a Google-style docstring
- Must include `Args:` section if it takes parameters
- Must include `Returns:` section if it returns a value
- Use `.. attention::` or `.. tip::` directives for non-obvious behavior

### 8. Check Type Annotations

For any new public API:
- All parameters must have type hints
- Return type must be annotated
- Use `A | B` over `Union[A, B]`
- Use `TYPE_CHECKING` guard for imports that would cause circular dependencies

### 9. Check `@configclass` Usage

For any new configuration class:
- Must use `@configclass` decorator (not bare `@dataclass`)
- Must use `from dataclasses import MISSING` for required fields
- Import from `embodichain.utils import configclass`

### 10. Select and Run Relevant Tests

Do not treat the CI test job as a requirement to run `pytest tests` locally for
every change. Choose the smallest command set that exercises the affected
behavior:

| Change scope | Default validation |
|---|---|
| `.github/workflows/**` only | `actionlint` on changed workflows; run related script tests only when workflow scripts changed |
| Docs content only | Relevant Sphinx build or docs-specific tests |
| One Python module | Matching `tests/**/test_<module>.py` |
| One package/subsystem | Tests for that package plus focused integration tests |
| Packaging/release code | Package build and artifact validation |
| Shared core, global test config, or multiple subsystems | Broader affected tests; full suite only when narrow coverage is not credible |

Skip runtime tests when no executable behavior is affected, but still run the
appropriate syntax or configuration validator. Run the full suite only when:

- the change affects shared core behavior used throughout the repository;
- global dependencies, test configuration, or environment initialization changed;
- several subsystems are modified together;
- a release-critical behavior cannot be validated narrowly; or
- the user explicitly requests it.

Before starting a command likely to take more than two minutes, report the
selected scope and why narrower validation is insufficient. Honor explicit user
instructions to skip or narrow tests.

### 11. Check Test Coverage

For any new public module or function:
- A corresponding test must exist at `tests/<subpackage>/test_<module>.py`
- Test file must also have the Apache 2.0 header
- Report if tests are missing

### 12. Summary Report

Output a pass/fail summary:

```
Pre-Commit Check Results
========================
[PASS] Black formatting
[PASS] Public API docs coverage
[PASS] Apache 2.0 headers (5/5 files)
[FAIL] from __future__ import annotations — missing in: foo.py
[PASS] __all__ exports
[PASS] Docstrings on public APIs
[PASS] Type annotations
[PASS] @configclass usage
[PASS] Targeted tests — tests/foo/test_bar.py
[N/A] Full test suite — isolated change covered by targeted tests
[WARN] Missing tests for: bar.py

Fix the above issues before committing.
```

## What CI Checks

The project's CI pipeline (`.github/workflows/main.yml`) runs:

1. **lint** job: `black --check --diff --color ./`
2. **lint** job: `python docs/scripts/check_api_docs.py`
3. **test** job: proportional pytest groups after lint passes
4. **build** job: Sphinx docs build after lint passes

This skill always covers the relevant lint and structural checks, then selects
tests proportionally. It does not require reproducing the entire CI pipeline for
every local change.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running `black` on only one file | Run `black .` on the whole project — CI checks everything |
| Forgetting test Apache header | Test files also need the 15-line copyright block |
| Using `Union[A, B]` | Use `A \| B` (with `from __future__ import annotations`) |
| Using bare `@dataclass` | Use `@configclass` from `embodichain.utils` |
| Missing `__all__` in new module | Add `__all__` with all public symbols |

## Quick Reference

| Check | Command/Method |
|-------|---------------|
| Black formatting | `black --check --diff --color ./` |
| Auto-fix formatting | `black .` |
| Public API docs | `python docs/scripts/check_api_docs.py` |
| Header check | Verify first line is `# ---...---` |
| `__future__` import | Grep for `from __future__ import annotations` |
| `__all__` export | Grep for `__all__` in module |
| Run targeted tests | `pytest tests/<affected-path>` |
| Run full tests | `pytest tests` only when the full-suite criteria above apply |
