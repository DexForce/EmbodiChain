---
name: pre-commit-check
description: Use before committing or creating a PR for EmbodiChain to verify code style, headers, annotations, exports, and docstrings pass CI checks
---

# Pre-Commit Check

Run all local checks that the CI pipeline enforces, catching issues before pushing.

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

### 2. Run Black Formatting Check

This is the **first CI gate** and will cause immediate failure:

```bash
black --check --diff --color ./
```

If it fails, run `black .` and review the formatting changes.

### 3. Check Apache 2.0 Copyright Header

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

### 4. Check `from __future__ import annotations`

Every `.py` file must have this import (after the header, before other imports). This enables `A | B` syntax and forward references.

### 5. Check `__all__` in Public Modules

For any new or modified module under `embodichain/`, verify it defines `__all__` listing all public symbols. Example:

```python
__all__ = ["MyClass", "my_function"]
```

Skip this check for `__init__.py` files that only re-export via `from . import *`.

### 6. Check Docstrings on Public APIs

For any new public function, class, or method:
- Must have a Google-style docstring
- Must include `Args:` section if it takes parameters
- Must include `Returns:` section if it returns a value
- Use `.. attention::` or `.. tip::` directives for non-obvious behavior

### 7. Check Type Annotations

For any new public API:
- All parameters must have type hints
- Return type must be annotated
- Use `A | B` over `Union[A, B]`
- Use `TYPE_CHECKING` guard for imports that would cause circular dependencies

### 8. Check `@configclass` Usage

For any new configuration class:
- Must use `@configclass` decorator (not bare `@dataclass`)
- Must use `from dataclasses import MISSING` for required fields
- Import from `embodichain.utils import configclass`

### 9. Check Test Coverage

For any new public module or function:
- A corresponding test must exist at `tests/<subpackage>/test_<module>.py`
- Test file must also have the Apache 2.0 header
- Report if tests are missing

### 10. Summary Report

Output a pass/fail summary:

```
Pre-Commit Check Results
========================
[PASS] Black formatting
[PASS] Apache 2.0 headers (5/5 files)
[FAIL] from __future__ import annotations — missing in: foo.py
[PASS] __all__ exports
[PASS] Docstrings on public APIs
[PASS] Type annotations
[PASS] @configclass usage
[WARN] Missing tests for: bar.py

Fix the above issues before committing.
```

## What CI Checks

The project's CI pipeline (`.github/workflows/main.yml`) runs:

1. **lint** job: `black --check --diff --color ./`
2. **test** job: `pytest tests`
3. **build** job: Sphinx docs build

This skill covers items 1 and 2 locally. Docs build is heavier and typically only needed for documentation changes.

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
| Header check | Verify first line is `# ---...---` |
| `__future__` import | Grep for `from __future__ import annotations` |
| `__all__` export | Grep for `__all__` in module |
| Run tests | `pytest tests/<path>` |
