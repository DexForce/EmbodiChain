---
name: update-api-docs
description: Generate or update EmbodiChain Sphinx API documentation for public Python exports. Use when the API docs checker or CI reports missing __all__ exports, after adding or changing public APIs, or when asked to fill, generate, or synchronize API-reference pages and their descriptions.
---

# Update API Docs

Generate useful API documentation for every public export reported by the
read-only checker. Treat static ``__all__`` declarations as the public API
contract and preserve existing hand-written documentation.

## Workflow

1. Run the checker in machine-readable mode:

   ```bash
   python docs/scripts/check_api_docs.py --format json
   ```

   Exit status 1 is expected when exports are missing. Read the JSON report; if
   ``missing_count`` is zero, report that the docs are aligned and stop.

2. Group missing entries by module. Read each reported source file and inspect
   the exported definition, signature, type annotations, and docstring. Locate
   existing API pages with:

   ```bash
   rg -n "automodule:: <module>|currentmodule:: <module>" docs/source/api_reference
   ```

3. Choose the documentation location:

   - Add the export to an existing curated module page when one exists. Follow
     that page's headings, autosummary groups, and detailed autodoc directives.
   - Document a package-level re-export under its public import path, not only
     under the implementation module.
   - If no suitable curated page exists, add or extend the module section in
     ``docs/source/api_reference/public_api.rst``. This file is an
     agent-maintained fallback, not checker output. Keep fallback module
     headings sorted by import path and entries in their declared ``__all__``
     order to minimize diff churn.

4. Write documentation that explains the API:

   - Add the export to the appropriate ``autosummary`` block.
   - Add the matching ``autoclass``, ``autofunction``, ``autodata``, or other
     detailed directive when the surrounding page provides detailed entries.
   - Add a concise section overview when names alone do not explain the group.
   - If the source docstring is missing or too vague for autodoc, improve it
     with a meaningful summary and Google-style ``Args``, ``Returns``, and
     ``Raises`` sections where applicable.
   - Derive descriptions from the implementation and tests. Do not invent
     behavior, examples, guarantees, or parameter semantics.

5. Keep the change scoped to documentation. Do not alter runtime behavior,
   signatures, or ``__all__`` merely to silence the checker. Do not replace
   curated prose with generic generated text or add placeholders such as
   "part of the public API."

6. Rerun the checker until it reports zero missing exports:

   ```bash
   python docs/scripts/check_api_docs.py
   ```

7. Run Black on every changed Python file, then validate the documentation
   workflow:

   ```bash
   pytest tests/docs/test_check_api_docs.py -q --confcutdir=tests/docs
   python -m sphinx -b dummy docs/source docs/build/api-docs-check
   ```

   When source docstrings were changed, also run focused tests for those
   modules. Fix new Sphinx warnings caused by the edit; distinguish them from
   unrelated pre-existing warnings.

## Completion Report

Report the documented import paths, the pages or docstrings updated, and the
validation results. If an export cannot be documented accurately from the
repository, identify the exact missing semantic information instead of
guessing.
