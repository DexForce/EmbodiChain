# Architecture Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline unless the user explicitly requests parallel agents.

**Goal:** Publish a versioned, source-backed architecture explorer with an overview and a Task Program view inside the existing Sphinx documentation.

**Architecture:** A documentation-only generator turns a curated source snapshot and selected AST facts into deterministic JSON. A static React Flow application renders it; Sphinx supplies the searchable text entry, iframe, and full-screen link. Source links use the snapshot commit and documentation links stay within the current documentation version.

**Tech Stack:** Python AST, JSON Schema Draft 2020-12, React, TypeScript, React Flow, Vite, Sphinx; pytest and browser interaction tests. Resolve frontend dependency versions during implementation and commit their lockfile.

**Spec:** [2026-09-17-architecture-explorer-design.md](../specs/2026-09-17-architecture-explorer-design.md)

## Global Constraints

- `schema_version` 固定为 `1`。
- 两个视图：全局架构、Task Program 集成。全局默认展示下面的 24 个节点。
  The referenced 24-node inventory is in the spec, not an additional topic map.
- `source_ref` 仅用于说明，源码链接必须绑定 `revision`。
- 样例保持当前历史提交，作为契约参考。正式发布快照每次从对应 checkout 重新生成，禁止直接把历史样例复制成最新版架构数据。
- 不 import `embodichain`，不要求 GPU、DexSim 或运行中的环境。
- 相同源码和输入产生相同输出，不写入时间戳或机器绝对路径。
- 现有 `lab/visualization` 继续只承载仿真可视化，本工具由 docs 工具链拥有。
- `agent_context/MAP.yaml` remains the sole topic inventory. Never infer dependency edges from `related_topics`.
- Follow `/add-test` for new Python tests and `/pre-commit-check` before commits. Run `black==26.3.1` with `black .` before every commit.

## Current Deliverable and Execution Order

The scope, schema, and pinned sample are ready for review. The application,
production generator, and Sphinx integration below are not implemented.
Tasks are sequential: each consumes the preceding deliverable. Do not add a
server, database, graph editor, runtime tracer, or automatic-layout dependency.

## Task 1: Validated Architecture Snapshots

**Files:**
- Use: `docs/architecture/architecture.schema.json`
- Preserve: `docs/architecture/task-program.sample.json`
- Create: `docs/scripts/architecture_data.py`
- Create: `docs/architecture/requirements.txt`
- Create: `tests/docs/test_architecture_data.py`

**Interfaces:**
- Consumes: the v1 JSON contract and a repository containing the pinned revision.
- Produces: `ArchitectureDataError(ValueError)`,
  `load_snapshot(path: Path) -> dict[str, Any]`, and
  `validate_snapshot(data: dict[str, Any], *, repo_root: Path, schema_path: Path) -> None`.
- Validation reads source and MAP using the exact `revision`, never by importing
  modules. `ArchitectureDataError` includes the node/edge ID or JSON field path.

- [ ] Read the spec, schema, README validation example, and existing
  `tests/docs/test_check_api_docs.py` module-loading pattern. Add the Apache header,
  future annotations, and typed public functions to the new script.
- [ ] Add failing tests using temporary Git repositories: create a tiny class,
  commit it, construct valid evidence, then mutate one property at a time.
  Reuse a `valid_snapshot` fixture that creates the temporary repository and schema.
  Include these concrete rejection cases:

```python
def test_rejects_dangling_target(valid_snapshot):
    data, root, schema_path = valid_snapshot
    data["edges"][0]["target"] = "missing"
    with pytest.raises(ArchitectureDataError, match="missing"):
        validate_snapshot(data, repo_root=root, schema_path=schema_path)


def test_rejects_stale_excerpt(valid_snapshot):
    data, root, schema_path = valid_snapshot
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Renamed:"
    with pytest.raises(ArchitectureDataError, match="excerpt"):
        validate_snapshot(data, repo_root=root, schema_path=schema_path)
```

- [ ] Cover duplicate IDs, unknown relation, parent-traversal paths, reversed
  line ranges, wrong lexical symbols, absent topics, missing/ambiguous docnames,
  view edges with excluded endpoints, and duplicated/omitted group members.
  Support `<module>` for Python module-level evidence and `<document>` for
  non-Python text evidence; test each without importing or executing source.
  Add a historical-source test that changes the working-tree file and still
  validates the original committed snapshot.
- [ ] Run `python -m pytest -q -c /dev/null -p no:cacheprovider --noconftest tests/docs/test_architecture_data.py`
  and confirm failures correspond to the missing validator.
- [ ] Implement structural validation with `Draft202012Validator`, then the
  reference/evidence checks in the README. Cache Git source reads and ASTs.
  Add `jsonschema` and `PyYAML` as documentation-tool dependencies; no package
  runtime dependency change. Missing Git revisions must fail with an actionable
  fetch message rather than fall back to HEAD.
- [ ] Run the focused tests and validate the pinned sample. Record the exact
  dependency versions used. Format, review, and commit this independently useful
  data-validation tool.

## Task 2: Current-Revision Generation and Both Views

**Files:**
- Create: `docs/architecture/curated.json`
- Create: `docs/scripts/build_architecture.py`
- Extend: `docs/scripts/architecture_data.py`
- Create: `tests/docs/test_build_architecture.py`

**Interfaces:**
- Consumes: Task 1 validator and a curated snapshot in the same v1 format.
- Produces: `build_snapshot(repo_root: Path, curated_path: Path) -> dict[str, Any]`
  and CLI `python docs/scripts/build_architecture.py --output-dir <directory>`.
- CLI writes `architecture.json` and `summary.md` to the explicit output directory.
  It exits nonzero before replacing outputs if evidence or references fail.
- `curated.json` is the editable semantic seed; evidence identifies the reviewed
  lexical symbol and exact excerpt. The generator resolves their positions at
  the checked-out commit and replaces revision/package metadata. The original
  `task-program.sample.json` remains unchanged.

- [ ] Add fixtures with a class/method moved by leading comments, a renamed
  symbol, and two matching excerpts inside the same symbol. Require unique
  resolution within the lexical symbol. Do not use global first-match search.
- [ ] Add these tests before implementing generation:

```python
def test_generation_is_deterministic(source_fixture):
    root, curated_path = source_fixture
    first = build_snapshot(root, curated_path)
    second = build_snapshot(root, curated_path)
    assert first == second
    assert "generated_at" not in first


def test_changed_evidence_is_not_silently_relocated(changed_source_fixture):
    root, curated_path = changed_source_fixture
    with pytest.raises(ArchitectureDataError, match="excerpt"):
        build_snapshot(root, curated_path)
```

- [ ] Run the new tests and confirm the intended failures. Implement generation
  in stable ID order, preserving explicit view/group order. Read tracked source
  from HEAD; reject uncommitted changes to relevant source, MAP, or doc targets
  so links cannot claim a commit that does not contain the displayed evidence.
  Semantic seed edits can be reviewed independently; this is a source snapshot,
  not a claim that prose already exists in the pinned commit.
- [ ] Expand `curated.json` to the spec's 24-node overview and the Task Program
  view. Reuse shared node IDs between views. Verify each new semantic relationship
  against source and retain its scope; do not manufacture edges to make a
  disconnected graph look complete.
- [ ] Extract only unambiguous class inheritance and module-level absolute
  imports between selected nodes. Label these `static-extracted`; describe
  conditional/type-only imports in scope. Skip unresolved aliases, dynamic
  imports, nested-function imports, and protocol-to-concrete bindings unless
  covered by reviewed evidence. Merge by source/target/relation/evidence identity.
- [ ] Test that generation does not import `embodichain`, resolves line moves,
  rejects ambiguous excerpts, validates its own output, and keeps topics tied
  to the pinned MAP. Test summary output includes every overview node with a
  same-version docname link and plain-text responsibility.
- [ ] Run both data test files, generate into a temporary directory twice and
  compare bytes. Format, review the two views' evidence, and commit.

## Task 3: Static Explorer with Testable Navigation

**Files:**
- Create under `docs/architecture/web/`: `package.json`, `package-lock.json`,
  `tsconfig.json`, `vite.config.ts`, `index.html`, `playwright.config.ts`
- Create under `docs/architecture/web/src/`: `main.tsx`, `App.tsx`,
  `graph.ts`, `state.ts`, `links.ts`, `ArchitectureNode.tsx`,
  `NodeDetails.tsx`, `styles.css`
- Create: `docs/architecture/web/tests/state.test.ts`
- Create: `docs/architecture/web/tests/explorer.spec.ts`

**Interfaces:**
- Consumes: `architecture.json`, served beside the application entry point.
- Produces: `npm run build` static bundle with relative asset URLs,
  `npm test` pure state tests, and `npm run test:browser` interaction tests.
- `ExplorerState` contains `viewId: string`, `nodeId: string | null`,
  `relations: string[]`, and `query: string`.
- `parseState(hash: string, data: ArchitectureSnapshot)` returns
  `{ state: ExplorerState; notices: string[] }`;
  `serializeState(state: ExplorerState): string` writes the documented fragment.
- `sourceUrl(revision: string, evidence: Evidence): string` uses the fixed
  repository URL and commit. `documentationUrl(docname: string, docsRoot: URL): string`
  resolves from the explicit current-version root supplied by Sphinx.

- [ ] Verify current library/runtime compatibility, fix dependency versions in
  the lockfile, and add scripts for dev, build, typecheck, state tests, and browser
  tests. Keep dependencies local to documentation tooling.
- [ ] Write failing state tests for fragment round-trip, unknown view/node,
  empty relation filters, search with no matches, and removal of a selected node.
  Use this source-link assertion:

```typescript
expect(sourceUrl(snapshot.revision, snapshot.nodes[0].evidence[0]))
  .toContain(`/blob/${snapshot.revision}/`);
```

- [ ] Implement the state and link functions. On an unknown view select the first
  view and return a notice; on an unknown node clear selection and return a notice.
  Filter invalid relation names. The details panel remains available if current
  filters hide all of the selected node's adjacent edges.
- [ ] Load and minimally validate the JSON envelope at startup, with visible
  loading/error states. Render custom cards, fixed group/column placement,
  relation filters, search, a keyboard node list, and the details panel. Disable
  connection editing; keep drag positions in session state only.
- [ ] Add browser checks against the built application, including:

```typescript
await page.getByRole('button', { name: 'Select AtomicActionEngine' }).click();
await expect(page.getByRole('complementary', { name: 'Node details' }))
  .toContainText('MotionGenerator');
await page.reload();
await expect(page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }))
  .toBeVisible();
```

- [ ] Check relationship selection, source evidence, keyboard selection,
  no-results/error states, view switching, and browser back/forward restoration.
  Confirm filters are applied within the selected view and all underlying
  parallel relationships remain accessible in details.
- [ ] Run typecheck, state tests, build, and browser tests. Inspect the actual
  page at desktop and narrow widths in light/dark themes; confirm readable
  cards and no inaccessible clipped detail controls. Commit the prototype.

## Task 4: Versioned Sphinx Entry and One Build Command

**Files:**
- Create: `docs/source/overview/architecture/index.md`
- Modify: `docs/source/index.rst`, `docs/source/conf.py`
- Create: `docs/scripts/build_architecture_assets.py`
- Modify: `docs/Makefile`, `docs/requirements.txt`, `.gitignore`
- Create: `tests/docs/test_architecture_assets.py`
- Extend: `docs/architecture/web/tests/explorer.spec.ts`

**Interfaces:**
- Consumes: Task 2 generation CLI and Task 3 frontend build.
- Produces: `python docs/scripts/build_architecture_assets.py` writes a complete
  static application and snapshot under `docs/source/_static/architecture/`,
  plus a generated summary include outside tracked content.
- The checked-in Sphinx entry includes the generated summary and uses a
  Sphinx-resolved relative URL for the iframe/full-screen link. Assets and summary
  are ignored. Builds fail if their generation fails; never publish stale assets.

- [ ] Write orchestration tests with stub subprocesses for generator/build
  success and failure, including proof that a failure does not replace the last
  complete output directory. Use staging plus atomic replacement for output.
- [ ] Implement the command: generate snapshot/summary into staging, run the
  frontend build, assemble assets, then replace outputs. Add Node toolchain setup
  to the documented local prerequisites and keep `npm ci` distinct from repeated
  build invocations.
- [ ] Add the Overview navigation entry, searchable text include, titled iframe,
  fallback text, and full-screen link. Supply version-relative `docsRoot` and
  explicit theme parameters. Open documentation links in the parent context.
  Keep script loading scoped to the architecture page.
- [ ] Wire the command into local `make html` and `make current-docs` preparation.
  Add documentation-tool dependencies to docs requirements. Non-HTML builders
  render the summary and links; they do not depend on an interactive iframe.
- [ ] Run a focused Sphinx fixture and browser tests serving the output at `/`,
  `/main/`, `/v0.2.4/`, and `/EmbodiChain/main/`. Assert JSON and assets load, API
  links stay in the same version, source links contain the snapshot SHA, and a
  shared fragment restores state in the full-screen page.
- [ ] Run `make -C docs current-docs` in the documented docs environment and
  inspect both entry and full-screen pages. Report pre-existing build failures
  separately from changes. Format, review, and commit the integration.

## Task 5: CI and Maintenance Gates

**Files:**
- Modify: `.github/workflows/main.yml`, `.github/workflows/docs-pages.yml`
- Extend: `tests/docs/test_architecture_assets.py`
- Modify: `docs/architecture/README.md`

**Interfaces:**
- Consumes: the same asset command used by local documentation builds.
- Produces: CI artifacts and per-version builds containing their own architecture
  assets, source revision, summary, and links.

- [ ] Inspect each existing Sphinx invocation and checkout boundary. Call the
  asset builder after checking out the ref to document, and before each HTML
  build. Historical refs predating the feature must retain their existing build
  behavior; check for the asset-builder file before invoking it on those refs.
  Leave `docs/scripts/build_versions.py` unchanged: it filters retained versions
  and currently has no build orchestration responsibility.
- [ ] Install the locked frontend dependencies and browser-test dependencies
  only in relevant documentation/test jobs. Cache using the lockfile, without
  sharing generated architecture JSON between refs.
- [ ] Add a two-revision fixture test with changed class locations and labels.
  Verify each version directory keeps its own JSON SHA, excerpts and doc links.
  Add a regression test for a historical ref with no architecture builder.
- [ ] Document the update procedure: edit the curated responsibilities/relations,
  refresh source selectors when behavior changes, build, inspect evidence,
  validate links, and review the rendered graph. Document that static extraction
  cannot establish a complete runtime call graph.
- [ ] Run the proportional Python/browser checks, `actionlint` on the changed
  workflows, `python docs/scripts/check_api_docs.py`, and the project-context
  affected check. Add/update context only when new behavior needs a routed topic;
  do not create a second architecture inventory in MAP.
- [ ] Run `black .`, review all generated/ignored files and staged changes, then
  commit. Record the full verification commands and any environment limitations.

## Completion Evidence

- Two usable views, including exactly the 24 scoped overview nodes.
- Every semantic edge has source evidence and an explicit scope.
- Stable node IDs, typed relations, searchable descriptions, and same-version links.
- Browser-tested embedded/full-screen interactions and fragment restoration.
- All local and CI documentation build routes generate from their own checkout.
- No modifications to simulation execution, runtime dependencies, or project topic ownership.
