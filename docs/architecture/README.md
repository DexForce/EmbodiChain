# Architecture Explorer

A static [frontend](web/README.md) provides a 28-module overview and a 16-object
Task Program view, with search, relationship filters, direct-neighbour exploration,
source evidence, and shareable navigation. It consumes a
[generated snapshot](generated/architecture.json); the Sphinx documentation embeds the viewer and a searchable text reference.
Local and CI Sphinx builds regenerate their own version-specific assets.

- [Design and scope](../superpowers/specs/2026-09-17-architecture-explorer-design.md)
- [Implementation plan](../superpowers/plans/2026-09-17-architecture-explorer.md)
- [JSON Schema](architecture.schema.json): closed v1 contract.
- [Curated seed](curated.json): responsibilities, boundaries, groups, and reviewed relations.
- [Generated text overview](generated/summary.md): readable fallback with pinned evidence links.
- [Original Task Program sample](task-program.sample.json): unchanged historical contract example.
- [Earlier frontend snapshot](preview.snapshot.json): retained as a reference, no longer consumed by the app.

`agent_context/MAP.yaml` remains the sole topic inventory. Groups are presentation
choices. Neither import declarations nor adjacency establish runtime order. Every
relationship includes source evidence and scope; `holds` does not imply exclusive
ownership. This is selected static coverage, not a complete dependency graph.

## Generate and validate

From the repository root, with Python 3.11+ and Git available:

```bash
python -m pip install -r docs/architecture/requirements.txt
python docs/scripts/build_architecture.py --output-dir docs/architecture/generated
python docs/scripts/architecture_data.py docs/architecture/generated/architecture.json
python docs/scripts/architecture_data.py docs/architecture/task-program.sample.json
```

These dependencies are documentation tools only. Generation never imports or
executes EmbodiChain modules and needs no simulator or GPU. The default source is
`HEAD`; `--revision <commit-or-ref>` explicitly selects another locally available
commit. Missing revisions fail with a fetch instruction. Relevant uncommitted
source, MAP, VERSION, or documentation changes are rejected when generating HEAD.
Historical generation and validation read their pinned Git objects independently
of the working tree.

Generation updates the revision and package version, uniquely relocates each
curated excerpt within its declared lexical symbol, extracts selected AST facts,
and validates the result before replacing outputs. Missing/ambiguous evidence
fails with the owning node or edge. Validation failures preserve existing outputs.
Outputs have deterministic ordering and contain no timestamps or absolute paths.

The validator checks schema, IDs, edge endpoints, view/group membership, topic IDs,
exact evidence text and lexical containment, VERSION, and unambiguous docnames at
the pinned revision. Python module evidence uses `<module>`; non-Python text uses
`<document>`. Prose interpretation still requires human review.

## Maintain the data

1. Edit `curated.json` for responsibilities, boundaries, selected nodes, and semantic
   relations. Reuse node IDs across views and keep scope explicit.
2. After source changes, review affected evidence. Pure line moves are resolved
   automatically; rename or behavior changes require updating selectors/excerpts.
3. Run the generator and validator above. Review both generated views and the text
   summary, especially added relationships and unchanged semantic claims.
4. Run `npm run build` in `docs/architecture/web`, then inspect the frontend and
   expanded evidence. Commit seed, generated output, and associated source changes.

A snapshot's revision identifies the source it describes, not the later commit
that stores the generated artifact. `source_ref` is descriptive; all source links
use the immutable revision. Standalone preview documentation links open the documentation **source at that
revision**. Sphinx supplies a same-origin version root so embedded and full-screen
documentation links open the corresponding HTML page in the parent context.

Static extraction handles absolute imports in selected modules and unambiguous
class bases with unique static bindings. Conditional/type-only imports retain
conditions in their scope. A file's unique selected module/package (or sole
selected node) represents its module-level imports; this is explicitly not proof
that a specific class uses each import. Ambiguous file ownership, relative and
wildcard imports, dynamic imports, re-exports without an exact selected match,
nested-function imports, rebound aliases, and protocol-to-concrete bindings are
not resolved. Semantic relations remain reviewed seed data. Extraction adds a
relationship to each view containing both endpoints; view/group ordering is preserved.

## Verification

```bash
python -m pytest -q -c /dev/null -p no:cacheprovider --noconftest \
  tests/docs/test_architecture_data.py tests/docs/test_build_architecture.py
```

Frontend checks are documented in its [README](web/README.md). The original sample
remains pinned to `3224ac1ee28b6730b245d5cb69dc25a8d2d8dd94`; it is not copied as
current release data. The Sphinx extension supplies a version-relative documentation root, searchable
text entry, embedded viewer, and full-screen link.

## Documentation integration

Install `docs/requirements.txt`, Node.js 22.12+, and the frontend lockfile with
`npm --prefix docs/architecture/web ci`. Run `make -C docs current-docs` in the
project's full documentation environment, or use `sphinx-build` directly. Both
routes trigger the same `architecture_sphinx` extension. To build only the static
assets, run `python docs/scripts/build_architecture_assets.py`.

The asset builder stages JSON, summary, and the compiled app together. Failed
generation or frontend builds leave the previous complete bundle untouched and
abort Sphinx. Generated resources live in the ignored
`docs/source/_static/architecture/` directory. The summary is read into the entry
page during source processing, so Sphinx indexes its text. Non-HTML builders
produce the summary without requiring Node. Only the architecture page embeds
application scripts, via its titled iframe.

The full documentation setup is described in
[Build Documentation](../source/quick_start/docs.md). For a simulator-independent
integration check, run `npm run test:docs` from `docs/architecture/web`. It builds
a small real Sphinx site with lightweight API targets, checks HTML and text
builders, then browser-tests `/`, `/main/`, `/v0.2.4/`, and `/EmbodiChain/main/`.

CI installs Node and npm dependencies after checking out the ref being documented.
Refs predating this feature skip those steps and retain their own Sphinx config.
The docs test lane checks Python generation, frontend state, browser interactions,
and the Sphinx fixture. No generated snapshot is shared between refs.
