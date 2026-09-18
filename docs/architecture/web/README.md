# Architecture Explorer preview

A static, read-only frontend for inspecting EmbodiChain's architecture. This
iteration consumes generated, source-validated data and supports Sphinx embedding.

## Run locally

Requirements: Node.js 22.12+ and npm. From this directory:

```bash
npm ci
npm run dev
```

For the production bundle:

```bash
npm run build
npm run preview -- --port 4183 --strictPort
```

Open `http://127.0.0.1:4183/`. The preview binds to loopback only. The generated
`dist/` directory is standalone and uses relative asset paths; serve it over HTTP.
Fonts are bundled locally. No simulation, GPU, API key, or backend is required.

## Presentation

The interface and descriptive snapshot text are in English. The default academic
theme uses a paper-white canvas, serif headings, restrained category colours,
numbered layers, and static directional edges. An optional dark theme remains
available. Card titles use 16px Inter, summaries 13px, and extended descriptions
14px with generous line spacing. Each card has an explicit **Explore module**
button usable with a pointer or keyboard. Selecting a different module resets the
inspector to its introduction and focuses its heading.

All 101 selected modules provide an introduction and documentation links. The
inspector puts guide/API cards before relationship evidence. Sphinx links open
the corresponding HTML in the current documentation version; standalone links
open the pinned Markdown or reStructuredText source using its resolved extension.

The typography and reading hierarchy were informed by the public
[Sphinx Book Theme typography examples](https://sphinx-book-theme.readthedocs.io/en/stable/reference/kitchen-sink/typography.html)
and [uv documentation](https://docs.astral.sh/uv/). Interactive card affordances
follow the capabilities demonstrated by
[React Flow custom nodes](https://reactflow.dev/examples/nodes/custom-node).
The implementation and visual styling are specific to this explorer.

## Included

- System overview: 83 modules and 139 recorded relationships.
- Task Program: 23 modules and 67 recorded relationships.
- Simulation: 34 modules and 56 recorded relationships, including rigid bodies and groups,
  articulations and robots, surface/volume deformables, lights, constraints, gizmos,
  and shared object interfaces/backend views.
- Data & Learning: 29 modules and 36 recorded relationships.
- Generation & Toolkits: 21 modules and 27 recorded relationships.
- Overview sections separate shared computation, data pipeline/persistence, generative
  simulation, learning/policy optimization, and devices/asset tools.
- Data & Learning separates online sampling, recording, depth sidecars, algorithm/model
  selection, and standard/differentiable rollouts. Generation & Toolkits separates
  image-to-scene generation, edit/import/export, and general asset ingestion.
- Search by module name, responsibility, topic, or source path.
- Relationship filters, adjacent-node highlighting, zoom, pan, and a minimap.
- A readable 100% initial scale with responsive columns and scroll-to-pan navigation.
- Direct-neighbour mode hides unrelated nodes and keeps only incident relationships.
  It preserves the selected node even when every relationship is filtered out.
- Module responsibilities, boundaries, direction and scope of relationships,
  expandable exact source evidence, and pinned source/documentation-source links.
- URL fragments restore the current view, selection, local-focus mode, search, and relation filters;
  browser back/forward navigation works across selections.
- Light/dark themes and a responsive module list and inspector.

Click a module on the canvas or in the left directory. In its details, click a
related module to continue exploring. Double-click a canvas module to zoom in;
Fit overview shows the entire current diagram. Click the percentage control to
return to 100%. Scroll or drag to pan, or use the zoom buttons/pinch to scale.
Select a module and enable Direct neighbours only for a compact local diagram.
Selecting another module follows its neighbourhood; switching views clears local focus.
Closing the inspector preserves selection; use Details to reopen it. On narrow
screens, close details before using the focus toggle. Large neighbourhoods remain
at a readable scale with the selected module at the top; scroll to see more. The left directory also provides
keyboard access. Open the relationship menu to toggle types or clear all filters.

## Data and limits

The frontend fetches the `architecture.json` served beside its HTML entry and
validates it against `../architecture.schema.json`. Vite serves and bundles
`../generated/architecture.json` for standalone development; Sphinx builds supply
the freshly generated JSON through `ARCHITECTURE_DATA_PATH`. Loading and invalid
data have explicit states. The current snapshot has 101 unique
nodes and 205 unique edges; views share nodes and relationships. Its `revision`
identifies the committed source being described, independently of the frontend
implementation commit.

To refresh it, install the Python requirements in `../requirements.txt`, then run
`npm run data` followed by `npm run build`. See the [data workflow](../README.md)
for evidence review and validation. Generation reads HEAD by default and rejects
relevant uncommitted source changes. It never imports simulator code.

Every registered MAP topic and top-level production package has at least one entry.
The data tests detect unrepresented topics/packages and nodes absent from every view.
This measures discoverability, not complete internal coverage. Specialist-only modules
also appear in the searchable Sphinx text reference. Trajectory augmentation and
URDF assembly retain explicit unmapped-relationship states where host integration
is outside the selected evidence; no dependency is invented to connect a card.

The original `../task-program.sample.json` remains unchanged. Curated summaries
and semantic relationships live in `../curated.json`; selected AST imports and
inheritance are added during generation. Static imports are labeled separately
from reviewed semantic relations; conditional imports carry their condition in
scope. Module-level imports do not prove that a specific class uses a symbol.

Cards show the number of relationships visible in the current diagram. A zero
with known evidence means filters or local focus hide those links. Nodes without
recorded relationships are labeled Not mapped. Their inspector explicitly states
that this is incomplete coverage, not proof of having no dependencies. The overview is not a complete dependency or runtime call graph.

Standalone documentation links open the documentation **source** at the pinned
commit. In Sphinx, `?docsRoot=../../&theme=light` explicitly supplies the current
version root and initial theme; documentation links open in the parent page.
The full-screen URL retains this context and its fragment restores exploration.
Both embedded and standalone views provide a user-controlled theme toggle.

## Verification

```bash
npm test
npx playwright install chromium
npm run test:browser
npm run test:docs
npm run build
npm run format:check
```

The data tests read the pinned commit through Git and therefore require the full
repository checkout with that commit available. They check the JSON Schema,
graph/group/topic/doc references, and exact source excerpts. Navigation tests
exercise real browser interactions rather than mocked React components.

`npm run test:browser` starts its own dev server on port 4179. The production
preview on 4183 can remain running independently for visual review. Build outputs,
dependencies, and browser test reports are ignored by Git.

`npm run test:docs` needs the Python documentation requirements. It builds a real
Sphinx HTML/text fixture with lightweight documentation targets, then serves the
built assets on port 4180 to test version prefixes and snapshot failure states.
This is independent of the simulator and the full API documentation import chain.
