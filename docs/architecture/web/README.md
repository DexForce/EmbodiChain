# Architecture Explorer preview

A static, read-only frontend for inspecting EmbodiChain's architecture. This
iteration is intended for visual and interaction review before the documentation
generator and Sphinx integration are implemented.

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
available. Source excerpts and pinned source references are unchanged.

## Included

- Overview: 28 modules and 40 recorded relationships, organized into five layers.
- Task Program: 16 objects and 25 recorded relationships, organized into four layers.
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

The frontend imports `../preview.snapshot.json` at build time, using the shared
`../architecture.schema.json` contract. The snapshot has 34 unique nodes and 53
unique edges; views share nodes and relationships. Its source revision is
`3224ac1ee28b6730b245d5cb69dc25a8d2d8dd94`, not the frontend implementation commit.

The original `../task-program.sample.json` remains unchanged. The preview adds
curated module summaries and selected static import declarations to that source
baseline. Static imports are labeled separately from reviewed semantic relations;
conditional imports carry their condition in the evidence scope. The overview also
includes reviewed CLI dispatch, task discovery, configuration composition, environment
inheritance, and simulation ownership links.

Cards show the number of relationships visible in the current diagram. A zero
with known evidence means filters or local focus hide those links. Nodes without
recorded relationships are labeled Not mapped. Their inspector explicitly states
that this is incomplete coverage, not proof of having no dependencies. The overview is not a complete dependency or runtime call graph.

Documentation links currently open the documentation **source** at the pinned
commit. The app does not claim that a matching published documentation version
exists. An Sphinx-aware documentation URL resolver, automatic source refresh,
and CI publishing remain separate work in the implementation plan.

## Verification

```bash
npm test
npx playwright install chromium
npm run test:browser
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
