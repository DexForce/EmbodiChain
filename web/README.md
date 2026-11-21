# Shared graph frontends

The architecture explorer and Task Engine viewer use the same pinned React,
TypeScript, Vite and React Flow versions. `graph-ui` owns the shared viewport
controls and light/dark palette. Each consumer owns its graph model, layout and
business details. Python remains the Task Engine declaration/evidence validator.

## Build the offline Task Engine viewer

```bash
npm --prefix web/task-engine ci
npm --prefix web/task-engine run test
npm --prefix web/task-engine run build
```

The build compiles one IIFE and one stylesheet, collects dependency notices and
copies the three `_graph_react.*` files into the Python package. These generated
files are committed and shipped in the wheel. Normal Python rendering needs no
Node process, development server or network connection.

`task-engine/src/model.ts` projects recorded statuses without changing verdicts.
`svg.ts` exports vector geometry directly from that same layout; it does not
rasterize the diagram. Original report/data export stays byte-for-byte faithful.
Large raw JSON sections are rendered only after the reader expands them.

## Check browser behavior

```bash
npm --prefix web/task-engine exec -- playwright install chromium firefox webkit
PYTHON=python npm --prefix web/task-engine run test:browser
```

This builds the production offline viewer and generates synthetic display pages
through the public Python renderer. Chromium, Firefox and WebKit check recorded
states, noncontiguous environment IDs, missing evidence, assets, keyboard focus,
translations, viewport controls, responsive navigation and vector/JSON exports
from file URLs with external networking blocked. Local video playback is also checked
when `ffmpeg` is available. These fixtures do not represent physical runs.

Set `PYTHON` to the project Python interpreter. Optionally set
`GRAPH_VIEWER_REFERENCE_HTML` to an absolute path containing a frozen pre-migration
viewer; the same generated payloads are then compared with that frontend.
Screenshots, traces and the machine-readable report go under `test-results/`.
Chromium and Firefox also use offline browser contexts. The pinned WebKit build
rejects plain local files under Playwright's offline emulation, so its HTTP(S)
requests are intercepted and aborted instead; unexpected requests still fail
the checks.

## Architecture consumer

```bash
npm --prefix docs/architecture/web ci
npm --prefix docs/architecture/web run build
npm --prefix docs/architecture/web test
```

The local shared package uses peer dependencies. Both consumers preserve linked
package paths and deduplicate React/React Flow, ensuring one provider/runtime per
page. The architecture explorer retains its own navigation and documentation
integration; the Task Engine view retains its standalone artifact contract.
