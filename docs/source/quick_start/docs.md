# Build Documentation

## 1. Install the documentation dependencies

Build the docs from a source checkout in a Python 3.11 virtual environment.
API generation imports optional GenSim modules, so this setup installs the
`gensim` extra and its ABI-specific `bpy` dependency. A Python 3.12 environment
can run the core package, but it cannot run this full documentation setup.

Install the project runtime and documentation toolchain from the repository
root:

```bash
pip install -e ".[gensim]" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site \
  --extra-index-url https://download.blender.org/pypi/
pip install -r docs/requirements.txt
```

HTML builds also require Node.js 22.12+ and the locked Architecture Explorer
frontend dependencies. Install these once from the repository root:

```bash
npm --prefix docs/architecture/web ci
```

Every Sphinx build generates architecture evidence from the checkout's HEAD.
HTML builds compile the static frontend; text/PDF builders include the searchable
module overview without running Node. Relevant source changes must be committed
before generating their pinned evidence. Generated bundles live under
`docs/source/_static/architecture/` and are ignored by Git. A failed generation
stops the build before replacing the last complete bundle.

The documentation requirements are pinned so local and CI builds use the same
Sphinx toolchain.

> If the build raises `locale.Error: unsupported locale setting`, run
> `export LC_ALL=C.UTF-8; export LANG=C.UTF-8` before rebuilding.

## 2. Build the HTML site

### Local development (current version only)

```bash
cd docs
make current-docs
```

This target treats warnings as errors. Preview the result at
`docs/build/html/index.html`. To view the interactive Architecture Explorer,
serve the output over HTTP (rather than opening it as a local file):

```bash
python -m http.server 4184 --bind 127.0.0.1 --directory build/html
```

Visit `http://127.0.0.1:4184/overview/architecture/index.html`. The page includes
an embedded viewer, a full-screen link, and a searchable text reference. API
links stay in the current documentation version; source links use its snapshot SHA.

### Multi-version docs (CI/production)

The production docs site hosts multiple versions side by side. Each version is built independently into its own subdirectory under `docs/build/html/`:

```
docs/build/html/
├── index.html           # Redirect → latest stable
├── versions.json        # Version manifest for the sidebar selector
├── main/                # Dev docs (latest main branch)
├── v0.1.3/              # Release docs
└── v0.1.2/              # Release docs
```

To build a specific version into this layout:

```bash
cd docs
sphinx-build source build/html/<version>
```

For example, to build the `main` branch docs:

```bash
sphinx-build source build/html/main
```

Then generate the version manifest and root redirect:

```bash
python3 scripts/generate_versions_json.py --build-dir build/html
```

This generates both `versions.json` (for the sidebar version selector) and `index.html` (redirects to the latest stable version, falling back to `main`).

> Old release versions beyond `DOCS_MAX_VERSIONS` (default: 5 in CI) are automatically pruned during CI builds.
>
> CI merges missing version directories from the live GitHub Pages site before each build so a `main` push cannot wipe docs built for release tags. See `docs/scripts/merge_published_site.py` and `tests/docs/test_merge_published_site.py`.
>
> Production deployment uses a dedicated GitHub Pages workflow that consumes the built multi-version site artifact. This keeps tag-based release docs publishing working even when the `github-pages` environment only allows deployments from the default branch workflow context.
