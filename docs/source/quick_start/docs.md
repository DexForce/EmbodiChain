# Build Documentation

## 1. Install the documentation dependencies

Build the docs from a source checkout in a Python 3.10 or 3.11 virtual
environment. API generation imports EmbodiChain modules, so install the project
runtime and the documentation toolchain from the repository root:

```bash
pip install -e ".[gensim]" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site \
  --extra-index-url https://download.blender.org/pypi/
pip install -r docs/requirements.txt
```

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
`docs/build/html/index.html`.

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
