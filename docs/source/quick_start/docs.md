# Build Documentation

## 1. Install the documentation dependencies

Activate a Python 3.11 virtual environment to match the documentation build in
CI. Sphinx imports GenSim modules to generate the API reference, so install the
`gensim` extra and the pinned documentation toolchain from the repository root.

```bash
pip install -e ".[gensim]" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site \
  --extra-index-url https://download.blender.org/pypi/
pip install -r docs/requirements.txt
```

HTML builds also require Node.js 22.12+ and the Architecture Explorer frontend
dependencies.

```bash
npm --prefix docs/architecture/web ci
```

## 2. Build the HTML site

### Local development (current version only)

From the repository root, build into a dedicated preview directory and serve it
over HTTP so the Architecture Explorer can load its data.

```bash
make -C docs current-docs BUILDDIR=build/preview
python -m http.server 4184 --bind 127.0.0.1 --directory docs/build/preview/html
```

`current-docs` synchronizes the Introduction from the repository README and
treats Sphinx warnings as errors. Open `http://127.0.0.1:4184/` to preview the
homepage and browse to the Architecture Explorer.

Sphinx pins architecture evidence to the checkout's `HEAD`. If it reports
`Relevant source has uncommitted changes`, build a committed revision containing
the listed changes to preview that architecture snapshot. The generated bundle
lives under `docs/source/_static/architecture/`. Text and PDF builders use the
searchable module overview without building the frontend.

If Python reports `locale.Error: unsupported locale setting`, set both `LC_ALL`
and `LANG` to `C.UTF-8` before rebuilding.

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

Build from a checkout of the branch or tag you want to document. From the
repository root, synchronize the Introduction, build into the matching version
directory, and regenerate the version selector. For the `main` checkout, run

```bash
make -C docs sync-readme
sphinx-build docs/source docs/build/html/main
python docs/scripts/generate_versions_json.py --build-dir docs/build/html
```

For a release checkout, use its tag in place of `main` in the Sphinx output path.
The version script writes `versions.json` and a root `index.html` that sends
readers to the latest stable build or falls back to `main`.

> Old release versions beyond `DOCS_MAX_VERSIONS` (default: 5 in CI) are automatically pruned during CI builds.
>
> CI merges missing version directories from the live GitHub Pages site before each build so a `main` push cannot wipe docs built for release tags. See `docs/scripts/merge_published_site.py` and `tests/docs/test_merge_published_site.py`.
>
> Production deployment uses a dedicated GitHub Pages workflow that consumes the built multi-version site artifact. This keeps tag-based release docs publishing working even when the `github-pages` environment only allows deployments from the default branch workflow context.
