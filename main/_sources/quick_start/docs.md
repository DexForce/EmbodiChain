# Build Documentation

## 1. Install the documentation dependencies

```bash
pip install -r docs/requirements.txt
```

> If you have issue like `locale.Error: unsupported locale setting`, please enter `export LC_ALL=C.UTF-8; export LANG=C.UTF-8` before build the API.

## 2. Build the HTML site

### Local development (current version only)

```bash
cd docs
make current-docs
```

Then you can preview the documentation in your browser at `docs/build/html/index.html`.

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

> Old release versions beyond `DOCS_MAX_VERSIONS` (default: 4) are automatically pruned during CI builds.
