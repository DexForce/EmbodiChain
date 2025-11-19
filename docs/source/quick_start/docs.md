# Build Documentation

## 1. Install the documentation dependencies

```bash
pip install -r docs/requirements.txt
```

> If you have issue like `locale.Error: unsupported locale setting`, please enter `export LC_ALL=C.UTF-8; export LANG=C.UTF-8` before build the API.

## 2. Build the HTML site

```bash
cd docs
make html
```

Then you can preview the documentation in your browser at `docs/build/html/index.html`.
