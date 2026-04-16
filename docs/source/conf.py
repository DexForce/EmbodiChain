# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

os.environ.setdefault(
    "AZURE_OPENAI_ENDPOINT", "https://mock-endpoint.openai.azure.com/"
)
os.environ.setdefault("AZURE_OPENAI_API_KEY", "mock-api-key-for-docs-build")

sys.path.insert(0, os.path.abspath("../.."))


project = "EmbodiChain"
copyright = "2025, The EmbodiChain Project Developers"
author = "The EmbodiChain Project Developers"

# Read version from VERSION file if it exists
with open(os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",  # optional, shows type hints
    "sphinx_design",
    "myst_parser",  # if you prefer Markdown pages
    "sphinx_copybutton",
    "sphinx_multiversion",
]
# Napoleon settings if using Google/NumPy docstring style:
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_typehints = "signature"
autodoc_class_signature = "separated"
# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = False
# default autodoc settings
autodoc_default_options = {
    "autosummary": True,
}

# If using MyST and writing .md API stubs:
myst_enable_extensions = ["colon_fence", "deflist", "html_admonition"]


templates_path = ["_templates"]
exclude_patterns = []


# -- sphinx-multiversion configuration -------------------------------------------------
# Enable multi-version builds (set to True in v0.2.0+)
ENABLE_MULTI_VERSION = True

# Version buffer size: number of releases to keep + main branch
VERSION_BUFFER_SIZE = 5

# Only build tags that look like v1.0.0 or branches like main/dev
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^(main|dev)$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^tags/v\d+\.\d+\.\d+$"
smv_outputdir_format = "{ref.name}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# Don't include version-redirect.js automatically - we add it manually to root
html_js_files = []
# html_logo = "_static/logo_e.png"

# Configure HTML base URL for better local previewing
# Use empty string to use relative paths from the build directory
html_baseurl = ""

# HTML context for better path handling
html_context = {
    "github_user": "dexforce",
    "github_repo": "EmbodiChain",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Configure theme with version selector (enabled in Phase 2)
html_theme_options = {
    "title": "EmbodiChain",
    "logo_only": False,
    "show_toc_level": 2,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "prev_next_buttons_location": "bottom",
    # version_menu: True,  # Enable version selector in navbar (Phase 2)
}
