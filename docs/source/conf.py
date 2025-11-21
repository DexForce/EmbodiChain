# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo_e.png"
