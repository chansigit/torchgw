"""Sphinx configuration for SGW documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SGW"
copyright = "2026, Sijie Chen"
author = "Sijie Chen"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "chansigit",
    "github_repo": "sgw",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
}
html_logo = "logo.svg"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
