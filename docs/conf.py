"""Sphinx configuration for TorchGW documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "TorchGW"
copyright = "2026, Sijie Chen"
author = "Sijie Chen"
release = "0.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_extra_path = ["_static/robots.txt", "_static/google17f7d289fafbfd7f.html"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "chansigit",
    "github_repo": "torchgw",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
}
html_logo = "logo.svg"
html_baseurl = "https://chansigit.github.io/torchgw/"

# SEO: meta tags for search engines
html_meta = {
    "description": "TorchGW — Fast Sampled Gromov-Wasserstein optimal transport solver in pure PyTorch. GPU-accelerated with Triton fused Sinkhorn kernels, mixed precision, and smart early stopping. 3-175x faster than POT.",
    "keywords": "Gromov-Wasserstein, optimal transport, PyTorch, Sinkhorn, graph matching, manifold alignment, single-cell, Triton, GPU",
    "author": "Sijie Chen",
    "google-site-verification": "google17f7d289fafbfd7f",
}

# Generate sitemap.xml for search engine crawling
extensions.append("sphinx_sitemap")

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
