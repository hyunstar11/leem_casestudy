import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Andrew Assessment"
copyright = f"{datetime.now().year}, anon"
author = "anon"
release = "0.1.0"

# The full version, including alpha/beta/rc tags
version = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames
source_suffix = [".rst", ".md", ".ipynb"]

# The master toctree document
master_doc = "index"

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Extension settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Sphinx Gallery settings
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",
    "capture_repr": ("_repr_html_", "__repr__"),
    "reset_modules": (),
    "reset_modules_order": "both",
    "ignore_pattern": r"__init__\.py",
    "min_reported_time": 0.007,
    "binder": {
        "org": "your-org",
        "repo": "your-repo",
        "branch": "main",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "../pyproject.toml",
    },
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}
