# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pirate_frb'
copyright = '2026, Kendrick Smith'
author = 'Kendrick Smith'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx.ext.autosummary',
    'sphinxarg.ext',   # sphinx-argparse
]

templates_path = ['_templates']
exclude_patterns = []

# Markdown only
source_suffix = {
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#007bff",
        "color-brand-content": "#0056b3",
    },
}

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
]

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': True,
}

# Note: Do NOT mock ksgpu or pirate_pybind11 here!
# Mocking ksgpu prevents its ctypes trick from running, which causes
# pirate_pybind11.so to fail loading (undefined symbols from ksgpu).
# For local builds, we import the real modules to get pybind11 docstrings.
# For ReadTheDocs (no compiled extensions), we'll need a different approach.
autodoc_mock_imports = []

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
