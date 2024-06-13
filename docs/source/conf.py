# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Numba CUDA'
copyright = '2024, NVIDIA'
author = 'NVIDIA'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['numpydoc']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
    import nvidia_sphinx_theme  # noqa: F401
    html_theme = "nvidia_sphinx_theme"
except ImportError:
    html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_favicon = "_static/numba-green-icon-rgb.svg"
html_show_sphinx = False
