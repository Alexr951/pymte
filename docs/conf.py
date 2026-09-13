"""Sphinx configuration."""

import pymte

project = "pymte"
author = "Alex Ronczewski"
copyright = "2026, Alex Ronczewski"  # noqa: A001
release = pymte.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "jupyter_execute"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "pymte documentation"
html_favicon = "_static/favicon.svg"
# Only the user guide has enough sibling pages to need the section
# navigation in the left sidebar.
html_sidebars = {
    "guide/*": ["sidebar-nav-bs"],
    "generated/*": [],
    **{
        page: []
        for page in (
            "index", "installation", "quickstart", "theory", "api", "migration", "datasets",
            "faq", "citing",
        )
    },
}  # fmt: skip
html_theme_options = {
    "github_url": "https://github.com/alexr951/pymte",
    "navigation_with_keys": False,
}

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_output_stderr = "remove"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
