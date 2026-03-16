import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "casanovoutils"
copyright = "2024, The Noble Lab"
author = "The Noble Lab"

extensions = [
    "myst_parser",
    "autoapi.extension",
]

# sphinx-autoapi configuration
autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# MyST parser extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "strikethrough",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/Noble-Lab/casanovoutils",
    "use_repository_button": True,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
