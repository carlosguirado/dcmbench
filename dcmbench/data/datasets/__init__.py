"""
Built-in datasets, one module each.

Each module exposes ``ENTRY`` (a ``DatasetEntry``). To add a dataset, add a
module following the same pattern and list it in ``dcmbench.data.registry``.
"""

# Files in github.com/carlosguirado/dcmbench-data, pinned to one commit.
DATA_REPO_COMMIT = "98c93d2b15f94ff2f34d8cdf3cb4ec001332426d"
DATA_REPO_URL = (
    "https://raw.githubusercontent.com/carlosguirado/dcmbench-data/"
    f"{DATA_REPO_COMMIT}/datasets/mode_choice"
)
