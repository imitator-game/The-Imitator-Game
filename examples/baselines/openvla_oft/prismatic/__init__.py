"""Prismatic package init.

Lazy-load heavy model utilities to avoid optional deps (e.g., dlimp) at import time.
"""

from typing import Any

__all__ = [
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
