"""Prismatic VLA package init.

Lazy-load RLDS dataset utilities to avoid optional deps (e.g., dlimp) unless requested.
"""

from typing import Any

__all__ = ["get_vla_dataset_and_collator"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .materialize import get_vla_dataset_and_collator
        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
