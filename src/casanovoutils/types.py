"""
Shared type aliases used across the package.
"""

from typing import Any, Callable, Iterable

PyteomicsSpectrum = list[dict[str, Any]]
Commands = dict[str, Callable] | Callable | Iterable[Callable]
