"""
Shared type aliases used across the package.
"""

from typing import Any, Callable

PyteomicsSpectrum = list[dict[str, Any]]
CommandDict = dict[str, Callable]
