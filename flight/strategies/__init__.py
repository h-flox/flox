"""
This module contains the base `Strategy` class and its default implementation.

In addition, this module has other strategy implementations provided in the `contrib`
submodule.
"""

from .strategy import DefaultStrategy, Strategy

__all__ = ["Strategy", "DefaultStrategy"]
