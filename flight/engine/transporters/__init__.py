"""
This module contains implementations of _Data **Transporters**_ which are used to handle
how to "transport" the data (e.g., locally, across nodes at a distributed cluster, or
across remote resources).
"""

from .base import AbstractTransporter, InMemoryTransporter

__all__ = ["AbstractTransporter", "InMemoryTransporter"]
