"""
This module contains classes, functions, and type defitions that are used to act as an
interface to PyTorch to simplify the components necessary to train neural networks in
Flight.
"""
from .module import TorchDataModule, TorchModule

__all__ = ["TorchDataModule", "TorchModule"]
