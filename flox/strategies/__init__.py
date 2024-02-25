"""
In FLoX, a `Strategy` is used to define the logic for a specific Federated Learning solution.
"""

from flox.strategies.base import Strategy
from flox.strategies.registry.fedavg import FedAvg
from flox.strategies.registry.fedprox import FedProx
from flox.strategies.registry.fedsgd import FedSGD

__all__ = ["Strategy", "FedSGD", "FedAvg", "FedProx"]
