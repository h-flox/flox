"""
In FLoX, a `Strategy` is used to define the logic for a specific Federated Learning solution.
"""

from flox.strategies_depr.base import Strategy
from flox.strategies_depr.registry.fedavg import FedAvg
from flox.strategies_depr.registry.fedprox import FedProx
from flox.strategies_depr.registry.fedsgd import FedSGD

__all__ = ["Strategy", "FedSGD", "FedAvg", "FedProx"]
