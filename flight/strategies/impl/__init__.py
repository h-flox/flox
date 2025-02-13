from flight.strategies.impl.fedasync import FedAsync
from flight.strategies.impl.fedavg import FedAvg

# from flight.strategies.impl.fedprox import FedProx
from flight.strategies.impl.fedsgd import FedSGD

__all__ = [
    "FedAsync",
    "FedAvg",
    # "FedProx",
    "FedSGD",
]
