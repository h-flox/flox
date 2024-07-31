import typing as t

from flight.federation.topologies.node import Node


class DataLoadable(t.Protocol):
    """
    The `DataLoadable` is a protocol that defines the key functionalities necessary to load data into
    a federation with Flight.

    Data in federated learning are naturally decentralized across multiple nodes/endpoints. In real-world
    settings, we do not need to worry about modeling the decentralization. But, in simulated settings for
    rapid prototyping we will need to worry about how to break up central datasets into some simulated
    decentralized/federated data distribution.

    A `DataLoadable` object will need to support two main use cases:

    1. simulated workflows where data are centrally located on one machine and we split it up into separate
       subsets for training across multiple nodes/endpoints,
    2. real-world workflows where data already exist on the nodes/endpoints and we only need to **load them
       from disc**.
    """

    def load(self, node: Node, mode: t.Literal["train", "test", "validation"]):
        pass
