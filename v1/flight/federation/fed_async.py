from .fed_abs import Federation


class AsyncFederation(Federation):
    pass


'''
import typing as t
from concurrent.futures import Future

from .fed_abs import Federation
from .topologies.node import Node


class AsyncFederation(Federation):
    def __init__(self):
        super().__init__()

    def start_aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        """
        Raises:
            - NotImplementedError: At this time, asynchronous federated learning is
              *not* supported by Flight for hierarchical topologies. Thus, the only
              node that will launch aggregator tasks is the coordinator node. This
              means this class method is not necessary at this time.
        """
        raise NotImplementedError(
            "This method is not implemented. Async federations only support 2-tier "
            "topologies (i.e., there are no intermediate aggregators)."
        )
'''
