import typing as t

from .fed_abs import Federation
from .topologies.node import Node, NodeKind


class AsyncFederation(Federation):
    def __init__(self):
        pass

    def start_aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        raise NotImplementedError(
            "This method is not implemented. Async federations only support 2-tier topologies "
            "(i.e., there are no intermediate aggregators)."
        )
