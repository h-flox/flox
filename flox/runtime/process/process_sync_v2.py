from concurrent.futures import Future

from flox import Flock
from flox.flock import FlockNode, NodeKind, AggrState
from flox.nn import FloxModule
from flox.runtime import Result
from flox.runtime.process.process import Process
from flox.strategies import Strategy


class SyncProcessV2(Process):
    flock: Flock
    strategy: Strategy
    global_module: FloxModule

    def __init__(self):
        pass

    def start(
        self, node: FlockNode | None = None, parent: FlockNode | None = None
    ) -> Future[Result]:
        pass

    def step(
        self, node: FlockNode | None = None, parent: FlockNode | None = None
    ) -> Future[Result]:
        node = self._handle_node(node)
        match self.flock.get_kind(node):
            case NodeKind.LEADER:
                return self._leader_tasks(node)
            case NodeKind.AGGREGATOR:
                return self._aggregator_tasks(node)
            case NodeKind.WORKER:
                return self._worker_tasks(node)
            case _:
                k = self.flock.get_kind(node)
                raise ValueError(
                    f"Illegal kind ({k}) of `FlockNode` (ID=`{node.idx}`)."
                )

    def _handle_node(self, node: FlockNode | None) -> FlockNode:
        if node is None:
            assert self.flock.leader is not None
            return self.flock.leader
        elif isinstance(node, FlockNode):
            return node
        else:
            raise ValueError

    def _leader_tasks(self, node: FlockNode) -> Future[Result]:
        cli_strategy = self.strategy.client_strategy
        children = list(self.flock.children(node.idx))
        workers = list(self.flock.workers)
        state = AggrState(node.idx, children, self.global_module)

        selected_workers = cli_strategy.select_worker_nodes(state, workers, seed=None)
        intermediate_aggrs = set()
        for worker in selected_workers:
            self.flock.topo.

    def _aggregator_tasks(self) -> Future[Result]:
        pass

    def _worker_tasks(self) -> Future[Result]:
        pass
