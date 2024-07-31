import functools
import typing as t
from concurrent.futures import Future

from ..learning.modules import HasParameters
from ..learning.modules.base import DataLoadable
from ..strategies.base import Strategy
from .fed_abs import Federation
from .topologies.node import Node, NodeKind
from .topologies.topo import Topology

if t.TYPE_CHECKING:
    from .jobs.types import AggrJobArgs, Result

    Engine: t.TypeAlias = t.Any


def log(msg: str):
    print(msg)


class SyncFederation(Federation):
    def __init__(
        self,
        module: HasParameters,
        data: DataLoadable,
        topology: Topology,
        strategy: Strategy,
        # engine: Engine,
        #
        logger=None,
        debug=None,
    ):
        super().__init__(topology, strategy)
        self.module = module
        self.data = data
        self.engine = Engine()
        # self.engine = engine
        self.exceptions = []
        self.global_model = None

        self.selected_children: t.Mapping[Node, t.Sequence[Node]] | None = None

    def start(self, rounds: int):
        for round_no in range(rounds):
            self.federation_round(round_no)

    def federation_round(self, round_no: int):
        log("Starting round")
        global_params = self.global_model.get_params()
        # NOTE: Be sure to wrap `result` calls to handle errors.
        try:
            round_future = self.federation_step()
            round_results = round_future.result()
        except Exception as exc:
            self.exceptions.append(exc)
            raise exc

    def federation_step(self) -> Future:
        self.params = self.global_model.state_dict()
        step_result = self.traverse_step().result()
        step_result.history["round"] = round_num

        if not debug_mode:
            trainer = Trainer()
            test_results = trainer.test(model, test_dataloader)
            # test_acc, test_loss = test_model(self.global_model)
            # step_result.history["test/acc"] = test_acc
            # step_result.history["test/loss"] = test_loss

        histories.append(step_result.history)
        self.global_model.load_state_dict(step_result.params)

        if self.pbar:
            self.pbar.update()

    def traverse_step(
        self,
        node: t.Optional[Node] = None,
        parent: t.Optional[Node] = None,
    ) -> Future[Result]:
        node = self._resolve_node(node)
        match node.kind:
            case NodeKind.COORDINATOR:
                log(f"Launching task on the {node.kind.title()}.")
                return self.start_coordinator_task(node)
            case NodeKind.AGGREGATOR:
                log(
                    f"Launching aggregation task on {node.kind.title()} "
                    f"node {node.idx}."
                )
                return self.start_aggregator_task(node, self.selected_children[node])
            case NodeKind.WORKER:
                log(f"Launching worker task on {node.kind.title()} node {node.idx}.")
                return self.start_worker_task(node, parent)

    def start_coordinator_task(
        self,
        node: Node,
    ) -> Future[Result]:
        selected_workers = self.coord_strategy.select_workers(
            state, self.topology.workers, seed=None
        )

        # Identify any intermediate aggregators that are on the path between the
        # coordinator and the selected worker nodes.
        intermediate_aggrs = set()
        for worker in selected_workers:
            parent = self.topology.parent(worker)
            while parent != self.topology.coordinator:
                intermediate_aggrs.add(parent)
                parent = self.topology.parent(parent)

        # Identify the immediate children of the coordinator node that are part
        # of this federation based on the above code.
        worker_set = set(selected_workers)
        coord_children = set(self.topology.children(node))
        if len(intermediate_aggrs):
            selected_children = coord_children.intersection(intermediate_aggrs)
        else:
            selected_children = coord_children.intersection(worker_set)

        self.selected_children = {node: selected_children}
        for aggr in intermediate_aggrs:
            included_aggrs_and_workers = intermediate_aggrs.union(worker_set)
            aggr_children = set(self.topology.children(aggr))
            aggr_children = aggr_children.intersection(included_aggrs_and_workers)
            self.selected_children = list(aggr_children)

        return self.start_aggregator_task(node, self.selected_children[node])

        # Identify which child nodes are necessary in this round of the federation.
        # Necessary nodes are all the selected worker nodes and any other node

    def start_aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        future = Future()
        children_futures = [
            self.traverse_step(node=child, parent=node) for child in selected_children
        ]
        args = AggrJobArgs(
            # ...,
            future=future,
            children=selected_children,
            children_futures=children_futures,
        )

        aggregation_callback = functools.partial(..., self.aggr_fn, args)
        for child_future in children_futures:
            child_future.add_done_callback(aggregation_callback)
        return future
