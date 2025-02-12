from __future__ import annotations

import functools
import typing as t
from concurrent.futures import Future

from numpy.random import default_rng
from tqdm import tqdm

from .fed_abs import Federation
from .future_callbacks import all_futures_finished
from .jobs.aggr import default_aggr_job
from .jobs.types import AggrJobArgs
from .records import broadcast_records
from .topologies.node import Node, NodeKind, AggrState
from .topologies.topo import Topology
from ..engine import Engine
from ..types import Record

if t.TYPE_CHECKING:
    from ..strategies.base import Strategy
    from .jobs import Result, TrainJob
    from ..learning.base import AbstractDataModule, AbstractModule


def log(msg: str):
    # print(f"\t{msg}")
    return None


class SyncFederation(Federation):
    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
        module: AbstractModule,
        data: AbstractDataModule,
        work_fn: TrainJob | None = None,
        # engine: Engine,
        #
        logger=None,
        debug=None,
    ):
        super().__init__(topology, strategy, work_fn=work_fn)
        self.module = module
        self.data = data
        self.engine = Engine(None, None)  # TODO
        self.exceptions = []
        self.global_model = module  # None

        self.selected_children: t.Mapping[Node, t.Sequence[Node]] | None = None
        self._when_to_aggr_cbk = all_futures_finished
        self._aggr_job = default_aggr_job
        self._pbar: tqdm | None = None
        self._round_num: int = 0

    def start(self, rounds: int) -> list[Record]:
        # TODO: I do NOT think we need `start` and `federation_round` to be separate.
        results = []
        records = []
        self._round_num = 0

        for round_no in tqdm(range(rounds)):
            self._round_num = round_no
            res = self.federation_round(round_no)
            results.append(res)
            records.extend(res.records)

        # return results
        return records

    def federation_round(self, round_no: int) -> Result:
        log(f"Starting round {round_no}")
        step_future = self.launch_tasks()

        # TODO: Reconsider how this is implemented in case there is a communication or
        #       other execution error.
        try:
            step_result = step_future.result()
        except Exception as err:
            self.engine.controller.shutdown()
            raise err

        # TEST THE GLOBAL MODEL.
        coord = self.topology.coordinator
        test_data = self.data.test_data(coord)
        if test_data:
            _ = self.global_model.test_step(test_data)  # TODO
            test_results = {"test/acc": -1, "test/loss": -1}
            broadcast_records(step_result.records, **test_results)

        # UPDATE PROGRESS BAR.
        self.global_model.set_params(step_result.params)
        if self._pbar:
            self._pbar.update()

        return step_result

    def launch_tasks(
        self,
        node: Node | None = None,
        parent: Node | None = None,
    ) -> Future[Result]:
        """
        Launches the proper task on a given `node`.

        This function should only be used internally within the `SyncFederation` class.
        This is *first* called during a call to `federation_round()`. That first call
        will give no arguments (i.e., `node=None` and `parent=None`). More simply,
        this means that the first node to begin launching its task will be the
        coordinator.

        From there, the function uses a breadth-first traversal to launch the tasks on
        the other nodes in the topology by traversing down to the leaves of the tree--
        like topology.

        Args:
            node (Node | None): The node to start tasks on. If `None`, then the
                coordinator of the topology (i.e., root node) is used. Defaults to
                `None`.
            parent (Node | None): The parent node of the argument `node`.
                Defaults to `None`.

        Returns:
            The `Future` returned by the task launched on `node`.
        """
        node = self._resolve_node(node)
        assert isinstance(node, Node)

        match node.kind:
            case NodeKind.COORD:
                log(f"Launching task on the {node.kind.title()}.")
                return self.coordinator_task(node)

            case NodeKind.AGGR:
                log(
                    f"Launching aggregation task on {node.kind.title()} "
                    f"node {node.idx}."
                )
                return self.aggregator_task(node, self.selected_children[node])

            case NodeKind.WORKER:
                log(f"Launching worker task on {node.kind.title()} node {node.idx}.")
                return self.worker_task(node, parent)

    def coordinator_task(self, node: Node) -> Future[Result]:
        coord = self.topology.coordinator
        children = list(self.topology.children(coord))
        state = AggrState(coord.idx, children=children)
        selected_workers = self.coord_strategy.select_workers(
            state, self.topology.workers, default_rng()
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
        if len(intermediate_aggrs) > 0:
            selected_children = coord_children.intersection(intermediate_aggrs)
        else:
            selected_children = coord_children.intersection(worker_set)

        self.selected_children = {node: list(selected_children)}
        for aggr in intermediate_aggrs:
            included_aggrs_and_workers = intermediate_aggrs.union(worker_set)
            aggr_children = set(self.topology.children(aggr))
            aggr_children = aggr_children.intersection(included_aggrs_and_workers)
            self.selected_children = list(aggr_children)

        future = self.aggregator_task(node, self.selected_children[node])
        log(f"Finished aggregator task on coordinator -- {future.done()=}.")
        return future

        # Identify which child nodes are necessary in this round of the federation.
        # Necessary nodes are all the selected worker nodes and any other node

    def aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        job = self._aggr_job
        parent_fut: Future = Future()

        child_futs = []
        for child in selected_children:
            fut = self.launch_tasks(node=child, parent=node)
            child_futs.append(fut)

        children = list(self.topology.children(node))
        aggr_args = AggrJobArgs(
            round_num=self._round_num,
            node=node,
            children=children,
            child_results=[],  # NOTE: this is updated in the callback,
            aggr_strategy=self.aggr_strategy,
            transfer=self.engine.transmitter,
        )

        cbk = functools.partial(
            self._when_to_aggr_cbk,
            job,
            aggr_args,
            parent_fut,
            child_futs,
            self.engine,
        )

        for fut in child_futs:
            fut.add_done_callback(cbk)

        return parent_fut
