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
from .topologies.node import Node, NodeKind, AggrState
from .topologies.topo import Topology
from ..engine import Engine
from ..learning.modules import HasParameters
from ..learning.modules.prototypes import DataLoadable
from ..strategies.base import Strategy

if t.TYPE_CHECKING:
    from .jobs.types import Result


def log(msg: str):
    # print(f"\t{msg}")
    return None


class SyncFederation(Federation):
    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
        module: HasParameters,
        data: DataLoadable,
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
        self.global_model = module  # None

        self.selected_children: t.Mapping[Node, t.Sequence[Node]] | None = None
        self._when_to_aggr_cbk = all_futures_finished
        self._aggr_job = default_aggr_job
        self._pbar: tqdm | None = None
        self._round_num: int | None = None

    def start(self, rounds: int) -> list[Result]:
        results = []
        for round_no in tqdm(range(rounds)):
            self._round_num = round_no
            res = self.federation_round(round_no)
            results.append(res)
        return results

    def federation_round(self, round_no: int) -> Result:
        log(f"Starting round {round_no}")
        # global_params = self.global_model.get_params()
        # broadcast_records(result, round_no=round_no)
        # NOTE: Be sure to wrap `result` calls to handle errors.
        result = self.federation_step()
        return result

    def federation_step(self) -> Result:
        step_future = self.traverse_step()
        try:
            step_result = step_future.result()
            self.global_model.set_params(step_result.params)
            if self._pbar:
                self._pbar.update()
            return step_result
        except Exception as err:
            raise err

    def traverse_step(
        self,
        node: t.Optional[Node] = None,
        parent: t.Optional[Node] = None,
    ) -> Future[Result]:
        node: Node = self._resolve_node(node)
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
        state = AggrState(coord.idx, children=self.topology.children(coord))
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

        self.selected_children = {node: selected_children}
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
            fut = self.traverse_step(node=child, parent=node)
            child_futs.append(fut)

        aggr_args = AggrJobArgs(
            round_num=self._round_num,
            node=node,
            children=self.topology.children(node),
            child_results=[],  # Note: this is updated in the callback,
            aggr_strategy=self.aggr_strategy,
            transfer=self.engine.data_plane,
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
