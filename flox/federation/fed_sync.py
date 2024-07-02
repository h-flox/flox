from __future__ import annotations

import functools
import typing as t
from concurrent.futures import Future
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from flox.federation.fed import Federation
from flox.federation.future_callbacks import all_child_futures_finished_cbk
from flox.federation.jobs import AggregateJob
from flox.federation.testing import test_model
from flox.federation.topologies import AggrState, Node, NodeKind
from flox.logger import Logger

if t.TYPE_CHECKING:
    from flox.federation.topologies import NodeID, Topology
    from flox.learn import FloxModule
    from flox.learn.data import FloxDataset
    from flox.runtime import ResultFuture
    from flox.runtime.runtime import Runtime
    from flox.strategies import Strategy


class SyncFederation(Federation):
    process_name = "federation::sync"

    def __init__(
        self,
        runtime: Runtime,
        topo: Topology,
        strategy: Strategy,
        module: FloxModule,
        dataset: FloxDataset,
        num_global_rounds: int,
        debug_mode: bool = False,
        logger: Logger | None = None,
        *args,
    ):
        super().__init__(
            runtime=runtime,
            topo=topo,
            num_global_rounds=num_global_rounds,
            module=module,
            dataset=dataset,
            strategy=strategy,
            logger=logger,
            debug_mode=debug_mode,
        )

        # self.runtime = runtime
        # self.topo = flock
        # self.strategy = strategy
        # self.global_model = module
        # self.dataset = dataset
        # self.num_global_rounds = global_rounds
        # self.debug_mode = debug_mode
        # self.logger = logger

        self._selected_children: dict[NodeID, Node] = {}
        self.params = None

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, pd.DataFrame]:
        if debug_mode:
            self.debug_mode = debug_mode
            if self.global_model is None:
                from flox.learn.model import DebugModule

                self.global_model = DebugModule()

        histories = []
        if not self.logger:
            pbar = tqdm(total=self.num_global_rounds, desc=SyncFederation.process_name)
        else:
            pbar = None

        for round_num in range(self.num_global_rounds):
            self.log(f"Starting round {round_num+1}/{self.num_global_rounds}.")
            self.params = self.global_model.state_dict()
            step_result = self.step().result()
            step_result.history["round"] = round_num

            if not debug_mode:
                test_acc, test_loss = test_model(self.global_model)
                step_result.history["test/acc"] = test_acc
                step_result.history["test/loss"] = test_loss

            histories.append(step_result.history)
            self.global_model.load_state_dict(step_result.params)
            if not self.logger:
                pbar.update()

        history = pd.concat(histories)
        return self.global_model, history

    def step(
        self, node: Node | None = None, parent: Node | None = None
    ) -> ResultFuture:
        node = self._resolve_node(node)
        if not isinstance(node, Node):
            raise ValueError("")

        match self.topo.get_kind(node):
            case NodeKind.COORDINATOR:
                self.log("Launching task on the leader.")
                return self.start_coordinator_tasks(node)
            case NodeKind.AGGREGATOR:
                self.log(f"Launching task on aggregator {node.idx}.")
                return self.start_aggregator_tasks(node, self._selected_children[node])
            case NodeKind.WORKER:
                self.log(f"Launching task on worker {node.idx}.")
                return self.start_worker_tasks(node, parent)
            case _:
                k = self.topo.get_kind(node)
                raise ValueError(
                    f"Illegal kind ({k}) of `FlockNode` (ID=`{node.idx}`)."
                )

    def _resolve_node(self, node: Node | None) -> Node:
        """
        Helper function for the recursive `step()` function: simply returns the given `node` if one is provided;
        if `None` is provided, then it returns the coordinator of the topology.

        Args:
            node (Node | None): Node to resolve.

        Returns:

        """
        if node is None:
            assert self.topo.coordinator is not None
            return self.topo.coordinator
        elif isinstance(node, Node):
            return node
        else:
            raise ValueError("SyncProcessV2._handle_node(): Illegal value for {node=}")

    def start_coordinator_tasks(self, node: Node) -> ResultFuture:
        cli_strategy = self.client_strategy
        children = list(self.topo.children(node.idx))
        workers = list(self.topo.workers)
        state = AggrState(node.idx, children, None)

        # Select worker nodes to train the neural network.
        # Then trace parent nodes back to leader.
        self.log("Leader is selecting worker nodes.")
        client_node = self.topo[self.topo.coordinator.idx]
        selected_workers = cli_strategy.select_worker_nodes(state, workers, seed=None)
        intermediate_aggrs = set()
        for worker in selected_workers:
            parent = self.topo.parent(worker)
            while parent != client_node:
                intermediate_aggrs.add(parent)
                parent = self.topo.parent(parent)

        self.log(f"Selected workers: {selected_workers}")

        # Identify the aggregator nodes that are ancestors of the selected worker nodes.
        self.log("Leader is identifying ancestral aggregators to worker nodes.")
        _selected_worker_set = set(selected_workers)
        selected_client_children = set(self.topo.children(client_node))
        if len(intermediate_aggrs) > 0:
            selected_client_children = selected_client_children.intersection(
                intermediate_aggrs
            )
        else:
            selected_client_children = selected_client_children.intersection(
                selected_workers
            )

        self._selected_children = {client_node: selected_client_children}
        for aggr in intermediate_aggrs:
            aggr_succ = set(self.topo.children(aggr))
            aggr_succ = aggr_succ.intersection(
                intermediate_aggrs.union(_selected_worker_set)
            )
            self._selected_children[aggr] = list(aggr_succ)

        # Launch the aggregator task that runs on the client.
        self.log("Submitting aggregation task onto the client.")
        self.log(
            f"Selected children for client is {self._selected_children[client_node]}"
        )
        return self.start_aggregator_tasks(node, self._selected_children[client_node])

    def start_aggregator_tasks(
        self, node: Node, children: t.Iterable[Node] | None = None
    ) -> ResultFuture:
        self.log(f"Preparing to submit AGGREGATION task on node {node.idx}.")
        if children is None:
            children = self.topo.children(node)

        job = AggregateJob()
        _ = AggrState(node.idx, children, None)  # FIXME: This state is never used.
        children_futures = [self.step(child, node) for child in children]
        future: ResultFuture = Future()
        finished_children_cbk = functools.partial(
            all_child_futures_finished_cbk,
            job,
            future,
            children,
            children_futures,
            node,
            self.runtime,
            self.aggr_strategy,
        )
        for child_fut in children_futures:
            child_fut.add_done_callback(finished_children_cbk)

        return future

    def log(self, msg: str):
        ts = str(datetime.now())
        ts = ts.split(".")[0]
        if self.logger:
            # print(f"( {ts} - SyncProcessV2 ) ❯  {msg}")
            self.logger.log(f"( {ts} - SyncProcess)", msg)
