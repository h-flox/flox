from __future__ import annotations

import functools
import typing as t
from concurrent.futures import Future
from copy import deepcopy
from datetime import datetime

import pandas as pd
from flox.jobs import LocalTrainJob, AggregateJob, DebugLocalTrainJob
from tqdm import tqdm

from flox.process.future_callbacks import all_child_futures_finished_cbk
from flox.process.process import Process
from flox.process.testing import test_model
from flox.topos import Node, NodeKind, AggrState

if t.TYPE_CHECKING:
    from flox import Topology
    from flox.data import FloxDataset
    from flox.learn import FloxModule
    from flox.runtime.runtime import Runtime
    from flox.runtime import Result
    from flox.strategies import (
        Strategy,
        TrainerStrategy,
        AggregatorStrategy,
        WorkerStrategy,
        ClientStrategy,
    )


class SyncProcess(Process):
    pbar_desc = "federated_fit::sync"

    def __init__(
        self,
        runtime: Runtime,
        flock: Topology,
        strategy: Strategy,
        module: FloxModule | None,
        dataset: FloxDataset,
        global_rounds: int,
        debug_mode: bool = False,
        logging: bool = False,
    ):
        self.runtime = runtime
        self.flock = flock
        self.strategy = strategy
        self.global_model = module
        self.dataset = dataset
        self.global_rounds = global_rounds
        self.debug_mode = debug_mode
        self._selected_children = {}
        self.logging = logging
        self.params = None

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, pd.DataFrame]:
        if debug_mode:
            self.debug_mode = debug_mode
            if self.global_model is None:
                from flox.process.debug_utils import DebugModule

                self.global_model = DebugModule()

        histories = []
        if not self.logging:
            pbar = tqdm(total=self.global_rounds, desc=self.pbar_desc)
        else:
            pbar = None

        for round_num in range(self.global_rounds):
            self.log(f"Starting round {round_num+1}/{self.global_rounds}.")
            self.params = self.global_model.state_dict()
            step_result = self.step().result()
            step_result.history["round"] = round_num

            if not debug_mode:
                test_acc, test_loss = test_model(self.global_model)
                step_result.history["test/acc"] = test_acc
                step_result.history["test/loss"] = test_loss

            histories.append(step_result.history)
            self.global_model.load_state_dict(step_result.params)
            if not self.logging:
                pbar.update()

        history = pd.concat(histories)
        return self.global_model, history

    def step(
        self, node: Node | None = None, parent: Node | None = None
    ) -> Future[Result]:
        node = self._handle_node(node)
        match self.flock.get_kind(node):
            case NodeKind.COORDINATOR:
                self.log("Launching task on the leader.")
                return self._leader_tasks(node)
            case NodeKind.AGGREGATOR:
                self.log(f"Launching task on aggregator {node.idx}.")
                return self._aggregator_tasks(node, self._selected_children[node])
            case NodeKind.WORKER:
                self.log(f"Launching task on worker {node.idx}.")
                return self._worker_tasks(node, parent)
            case _:
                k = self.flock.get_kind(node)
                raise ValueError(
                    f"Illegal kind ({k}) of `FlockNode` (ID=`{node.idx}`)."
                )

    def _handle_node(self, node: Node | None) -> Node:
        if node is None:
            assert self.flock.coordinator is not None
            return self.flock.coordinator
        elif isinstance(node, Node):
            return node
        else:
            raise ValueError("SyncProcessV2._handle_node(): Illegal value for {node=}")

    def _leader_tasks(self, node: Node) -> Future[Result]:
        cli_strategy = self.client_strategy
        children = list(self.flock.children(node.idx))
        workers = list(self.flock.workers)
        state = AggrState(node.idx, children, None)

        # STEP 1: Select worker nodes to train the neural network. Then trace parent nodes back to leader.
        self.log("Leader is selecting worker nodes.")
        client_node = self.flock[self.flock.coordinator.idx]
        selected_workers = cli_strategy.select_worker_nodes(state, workers, seed=None)
        intermediate_aggrs = set()
        for worker in selected_workers:
            parent = self.flock.parent(worker)
            while parent != client_node:
                intermediate_aggrs.add(parent)
                parent = self.flock.parent(parent)
                # self.log(f"Worker selection phase: {worker=}, {parent=}")

        self.log(f"Selected workers: {selected_workers}")

        # STEP 2: Identify the aggregator nodes that are ancestors of the selected worker nodes.
        self.log("Leader is identifying ancestral aggregators to worker nodes.")
        _selected_worker_set = set(selected_workers)
        selected_client_children = set(self.flock.children(client_node))
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
            aggr_succ = set(self.flock.children(aggr))
            aggr_succ = aggr_succ.intersection(
                intermediate_aggrs.union(_selected_worker_set)
            )
            self._selected_children[aggr] = list(aggr_succ)

        # STEP 3: Launch the aggregator task that runs on the client.
        self.log("Submitting aggregation task onto the client.")
        self.log(
            f"Selected children for client is {self._selected_children[client_node]}"
        )
        return self._aggregator_tasks(node, self._selected_children[client_node])

    def _aggregator_tasks(
        self, node: Node, children: t.Iterable[Node] | None = None
    ) -> Future[Result]:
        self.log(f"Preparing to submit AGGREGATION task on node {node.idx}.")
        if children is None:
            children = self.flock.children(node)

        job = AggregateJob()
        state = AggrState(node.idx, children, None)
        children_futures = [self.step(child, node) for child in children]
        future: Future[Result] = Future()
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

    def _worker_tasks(self, node: Node, parent: Node) -> Future[Result]:
        self.log(f"Preparing to submit WORKER task on node {node.idx}.")

        if self.debug_mode:
            self.log("Using debug job, `pure_debug_train_job`.")
            job = DebugLocalTrainJob()
            dataset = None
            model_state_dict = None
        else:
            job = LocalTrainJob()
            dataset = self.runtime.transfer(self.dataset)
            model_state_dict = self.runtime.transfer(self.params)

        return self.runtime.submit(
            job,
            node=node,
            parent=parent,
            global_model=self.runtime.transfer(deepcopy(self.global_model)),
            module_state_dict=model_state_dict,
            worker_strategy=self.worker_strategy,
            trainer_strategy=self.trainer_strategy,
            dataset=dataset,
        )

    @property
    def client_strategy(self) -> ClientStrategy:
        return self.strategy.client_strategy

    @property
    def aggr_strategy(self) -> AggregatorStrategy:
        return self.strategy.aggr_strategy

    @property
    def worker_strategy(self) -> WorkerStrategy:
        return self.strategy.worker_strategy

    @property
    def trainer_strategy(self) -> TrainerStrategy:
        return self.strategy.trainer_strategy

    def log(self, msg: str):
        ts = str(datetime.now())
        ts = ts.split(".")[0]
        if self.logging:
            print(f"( {ts} - SyncProcessV2 ) ❯  {msg}")
