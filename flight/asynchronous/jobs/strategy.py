from __future__ import annotations

import typing as t
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from dataclasses import dataclass, field

from torch.utils.data import Subset, TensorDataset

from flight.events import (
    FlightEventEnum,
    add_event_handler_to_obj,
    fire_event_handler_by_type,
)

from flight.jobs.protocols import Result
from flight.jobs.worker import worker_job, WorkerJobArgs
from flight.learning.module import TorchModule
from flight.learning.parameters import Params
from flight.runtime import Runtime

if t.TYPE_CHECKING:
    from flight.learning.parameters import Params
    from flight.strategies.strategy import Strategy
    from flight.system import Topology, Node
    from flight.system.types import NodeID


class AsyncStrategyEvents(FlightEventEnum):
    STARTED = "started"
    COMPLETED = "completed"
    AGGREGATION_COMPLETED = "aggregation_completed"
    WORKER_JOB_STARTED = "worker_job_started"
    WORKER_JOB_COMPLETED = "worker_job_completed"


@dataclass
class AsyncStrategyState:
    global_params: Params
    worker_params: dict[NodeID, Params] = field(default_factory=dict)
    worker_rounds: dict[NodeID, int] = field(default_factory=dict)
    completed_rounds: int = 0


class AsyncStrategy:
    def __init__(
        self,
        runtime: Runtime,
        topology: Topology,
        num_global_rounds: int,
        module: TorchModule,
        dataset: TensorDataset,
        strategy: Strategy,
    ):
        self.runtime = runtime
        self.topology = topology
        self.num_global_rounds = num_global_rounds
        self.module = module
        self.dataset = dataset
        self.strategy = strategy

        initial_params = self.module.get_params()
        self.state = AsyncStrategyState(global_params=initial_params)

        for worker in self.topology.workers:
            self.state.worker_rounds[worker.idx] = 0
            self.state.worker_params[worker.idx] = deepcopy(initial_params)

    def start(self) -> tuple[TorchModule, t.Any]:
        self.fire_event_handler(AsyncStrategyEvents.STARTED)

        futures = {
            self._dispatch_worker_job(worker) for worker in self.topology.workers
        }

        while futures:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in dones:
                result: Result = future.result()
                worker_node_id = result.node.idx

                if result.params is not None:
                    self.state.worker_params[worker_node_id] = result.params
                
                self.fire_event_handler(
                    AsyncStrategyEvents.WORKER_JOB_COMPLETED, {"result": result}
                )

                self.partial_aggregation_policy(last_updated_node=worker_node_id)
                self.state.completed_rounds += 1

                if self.state.worker_rounds[worker_node_id] < self.num_global_rounds:
                    self.state.worker_rounds[worker_node_id] += 1
                    new_future = self._dispatch_worker_job(self.topology[worker_node_id])
                    futures.add(new_future)

        self.fire_event_handler(AsyncStrategyEvents.COMPLETED)
        self.module.set_params(self.state.global_params)
        return self.module, None # No history for now

    def _dispatch_worker_job(self, worker_node: Node) -> Future:
        worker_dataset = self._get_dataset_for_worker(worker_node.idx)
        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.module,
            data=worker_dataset,
            params=self.state.global_params,
            node=worker_node,
        )
        future = self.runtime.submit(worker_job, args)
        self.fire_event_handler(AsyncStrategyEvents.WORKER_JOB_STARTED, {"worker_id": worker_node.idx})
        return future

    def partial_aggregation_policy(self, last_updated_node: t.Optional[NodeID] = None, *args, **kwargs):
        """
        Implements partial aggregation policy using FedAvg algorithm.
        
        This method aggregates model parameters from all workers using weighted averaging.
        In asynchronous FL, we consider all workers to have participated by contributing
        their most recent parameters.
        
        Args:
            last_updated_node: The node that just completed its training round
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        if not self.state.worker_params:
            return
            
        valid_params = {
            node_id: params 
            for node_id, params in self.state.worker_params.items() 
            if params is not None
        }
        
        if not valid_params:
            return

    
        first_params = next(iter(valid_params.values()))
        aggregated_params = deepcopy(first_params)
        
        for node_id, params in valid_params.items():
            if params is first_params:
                continue
                
            for key in aggregated_params:
                if key in params:
                    aggregated_params[key] += params[key]
        
        num_workers = len(valid_params)
        for key in aggregated_params:
            aggregated_params[key] /= num_workers
            
        self.state.global_params = aggregated_params
        
        self.fire_event_handler(
            AsyncStrategyEvents.AGGREGATION_COMPLETED,
            {
                "completed_aggregations": self.state.completed_rounds,
                "num_workers_aggregated": num_workers,
                "last_updated_node": last_updated_node,
                "worker_ids": list(valid_params.keys())
            },
        )

    def _get_dataset_for_worker(self, worker_id: NodeID) -> Subset:
        all_workers = list(self.topology.workers)
        worker_indices = list(range(len(self.dataset)))
        worker_idx = [n.idx for n in all_workers].index(worker_id)
        num_workers = len(all_workers)
        indices_per_worker = len(worker_indices) // num_workers
        start = worker_idx * indices_per_worker
        end = start + indices_per_worker
        if worker_idx == num_workers - 1:
            end = len(worker_indices)
        
        return Subset(self.dataset, worker_indices[start:end])

    def add_event_handler(self, event_type, handler):
        add_event_handler_to_obj(self, event_type, handler)

    def fire_event_handler(self, event_type, context=None):
        fire_event_handler_by_type(self, event_type, context)