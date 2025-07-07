from __future__ import annotations

import time
import typing as t
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch.utils.data import Subset, TensorDataset
import numpy as np

from flight.events import FlightEventEnum
from flight.jobs.protocols import Result
from flight.jobs.worker import WorkerJobArgs, worker_job
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from ignite.engine import Engine, Events

if t.TYPE_CHECKING:
    from flight.strategies.strategy import Strategy
    from flight.system.node import Node
    from flight.system.topology import Topology


class WorkerTimeTracker:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.job_times = {i: [] for i in range(num_workers)}
        self._job_start_times = {}

    def record_job_start(self, worker_idx):
        self._job_start_times[worker_idx] = time.time()
        if worker_idx not in self.job_times:
            self.job_times[worker_idx] = []

    def record_job_end(self, worker_idx):
        start = self._job_start_times.pop(worker_idx, None)
        if start is not None:
            end = time.time()
            if worker_idx not in self.job_times:
                self.job_times[worker_idx] = []
            self.job_times[worker_idx].append((start, end))
            print(f"Recorded job end for worker {worker_idx}: start={start}, end={end}")
        else:
            print(
                f"record_job_end called for worker {worker_idx} but no start time "
                f"found!"
            )


class AsyncWorkflowEvents(FlightEventEnum):
    STARTED = "started"
    """
    Triggered at the start of a federation.
    """

    COMPLETED = "completed"
    """
    Triggered at the end of a federation.
    """

    AGGREGATION_COMPLETED = "aggregation_completed"
    """
    Triggered at the end of an aggregation.
    """

    WORKER_JOB_STARTED = "worker_job_started"
    """
    Triggered at the start of a worker job.
    """

    WORKER_JOB_COMPLETED = "worker_job_completed"
    """
    Triggered at the end of a worker job.
    """


@dataclass
class AsyncWorkflowState:
    global_params: t.Any
    """
    The global parameters of the model.
    """
    worker_params: dict = field(default_factory=dict)
    """
    The parameters of the workers.
    """
    worker_rounds: dict = field(default_factory=dict)
    """
    The number of rounds each worker has completed.
    """
    completed_worker_jobs: int = 0
    """
    The number of completed worker jobs.
    """


def evaluate_fn(model, data_loader, device=None):
    """
    Computes the loss for each batch (round) over the data_loader using Ignite Engine.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader for evaluation.
        device: Device to run the model on (optional).

    Returns:
        list: List of loss values, one per batch/round.
    """
    model.eval()
    criterion = model.configure_criterion()
    total_losses = []

    def eval_step(engine, batch):
        x, y = batch
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)
        return loss.item()

    evaluator = Engine(eval_step)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def collect_loss(engine): #noqa
        total_losses.append(engine.state.output)

    evaluator.run(data_loader)
    return total_losses


class AsyncWorkflow:
    """
    Handles the asynchronous strategy for federated learning, managing the dispatching
    and aggregation of worker jobs.

    Args:
        runtime (Runtime): The runtime to use for the workflow.
        topology (Topology): The topology of the network.
        num_global_rounds (int): The number of global rounds to run.
        module (TorchModule): The model to train.
        dataset (TensorDataset): The dataset to use for training.
        strategy (Strategy): The strategy to use for the workflow.
        aggregation_policy (Callable): The aggregation policy to use for the
        workflow.
        worker_time_tracker (WorkerTimeTracker): The worker time tracker to use for
        the workflow.
        loss_function (Callable): Function to compute loss, signature
        (model, data_loader, device) -> float.
    """

    def __init__(
        self,
        runtime: Runtime,
        topology: Topology,
        num_global_rounds: int,
        module: TorchModule,
        dataset: TensorDataset,
        strategy: Strategy,
        aggregation_policy: t.Callable[[t.Any, int | None], None] | None = None,    
    ):
        """
        Initializes the asynchronous workflow.

        Args:
            runtime (Runtime): The runtime to use for the workflow.
            topology (Topology): The topology of the network.
            num_global_rounds (int): The number of global rounds to run.
            module (TorchModule): The model to train.
            dataset (TensorDataset): The dataset to use for training.
            strategy (Strategy): The strategy to use for the workflow.
            aggregation_policy (Callable): The aggregation policy to use for the
                workflow.
            data_loader, device) -> float.
        Returns:
            None
        """
        self.runtime = runtime
        self.topology = topology
        self.num_global_rounds = num_global_rounds
        self.module = module
        self.dataset = dataset
        self.strategy = strategy
        self.aggregation_policy = aggregation_policy
        self.worker_time_tracker = WorkerTimeTracker(len(self.topology.workers))
        self.loss_function = evaluate_fn

        initial_params = self.module.get_params()
        self.state = AsyncWorkflowState(global_params=initial_params)

        for worker in self.topology.workers:
            self.state.worker_rounds[worker.idx] = 0
            self.state.worker_params[worker.idx] = deepcopy(initial_params)

    def start(self) -> tuple[TorchModule, t.Any, dict]:
        """
        Starts the asynchronous federated learning strategy by dispatching worker
        jobs.

        Returns:
            The final model, per-round global loss list, and per-worker loss dict.
        """
        # self.fire_event_handler(AsyncWorkflowEvents.STARTED)

        num_workers = len(self.topology.workers)
        max_jobs = num_workers * self.num_global_rounds
        jobs_completed = 0
        worker_idx_map = {worker.idx: worker for worker in self.topology.workers}

        futures = set()
        for worker in self.topology.workers:
            if self.state.worker_rounds[worker.idx] < self.num_global_rounds:
                futures.add(self._dispatch_worker_job(worker))
                self.state.worker_rounds[worker.idx] += 1

        per_round_losses = []
        # Evaluate initial model before any training
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=32)
        per_round_losses.append(self.loss_function(self.module, data_loader))

        # Track per-worker loss per round
        worker_losses: dict[int, list[float]] = {worker.idx: [] for worker in self.topology.workers}

        rounds_completed = 0
        while futures and jobs_completed < max_jobs:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in dones:
                try:
                    result: Result = future.result()
                except Exception as exc:
                    print(f"Worker job failed with exception: {exc}")
                    continue

                worker_node_id = result.node.idx

                if result.params is not None:
                    self.state.worker_params[worker_node_id] = result.params

                if self.worker_time_tracker is not None:
                    self.worker_time_tracker.record_job_end(worker_node_id)

                if self.aggregation_policy:
                    self.aggregation_policy(self, worker_node_id)
                else:
                    self.partial_aggregation_policy(last_updated_node=worker_node_id)
                self.state.completed_worker_jobs += 1
                jobs_completed += 1

                if self.state.worker_rounds[worker_node_id] < self.num_global_rounds:
                    new_future = self._dispatch_worker_job(
                        worker_idx_map[worker_node_id]
                    )
                    futures.add(new_future)
                    self.state.worker_rounds[worker_node_id] += 1

                # Evaluate and record this worker's loss on its own data after its job completes
                worker_model = deepcopy(self.module)
                worker_model.set_params(self.state.worker_params[worker_node_id])
                worker_data = self._get_dataset_for_worker(worker_node_id)
                worker_loader = torch.utils.data.DataLoader(worker_data, batch_size=32)
                loss_list = self.loss_function(worker_model, worker_loader)
                # Store mean loss for this round for this worker
                worker_losses[worker_node_id].append(float(np.mean(loss_list)))

            # After each global round (when all workers have completed a round), evaluate and store loss
            rounds_completed += 1
            if jobs_completed % num_workers == 0:
                self.module.set_params(self.state.global_params)
                per_round_losses.append(self.loss_function(self.module, data_loader))

        self.module.set_params(self.state.global_params)
        return self.module, per_round_losses, worker_losses

    def _dispatch_worker_job(self, worker_node: Node) -> Future:
        """
        Dispatches a worker job to the specified worker node.

        Args:
            worker_node (Node): The worker node to dispatch the job to.

        Returns:
            The future representing the worker job.
        """

        worker_dataset = self._get_dataset_for_worker(worker_node.idx)
        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.module,
            data=worker_dataset,
            params=self.state.global_params,
            node=worker_node,
        )
        # Record job start
        if self.worker_time_tracker is not None:
            self.worker_time_tracker.record_job_start(worker_node.idx)
        future = self.runtime.submit(worker_job, args)
        # self.fire_event_handler(
        #    AsyncWorkflowEvents.WORKER_JOB_STARTED, {"worker_id": worker_node.idx}
        # )
        return future

    def partial_aggregation_policy(
        self, 
        last_updated_node: int | None = None,
    ) -> None:
        """
        Implements partial aggregation policy using the FedAvg algorithm.
        Aggregates model parameters from all workers using weighted averaging,
        where each worker's weight is proportional to the number of data
        samples it holds.

        Args:
            last_updated_node (int | None): The ID of the last updated node. 
                Defaults to `None`.
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

        n_k = {}
        for node_id in valid_params:
            worker_dataset = self._get_dataset_for_worker(node_id)
            n_k[node_id] = len(worker_dataset)
        
        n = sum(n_k.values())
        if n == 0:
            return  # Avoid division by zero
        weights = {node_id: n_k[node_id] / n for node_id in valid_params}

        first_params = next(iter(valid_params.values()))
        aggregated_params = deepcopy(first_params)
        for key in aggregated_params:
            aggregated_params[key] = aggregated_params[key] * 0.0

        for node_id, params in valid_params.items():
            for key in aggregated_params:
                if key in params:
                    aggregated_params[key] += params[key] * weights[node_id]

        self.state.global_params = aggregated_params

        # self.fire_event_handler(
        #    AsyncWorkflowEvents.AGGREGATION_COMPLETED,
        #    {
        #        "completed_worker_jobs": self.state.completed_worker_jobs,
        #        "num_workers_aggregated": len(valid_params),
        #        "last_updated_node": last_updated_node,
        #        "worker_ids": list(valid_params.keys()),
        #        "weights": weights,
        #        "n_k": n_k,
        #        "n": n,
        #    },
        #)

    def _get_dataset_for_worker(self, worker_id: int) -> Subset:
        """
        Returns a subset of the dataset for the specified worker ID.

        Args:
            worker_id (int): The ID of the worker.

        Returns:
            The subset of the dataset for the specified worker.
        """
        all_workers = list(self.topology.workers)
        worker_indices = list(range(len(self.dataset)))
        
        try:
            worker_idx = [n.idx for n in all_workers].index(worker_id)
        except ValueError:
            raise ValueError(f"Worker ID {worker_id} not found in topology workers.")
        
        num_workers = len(all_workers)
        indices_per_worker = len(worker_indices) // num_workers
        start = worker_idx * indices_per_worker
        end = start + indices_per_worker
        if worker_idx == num_workers - 1:
            end = len(worker_indices)
        
        return Subset(self.dataset, worker_indices[start:end])
