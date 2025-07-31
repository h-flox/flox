from __future__ import annotations

import typing as t
from concurrent.futures import FIRST_COMPLETED, Future, wait
from copy import deepcopy
from dataclasses import dataclass, field
import datetime
from typing import Optional
from torch.utils.data import DataLoader

import torch.nn as nn
from ignite.metrics import Loss

import torch
from torch.utils.data import Subset, TensorDataset

from flight.learning.module import FederatedDataModule
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
    Evaluates the model on the given data_loader using Ignite Engine and returns the average loss.
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()

    def eval_step(engine, batch):
        x, y = batch
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        with torch.no_grad():
            logits = model(x)
        return logits, y 

    evaluator = Engine(eval_step)
    Loss(criterion).attach(evaluator, 'loss')

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_eval_results(engine):
        print(f"Eval Epoch {engine.state.epoch}, Avg Loss: {engine.state.metrics['loss']:.4f}")

    evaluator.run(data_loader, max_epochs=3)
    avg_loss = evaluator.state.metrics['loss']
    return avg_loss


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
        loss_function (Callable): Function to compute loss, signature
        (model, data_loader, device) -> float.
    """

    def __init__(
        self,
        runtime: Runtime,
        topology: Topology,
        num_global_rounds: int,
        module: TorchModule,
        dataset: FederatedDataModule,
        test_dataset: FederatedDataModule,
        strategy: Strategy,
        aggregation_policy: t.Callable[[t.Any, int | None], None] | None = None,
        # loss_function: Optional[t.Callable] = None,
    ):
        """
        Initializes the asynchronous workflow.

        Args:
            runtime (Runtime): The runtime to use for the workflow.
            topology (Topology): The topology of the network.
            num_global_rounds (int): The number of global rounds to run.
            module (TorchModule): The model to train.
            dataset (FederatedDataModule): The dataset to use for training.
            strategy (Strategy): The strategy to use for the workflow.
            aggregation_policy (Callable): The aggregation policy to use for the
                workflow.
            loss_function (Callable): Function to compute loss, with signature 
                (model, data_loader, device) -> float. Defaults to evaluate_fn.
            round_logs (list[dict]): List to store logs for each round, including 
                metrics such as loss, time, and worker index.
        Returns:
            None
        """
        self.runtime = runtime
        self.topology = topology
        self.num_global_rounds = num_global_rounds
        self.module = module
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.strategy = strategy
        self.aggregation_policy = aggregation_policy
        self.evaluation_function = evaluate_fn
        self.round_logs: list[dict[str, t.Any]] = []
        self.global_params_history = []

        initial_params = self.module.get_params()
        self.state = AsyncWorkflowState(global_params=initial_params)

        for worker in self.topology.workers:
            self.state.worker_rounds[worker.idx] = 0
            self.state.worker_params[worker.idx] = deepcopy(initial_params)

    def start(self) -> tuple[TorchModule, list]:
        """
        Starts the asynchronous federated learning strategy by dispatching worker
            jobs.

        Returns:
            The final model and per-round logs.
        """

        num_workers = len(self.topology.workers)
        max_jobs = num_workers * self.num_global_rounds
        jobs_completed = 0
        rounds_completed = 0
        worker_idx_map = {worker.idx: worker for worker in self.topology.workers}

        futures = set()
        for worker in self.topology.workers:
            if self.state.worker_rounds[worker.idx] < self.num_global_rounds:
                futures.add(self._dispatch_worker_job(worker))

        while futures and jobs_completed < max_jobs:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in dones:
                try:
                    result: Result = future.result()
                except Exception as exc:
                    print(f"Worker job failed with exception: {exc}")
                    continue

                worker_node_id = result.node.idx
                print(f">> Got RESULT from worker node {worker_node_id}.")

                if result.params is not None:
                    self.state.worker_params[worker_node_id] = result.params

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
                    print(f"\t-> Submitted new job for worker {worker_node_id}.")
                else:
                    print(f"\t-> No more jobs for worker {worker_node_id}.")

                # Evaluate loss for this worker's data
                worker_loader = self.test_dataset.train_data(worker_node_id)
                test_loss = self.evaluation_function(self.module, worker_loader)

                self.round_logs.append({
                    'worker_idx': worker_node_id,
                    'test_worker_dataset_size': None if worker_loader is None else len(worker_loader),
                    'round': self.state.worker_rounds[worker_node_id],
                    'test_loss': test_loss,
                    'time': datetime.datetime.now()
                })

            rounds_completed += 1
            self.module.set_params(result.params)
        
        return self.module, self.round_logs

    def _dispatch_worker_job(self, worker_node: Node) -> Future:
        """
        Dispatches a worker job to the specified worker node.

        Args:
            worker_node (Node): The worker node to dispatch the job to.

        Returns:
            The future representing the worker job.
        """

        worker_dataset = self.dataset.train_data(worker_node.idx)
        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.module,
            data=worker_dataset,
            params=self.state.global_params,
            node=worker_node,
        )
        
        future = self.runtime.submit(worker_job, args)
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
            worker_dataset = self.dataset.train_data(node_id)
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

    # def _get_dataset_for_worker(self, worker_id: int) -> Subset:
    #     """
    #     Returns a subset of the dataset for the specified worker ID.

    #     Args:
    #         worker_id (int): The ID of the worker.

    #     Returns:
    #         The subset of the dataset for the specified worker.
    #     """
    #     all_workers = list(self.topology.workers)
    #     worker_indices = list(range(len(self.dataset)))
        
    #     try:
    #         worker_idx = [n.idx for n in all_workers].index(worker_id)
    #     except ValueError:
    #         raise ValueError(f"Worker ID {worker_id} not found in topology workers.")
        
    #     num_workers = len(all_workers)
    #     indices_per_worker = len(worker_indices) // num_workers
    #     start = worker_idx * indices_per_worker
    #     end = start + indices_per_worker
    #     if worker_idx == num_workers - 1:
    #         end = len(worker_indices)
        
    #     return Subset(self.dataset, worker_indices[start:end])
