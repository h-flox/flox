import time
from concurrent.futures import as_completed

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from flight.asynchronous.workflow import (
    AsyncWorkflow,
    AsyncWorkflowEvents,
    WorkerTimeTracker,
)
from flight.jobs.protocols import Result
from flight.jobs.worker import WorkerJobArgs
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from flight.strategies import DefaultStrategy
from flight.system.utils import flat_topology


def simulated_worker_job(args):
    """
    Simulates a worker job.

    Args:
        args (WorkerJobArgs): The arguments for the worker job.

    Returns:
        Result: The result of the worker job. Whether it is a success or failure,
        the result is None. This is because the job is only used to pre-warm the worker.

    Raises:
        ValueError: If the data type is not supported.
    """
    start_time = time.time()

    if isinstance(args.data, DataLoader):
        loader = args.data

    elif isinstance(args.data, Dataset):
        loader = DataLoader(args.data, batch_size=32)

    else:
        raise ValueError("Unsupported data type for worker job.")

    model = args.model
    model.train()
    optimizer = model.configure_optimizers()
    criterion = model.configure_criterion()

    for batch in loader:
        optimizer.zero_grad()
        x, y = batch
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        break

    end_time = time.time()
    print(
        f"Worker job finished for node {args.node.idx if args.node else 'unknown'} "
        f"at {end_time}, duration: {end_time - start_time}"
    )
    node = args.node
    params = model.get_params()

    return Result(node=node, params=params, module=model)


class SimAsyncWorkflow(AsyncWorkflow):
    """
    Simulates an asynchronous workflow.

    Args:
        worker_node (Node): The worker node to dispatch the job to.

    Returns:
        Future: The future representing the worker job.
    """

    def _dispatch_worker_job(self, worker_node):
        """
        Dispatches a worker job to the specified worker node.

        Args:
            worker_node (Node): The worker node to dispatch the job to.

        Returns:
            Future: The future representing the worker job.
        """
        worker_dataset = self._get_dataset_for_worker(worker_node.idx)

        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.module,
            data=worker_dataset,
            params=self.state.global_params,
            node=worker_node,
        )

        tracker = self.worker_time_tracker

        if tracker is not None:
            tracker.record_job_start(worker_node.idx)
            fut = self.runtime.submit(simulated_worker_job, args)
            fut.add_done_callback(
                lambda _, idx=worker_node.idx, t=tracker: t.record_job_end(idx)
            )

            return fut

        else:

            return self.runtime.submit(simulated_worker_job, args)

    def start(self):
        """
        Starts the asynchronous workflow.
        """
        self.fire_event_handler(AsyncWorkflowEvents.STARTED)

        num_workers = len(self.topology.workers)
        total_jobs = num_workers * self.num_global_rounds
        jobs_queue = []
        worker_idx_map = {worker.idx: worker for worker in self.topology.workers}

        for round_idx in range(self.num_global_rounds):
            for worker in self.topology.workers:
                jobs_queue.append(worker.idx)

        active_futures = set()
        jobs_assigned = 0
        jobs_completed = 0
        worker_job_count = {worker.idx: 0 for worker in self.topology.workers}

        for worker in self.topology.workers:
            if jobs_queue:
                worker_idx = jobs_queue.pop(0)
                active_futures.add(
                    self._dispatch_worker_job(worker_idx_map[worker_idx])
                )
                worker_job_count[worker_idx] += 1
                jobs_assigned += 1

        while active_futures and jobs_completed < total_jobs:
            done_futures = []
            for future in as_completed(active_futures, timeout=None):
                done_futures.append(future)
                if len(done_futures) >= 1:
                    break
            for future in done_futures:
                active_futures.remove(future)
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Worker job failed with exception: {exc}")
                    continue
                worker_node_id = result.node.idx
                if result.params is not None:
                    self.state.worker_params[worker_node_id] = result.params
                self.fire_event_handler(
                    AsyncWorkflowEvents.WORKER_JOB_COMPLETED, {"result": result}
                )
                if self.aggregation_policy:
                    self.aggregation_policy(self, worker_node_id)
                else:
                    self.partial_aggregation_policy(last_updated_node=worker_node_id)
                self.state.completed_worker_jobs += 1
                jobs_completed += 1
                if jobs_queue:
                    next_worker_idx = jobs_queue.pop(0)
                    print(
                        f"Submitting job for worker {next_worker_idx} at {time.time()}"
                    )
                    active_futures.add(
                        self._dispatch_worker_job(worker_idx_map[next_worker_idx])
                    )
                    worker_job_count[next_worker_idx] += 1
                    jobs_assigned += 1
        self.fire_event_handler(AsyncWorkflowEvents.COMPLETED)
        self.module.set_params(self.state.global_params)

        return self.module, None


class SimpleModule(TorchModule):
    """
    A simple module for training a model.

    Args:
        TorchModule: The base class for the simple module.
    """

    def __init__(self):
        """
        Initializes the simple module.

        Args:
            None

        Returns:
            None
        """

        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        """
        Forward pass of the simple module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.linear(x)

    def configure_criterion(self, *args, **kwargs):
        """
        Configures the criterion for the simple module.

        Args:
            None

        Returns:
            torch.nn.CrossEntropyLoss: The criterion for the simple module.
        """
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self, *args, **kwargs):
        """
        Configures the optimizer for the simple module.

        Args:
            None

        Returns:
            torch.optim.SGD: The optimizer for the simple module.
        """
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        """
        Training step of the simple module.

        Args:
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        x, y = batch
        logits = self.linear(x)
        loss = self.configure_criterion()(logits, y)

        return loss


if __name__ == "__main__":
    num_workers = 5
    num_global_rounds = 8
    module = SimpleModule()
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    topology = flat_topology(num_workers)
    runtime = Runtime.simple_setup(max_workers=num_workers)
    tracker = WorkerTimeTracker(num_workers)

    wf = SimAsyncWorkflow(
        runtime=runtime,
        topology=topology,
        num_global_rounds=num_global_rounds,
        module=module,
        dataset=dataset,
        strategy=DefaultStrategy(),
        worker_time_tracker=tracker,
    )
    wf.start()

    if not any(tracker.job_times.values()):
        print("No job times recorded! Check if jobs are running and being tracked.")

    else:

        for worker in tracker.job_times:

            if len(tracker.job_times[worker]) > 1:
                tracker.job_times[worker] = tracker.job_times[worker][1:]

        min_time = min(
            start for times in tracker.job_times.values() for start, _ in times
        )

        for worker in tracker.job_times:
            tracker.job_times[worker] = [
                (s - min_time, e - min_time) for s, e in tracker.job_times[worker]
            ]

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        num_jobs = max(len(times) for times in tracker.job_times.values())
        job_colors = sns.color_palette("tab20", n_colors=num_jobs)

        for worker_idx, times in tracker.job_times.items():

            for job_num, (start, end) in enumerate(times):
                color = job_colors[job_num % len(job_colors)]

                ax.barh(
                    worker_idx,
                    end - start,
                    left=start,
                    height=0.8,
                    color=color,
                    edgecolor="black",
                )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Worker Index")
        ax.set_title("Asynchronous Worker Job Completion Times")
        plt.tight_layout()
        plt.savefig("Worker_Time_Simulation.pdf")
        plt.close()
