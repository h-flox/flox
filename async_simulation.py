import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from flight.asynchronous.workflow import AsyncWorkflow
from flight.jobs.protocols import Result
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from flight.strategies import DefaultStrategy
from flight.system.utils import flat_topology
import pandas as pd

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

    node = args.node
    params = model.get_params()

    return Result(node=node, params=params, module=model)

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
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 2)

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
        return torch.optim.SGD(self.parameters(), lr=0.005)

    def training_step(self, batch):
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

    # Parameters
    num_workers = 5
    num_global_rounds = 20
    module = SimpleModule()
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    topology = flat_topology(num_workers)
    runtime = Runtime.simple_setup(max_workers=num_workers)

    # Create the workflow
    wf = AsyncWorkflow(
        runtime=runtime,
        topology=topology,
        num_global_rounds=num_global_rounds,
        module=module,
        dataset=dataset,
        strategy=DefaultStrategy(),
    )

    # Start the workflow
    final_model, round_logs = wf.start()
    df = pd.DataFrame.from_records(round_logs)
    df["timedelta"] = df.time - df.time.min()
    df.to_csv("async_simulation/async_simulation_logs.csv", index=False)

    # Reading form .csv file
    df = pd.read_csv("async_simulation/async_simulation_logs.csv")
    df['timedelta'] = pd.to_timedelta(df['timedelta']).dt.total_seconds()

    # Plotting of Loss vs Time
    plt.figure(figsize=(9, 5))
    for worker_idx, group in df.groupby('worker_idx'):
        plt.plot(group['timedelta'], group['loss'], marker='o', label=f'Worker {worker_idx}')
    plt.title("Worker Loss vs Time")
    plt.xlabel("Time (seconds)")
    plt.grid(True)
    plt.legend(title="Worker")
    plt.savefig("async_simulation/async_Loss_Time.pdf")
    plt.show()
    plt.close()

    # Plotting of Loss vs Rounds
    plt.figure(figsize=(9, 5))
    for worker_idx, group in df.groupby('worker_idx'):
        plt.plot(group['round'], group['loss'], marker='o', label=f'Worker {worker_idx}')
    plt.title("Worker Loss vs Time")
    plt.xlabel("Rounds")
    plt.grid(True)
    plt.legend(title="Worker")
    plt.savefig("async_simulation/async_Loss_Round.pdf")
    plt.show()
    plt.close()

    # Scheduled job execution plot 
    # Ensure 'timedelta' is sorted for each worker
    df = df.sort_values(['worker_idx', 'timedelta'])
    df['duration'] = df.groupby('worker_idx')['timedelta'].shift(-1) - df['timedelta']
    plt.figure(figsize=(9, 5))

    # Use job index (e.g., 'round') for color mapping
    unique_jobs = sorted(df['round'].unique())
    color_map = {job_idx: color for job_idx, color in zip(unique_jobs, sns.color_palette("tab20", n_colors=len(unique_jobs)))}

    for i, worker_idx in enumerate(sorted(df['worker_idx'].unique())):
        worker_jobs = df[df['worker_idx'] == worker_idx]
        for _, job in worker_jobs.iterrows():
            plt.barh(
                y=worker_idx,
                width=job['duration'],
                left=job['timedelta'],
                color=color_map[job['round']],
                edgecolor='black'
            )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Worker Index")
    plt.title("Asynchronous Worker Job Completion Times")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("async_simulation/async_jobs.pdf")
    plt.show()
    plt.close()