import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
import seaborn as sns

from flight.asynchronous.workflow import AsyncWorkflow
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from flight.strategies.strategy import DefaultStrategy
from flight.system.utils import flat_topology
from flight.utils.fed_data import federated_split

# Define simple model (2 linear layers with ReLU)
class SimpleMNISTModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, NUM_LABELS),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.configure_criterion()(logits, y)
        return loss

if __name__ == "__main__":
    # Parameters
    NUM_LABELS = 10
    NUM_WORKERS = 5
    NUM_GLOBAL_ROUNDS = 10

    # Load MNIST training data
    def load_data(dataset, train):
        if dataset == "MNIST":
            # Download MNIST dataset if not already present
            return MNIST(
                root=".",
                train=train,
                transform=ToTensor(),
                download=True,
            )
        elif dataset == "FashionMNIST":
            # Download FashionMNIST dataset if not already present
            return FashionMNIST(
                root=".",
                train=train,
                transform=ToTensor(),
                download=True,
            )

    # Create federated topology
    topo = flat_topology(NUM_WORKERS)

    # Splitting data among workers using Federated split
    train_data = load_data("FashionMNIST", train=True)
    test_data = load_data("FashionMNIST", train=False)

    fed_data = federated_split(
        topo=topo,
        data=train_data,
        num_labels=NUM_LABELS,
        label_alpha=1.0,
        sample_alpha=1.0,
    )

    # Setup runtime and strategy
    runtime = Runtime.simple_setup(max_workers=NUM_WORKERS)

    # Run asynchronous workflow
    wf = AsyncWorkflow(
        runtime=runtime,
        topology=topo,
        num_global_rounds=NUM_GLOBAL_ROUNDS,
        module=SimpleMNISTModule(),
        dataset=TensorDataset(train_data.data.view(-1, 28 * 28).float(), train_data.targets),
        strategy=DefaultStrategy(),
    )

    # Start the workflow
    final_model, round_logs = wf.start()
    df = pd.DataFrame.from_records(round_logs)
    
    # df["sample_alpha"] = 1.0
    df["timedelta"] = df.time - df.time.min()
    df.to_csv("async_simulation/async_mnist_simulation_logs.csv", index=False)

    # Using .csv to do the plotting
    df = pd.read_csv("async_simulation/async_mnist_simulation_logs.csv")
    df['timedelta'] = pd.to_timedelta(df['timedelta']).dt.total_seconds()

    #test_loader = DataLoader(test_data)
    #test_loss = evaluate_fn(final_model, test_loader)

    # Plotting of Loss vs Time
    plt.figure(figsize=(10, 6))
    # Training Loss
    for worker_idx, group in df.groupby('worker_idx'):
        plt.plot(group['timedelta'], group['loss'], marker='o', label=f'Worker {worker_idx}')
    # Test Loss
    #plt.axhline(y=test_loss, color='black', linestyle='--', label='Test Loss')
    plt.title("Worker Loss vs Time")
    plt.xlabel("Time (seconds)")
    plt.grid(True)
    plt.legend(title="Worker")
    plt.savefig("async_simulation/MNIST_Loss_Time.pdf")
    plt.show()
    plt.close()

    # Plotting of Loss vs Rounds
    plt.figure(figsize=(10, 6))
    # Training Loss
    for worker_idx, group in df.groupby('worker_idx'):
        plt.plot(group['round'], group['loss'], marker='o', label=f'Worker {worker_idx}')
    # Test Loss
    #plt.axhline(y=test_loss, color='black', linestyle='--', label='Test Loss')
    plt.title("Worker Loss vs Rounds")
    plt.xlabel("Rounds")
    plt.grid(True)
    plt.legend(title="Worker")
    plt.savefig("async_simulation/MNIST_Loss_Round.pdf")
    plt.show()
    plt.close()
    
    df['duration'] = df.groupby('worker_idx')['timedelta'].shift(-1) - df['timedelta']
    plt.figure(figsize=(9, 5))

    # Use job index for color mapping
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
    plt.savefig("async_simulation/MNIST_jobs.pdf")
    plt.show()
    plt.close()


