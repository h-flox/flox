import csv

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flight.asynchronous.workflow import AsyncWorkflow
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from flight.strategies.strategy import DefaultStrategy
from flight.system.utils import flat_topology
from flight.utils.fed_data import federated_split

# Parameters
NUM_WORKERS = 5
NUM_LABELS = 10
NUM_GLOBAL_ROUNDS = 20

# Load MNIST test data (as per user request)
data = MNIST(
    root=".",
    download=True,
    train=False,
    transform=ToTensor(),
)

# Convert MNIST to TensorDataset for AsyncWorkflow
images_list = []
labels_list = []
for img, label in data:
    images_list.append(img)
    labels_list.append(label)
images = torch.stack(images_list)
labels = torch.tensor(labels_list)
tensor_dataset = torch.utils.data.TensorDataset(images, labels)

# Create federated topology
topo = flat_topology(NUM_WORKERS)

# Splitting data among workers using Federated split
fed_data = federated_split(
    topo=topo,
    data=data,
    num_labels=NUM_LABELS,
    label_alpha=1e8,  
    sample_alpha=1e8,  
)


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
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.configure_criterion()(logits, y)
        return loss


# Setup runtime and strategy
runtime = Runtime.simple_setup(max_workers=NUM_WORKERS)

# Run asynchronous workflow
wf = AsyncWorkflow(
    runtime=runtime,
    topology=topo,
    num_global_rounds=NUM_GLOBAL_ROUNDS,
    module=SimpleMNISTModule(),
    dataset=tensor_dataset,
    strategy=DefaultStrategy(),
)


def main():

    # Start the workflow and get the worker losses
    _, _, worker_losses = wf.start()

    # Save the worker losses to a CSV file
    with open("async_mnist_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["worker_id", "epoch", "loss", "time"])

        # Try to get per-worker, per-epoch times if available
        # If not, record None as a placeholder
        for worker_id, losses in worker_losses.items():
            # Try to get times from wf.worker_time_tracker.job_times if available
            start_time = wf.worker_time_tracker.job_times.get(worker_id, [])[0][0]
            times = wf.worker_time_tracker.job_times.get(worker_id, [])
            for epoch, loss in enumerate(losses):
                # Use the end time for the epoch if available, else None
                time_val = times[epoch][1] - start_time if epoch < len(times) else None
                writer.writerow([worker_id, epoch, loss, time_val])

    # Plot the worker losses
    df = pd.read_csv("async_mnist_results.csv")

    # Plot the worker losses Vs Rounds on MNIST
    for worker_id in df["worker_id"].unique():
        worker_data = df[df["worker_id"] == worker_id]
        plt.plot(
            worker_data["epoch"],
            worker_data["loss"],
            marker=".",
            label=f"Worker {worker_id}",
        )

    plt.title("Worker Loss Vs Rounds on MNIST")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend(title="Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("MNIST_Loss_Rounds_Simulation.pdf")
    plt.close()

    # Plot the Worker Loss Vs Time on MNIST
    for worker_id in df["worker_id"].unique():
        worker_data = df[df["worker_id"] == worker_id]
        plt.plot(
            worker_data["time"],
            worker_data["loss"],
            marker=".",
            label=f"Worker {worker_id}",
        )
    
    plt.title("Worker Loss Vs Time on MNIST")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend(title="Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("MNIST_Loss_Time_Simulation.pdf")
    plt.close()

if __name__ == "__main__":
    main()
