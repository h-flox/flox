import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.utils.data import Subset
### Define simple model (2 linear layers with ReLU activation)

class SimpleMNISTModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.configure_criterion()(logits, y)
        return loss

    def configure_criterion(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


### CNN Architecture

# class SimpleMNISTModule(TorchModule):
#     def __init__(self):
#         super().__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
#         # Dropout for regularization
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout(0.5)
        
#         # Fully connected layers
#         # After 2 max pools: 28x28 -> 14x14 -> 7x7
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         # Reshape flattened input back to image format
#         # From [batch_size, 784] to [batch_size, 1, 28, 28]
#         if len(x.shape) == 2:
#             x = x.view(x.size(0), 1, 28, 28)
        
#         # First conv block
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
        
#         # Second conv block  
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
        
#         # Flatten and fully connected layers
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
        
#         return x

#     def training_step(self, batch):
#         x, y = batch
#         logits = self(x)
#         loss = self.configure_criterion()(logits, y)
#         return loss

#     def configure_criterion(self):
#         return nn.CrossEntropyLoss()

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.001)

def evaluate_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

if __name__ == "__main__":
    # Parameters
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
                download=False,
            )
        
        elif dataset == "FashionMNIST":
            # Download FashionMNIST dataset if not already present
            return FashionMNIST(
                root=".",
                train=train,
                transform=ToTensor(),
                download=False,
            )

    # Create federated topology
    topo = flat_topology(NUM_WORKERS)

    # Splitting data among workers using Federated split
    train_data = load_data("MNIST", train=True)
    train_data = Subset(train_data, range(0, 60000))  # Limit to 2000 samples for faster testing

    test_data = load_data("MNIST", train=False)
    test_data = Subset(test_data, range(0, 5000))  # Limit to 300 samples for faster testing

    for s in [1.0, 100.0]: # [1.0, 3.0, 30.0]:
        for l in [1.0, 100.0]: # [0.1, 1.0, 10.0]:
            train_fed_data = federated_split(
                topo=topo,
                data=train_data,
                num_labels=10,
                label_alpha=l,
                sample_alpha=s,
                rng=42,
                # train_test_valid_split=(0.8, 0.1, 0.1),
            )
            test_fed_data = federated_split(
                topo=topo,
                data=test_data,
                num_labels=10,
                label_alpha=l,
                sample_alpha=s,
                rng=42,
                # train_test_valid_split=(0.8, 0.1, 0.1),
            )

            # print(fed_data.train_data(0))
            # exit(0)

            # Setup runtime and strategy
            runtime = Runtime.simple_setup(max_workers=NUM_WORKERS)

            # Run asynchronous workflow
            wf = AsyncWorkflow(
                runtime=runtime,
                topology=topo,
                num_global_rounds=NUM_GLOBAL_ROUNDS,
                module=SimpleMNISTModule(),
                dataset=train_fed_data,
                strategy=DefaultStrategy(),
                test_dataset=test_fed_data,
            )

            # Start the workflow
            final_model, round_logs = wf.start()
            df = pd.DataFrame.from_records(round_logs)

            # Prepare test data loader
            test_loader = DataLoader(
                test_data,
                batch_size=128,
                shuffle=False
            )

            test_accuracies = []
            
            # for i, params in enumerate(wf.global_params_history):
            # wf.module.set_params(params)
            acc = evaluate_accuracy(wf.module, test_loader)
            test_accuracies.append(acc)

            results_df = pd.DataFrame({
                'sample_alpha': s,
                'label_alpha': l,
                'test_accuracy': test_accuracies
            })

            results_df = pd.concat([df, results_df], axis=1)

            # Add timedelta column (seconds since first round)
            results_df['timedelta'] = pd.to_datetime(results_df['time']) - pd.to_datetime(results_df['time']).min()
            results_df['timedelta'] = results_df['timedelta'].dt.total_seconds()

            results_df.to_csv(f"comparison/csv/async/new/test_accuracy_L{l}_S{s}.parquet", index=False)
