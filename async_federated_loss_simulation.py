import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import time

from flight.runtime import Runtime
from flight.system import flat_topology
from flight.learning.module import TorchModule
from flight.strategies.strategy import DefaultStrategy
from flight.asynchronous.workflow import AsyncStrategy


@dataclass
class Config:
    num_workers: int = 5
    num_rounds: int = 20
    dataset_size: int = 1000
    input_features: int = 10
    hidden_size: int = 64
    seed: int = 42


class SimpleModel(TorchModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x): return self.net(x)
    def configure_criterion(self): return nn.CrossEntropyLoss()
    def configure_optimizers(self): return optim.SGD(self.parameters(), lr=0.1)


class LossTracker:
    def __init__(self):
        self.global_losses, self.rounds, self.times = [], [], []
    
    def add_loss(self, round_num, loss, time_taken=0):
        self.global_losses.append(loss)
        self.rounds.append(round_num)
        self.times.append(time_taken)


def generate_data(config):
    torch.manual_seed(config.seed)
    X = torch.randn(config.dataset_size, config.input_features)
    y = (torch.sum(X[:, :3], dim=1) > 0).long()
    return TensorDataset(X, y)


def calculate_loss(model, dataset):
    model.eval()
    criterion = model.configure_criterion()
    dataloader = DataLoader(dataset, batch_size=32)
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    return total_loss / len(dataloader)


def create_aggregation_policy(loss_tracker):
    """Create an aggregation policy with access to the loss tracker."""
    def policy(async_strategy, worker_id):
        start_time = time.time()
        async_strategy.partial_aggregation_policy(last_updated_node=worker_id)
        agg_time = time.time() - start_time
        
        global_loss = calculate_loss(async_strategy.module, async_strategy.dataset)
        round_num = async_strategy.state.completed_worker_jobs
        loss_tracker.add_loss(round_num, global_loss, agg_time)
        print(f"Round {round_num}: Loss = {global_loss:.4f}, Time = {agg_time:.4f}s")
    
    return policy


def plot_results(loss_tracker, config):
    """Plot only the Global Loss vs Training Rounds graph."""
    plt.figure(figsize=(6, 4))
    
    # Global Loss vs Rounds (Main loss function graph)
    plt.plot(loss_tracker.rounds, loss_tracker.global_losses, 'b-', linewidth=3, marker='o', markersize=8)
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Global Loss', fontsize=14)
    plt.title('Global Loss Function vs Training Rounds\n(Partial Aggregation Policy)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add statistics as text
    if len(loss_tracker.global_losses) > 0:
        initial_loss = loss_tracker.global_losses[0]
        final_loss = loss_tracker.global_losses[-1]
        loss_reduction = initial_loss - final_loss
        
        stats_text = f'Initial Loss: {initial_loss:.4f}\nFinal Loss: {final_loss:.4f}\nLoss Reduction: {loss_reduction:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('global_loss_vs_rounds.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Global Loss vs Training Rounds graph saved as 'global_loss_vs_rounds.png'")
    
    # Show the plot
    plt.show()


def main():
    """Main function to run the simulation."""
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print("="*60)
    print("ASYNC FEDERATED LEARNING LOSS SIMULATION")
    print("="*60)
    print(f"Workers: {config.num_workers}")
    print(f"Rounds: {config.num_rounds}")
    print(f"Dataset Size: {config.dataset_size}")
    print(f"Input Features: {config.input_features}")
    print("-"*60)
    
    # Setup
    runtime = Runtime.simple_setup(max_workers=4, exec_kind="thread")
    topology = flat_topology(config.num_workers)
    model = SimpleModel(config.input_features, config.hidden_size, 2)
    dataset = generate_data(config)
    loss_tracker = LossTracker()
    
    # Create async strategy with custom aggregation policy
    async_strategy = AsyncStrategy(
        runtime=runtime, 
        topology=topology, 
        num_global_rounds=config.num_rounds,
        module=model, 
        dataset=dataset, 
        strategy=DefaultStrategy(),
        aggregation_policy=create_aggregation_policy(loss_tracker)
    )
    
    # Run simulation
    print("Starting async federated learning simulation...")
    start_time = time.time()
    
    try:
        final_model, _ = async_strategy.start()
        total_time = time.time() - start_time
        
        # Results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Rounds: {len(loss_tracker.rounds)}")
        if len(loss_tracker.global_losses) > 0:
            print(f"Initial Loss: {loss_tracker.global_losses[0]:.4f}")
            print(f"Final Loss: {loss_tracker.global_losses[-1]:.4f}")
            print(f"Loss Reduction: {loss_tracker.global_losses[0] - loss_tracker.global_losses[-1]:.4f}")
            print(f"Average Aggregation Time: {np.mean(loss_tracker.times):.4f} seconds")
        
        # Plot results
        if len(loss_tracker.global_losses) > 0:
            plot_results(loss_tracker, config)
        else:
            print("No loss data collected. Check if aggregation policy is working.")
            
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if hasattr(runtime.control_plane, 'shutdown'):
            runtime.control_plane.shutdown()


if __name__ == "__main__":
    main() 