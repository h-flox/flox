from __future__ import annotations

import typing as t
import datetime

if t.TYPE_CHECKING:
    from v1.flight.learning import AbstractDataModule, AbstractModule
    from v1.flight.learning.torch import TorchModule
    from v1.flight.topologies import Node
    from v1.flight.types import Record


def torch_local_train(
    data: AbstractDataModule,
    local_model: TorchModule,
    node: Node,
) -> list[Record]:
    """
    Performs local training on a PyTorch model using the provided data.
    
    Args:
        data: The data module containing training data
        local_model: The PyTorch model to train
        node: The worker node performing the training
        
    Returns:
        List of training records containing metrics and timing information
    """
    import torch
    from torch.utils.data import DataLoader
    
    assert isinstance(local_model, TorchModule)
    
    # Get training data for this node
    train_loader = data.train_data(node)
    
    # Configure optimizer and loss function
    optimizer = local_model.configure_optimizers()
    criterion = local_model.configure_criterion()
    
    # Training parameters
    num_epochs = 3  # Default number of epochs
    device = torch.device("cpu")  # Default to CPU
    
    # Move model to device
    local_model.to(device)
    local_model.train()
    
    records = []
    training_start = datetime.datetime.now()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(device) if torch.is_tensor(batch) else batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = local_model.training_step(batch, batch_idx)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Create record for this batch
            current_time = datetime.datetime.now()
            record = {
                "node/idx": node.idx,
                "node/kind": node.kind,
                "train/epoch": epoch,
                "train/batch": batch_idx,
                "train/loss": loss.item(),
                "train/time": current_time,
                "train/timedelta": (current_time - training_start).total_seconds(),
            }
            records.append(record)
        
        # Record epoch summary
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        current_time = datetime.datetime.now()
        epoch_record = {
            "node/idx": node.idx,
            "node/kind": node.kind,
            "train/epoch": epoch,
            "train/epoch_loss": avg_epoch_loss,
            "train/time": current_time,
            "train/timedelta": (current_time - training_start).total_seconds(),
        }
        records.append(epoch_record)
    
    return records


def torch_local_test(
    args: TrainJobArgs,
    data: TorchDataModule | AbstractDataModule,
    local_model: TorchModule | AbstractModule,
    node_state: WorkerState,
) -> list[Record]:
    local_model, data = _validate_torch_types(local_model, data)
    trainer_init_params = dict(progress_bar=False)
    trainer = TorchTrainer(node=args.node, **trainer_init_params)
    records = trainer.test(node_state, local_model, data)
    return records


def torch_local_validate(
    args: TrainJobArgs,
    data: TorchDataModule | AbstractDataModule,
    local_model: TorchModule | AbstractModule,
    node_state: WorkerState,
):
    local_model, data = _validate_torch_types(local_model, data)
    trainer_init_params = dict(progress_bar=False)
    trainer = TorchTrainer(node=args.node, **trainer_init_params)
    records = trainer.validate(...)
    return records
