from __future__ import annotations

from flight.federation.jobs import TrainJobArgs
from flight.federation.topologies import WorkerState
from flight.learning import AbstractDataModule, AbstractModule
from flight.learning.torch import TorchModule, TorchDataModule, TorchTrainer
from flight.types import Record


def _validate_torch_types(
    model: TorchModule | AbstractModule,
    data: TorchDataModule | AbstractDataModule,
) -> tuple[TorchModule, TorchDataModule]:
    """
    Validates the types of the arguments.

    Specifically, it ensures that all arguments are appropriate for deep learning with
    PyTorch on the backend.

    Throws:
        - `ValueError`: If there is an illegal type for any of the arguments.

    Returns:
        Tuple containing the following items:
            - `TorchModule`: ...
            - `TorchDataModule`: ...
    """
    if not isinstance(model, TorchModule):
        raise ValueError("Local model is not an instance of `TorchModule`.")

    if not isinstance(data, TorchDataModule):
        raise ValueError("Data is not an instance of `TorchDataModule`.")

    return model, data


def torch_local_train(
    args: TrainJobArgs,
    data: TorchDataModule | AbstractDataModule,
    local_model: TorchModule | AbstractModule,
    node_state: WorkerState,
) -> list[Record]:
    local_model, data = _validate_torch_types(local_model, data)

    # TODO: Add this as an attr. of TrainArgJobs.
    trainer_init_params = dict(progress_bar=False)
    trainer = TorchTrainer(node=args.node, **trainer_init_params)

    trainer_fit_params = dict()
    records = trainer.fit(
        node_state,
        local_model,
        data,
        **trainer_fit_params,
    )

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
