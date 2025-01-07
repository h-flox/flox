from __future__ import annotations

from flight.federation.jobs import TrainJobArgs
from flight.federation.topologies import WorkerState
from flight.learning import AbstractDataModule, AbstractModule
from flight.learning.torch import TorchModule, TorchDataModule, TorchTrainer
from flight.types import Record


def torch_local_train(
    args: TrainJobArgs,
    data: TorchDataModule | AbstractDataModule,
    local_model: TorchModule | AbstractModule,
    node_state: WorkerState,
) -> list[Record]:
    if not isinstance(local_model, TorchModule):
        raise ValueError("Local model is not an instance of `TorchModule`.")

    if not isinstance(data, TorchDataModule):
        raise ValueError("Data is not an instance of `TorchDataModule`.")

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
