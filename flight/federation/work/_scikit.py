from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node
    from flight.learning import AbstractDataModule, AbstractModule
    from flight.learning.scikit import ScikitDataModule, ScikitModule
    from flight.types import Record


def scikit_local_train(
    data: ScikitDataModule | AbstractDataModule,
    local_model: ScikitModule | AbstractModule,
    node: Node,
) -> list[Record]:
    from flight.learning.scikit import ScikitDataModule, ScikitModule, ScikitTrainer

    assert isinstance(local_model, ScikitModule)
    assert isinstance(data, ScikitDataModule)

    trainer_init_params = dict()  # TODO: Add this as an attr. of TrainArgJobs.
    trainer = ScikitTrainer(node, **trainer_init_params)
    records = trainer.fit(local_model, data)
    return records
