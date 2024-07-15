import typing as t

from flight.federation.topologies.node import Node
from flight.learning.datasets.core import DataLoadable
from flight.learning.module import FlightModule

if t.TYPE_CHECKING:
    from flox.strategies import TrainerStrategy, WorkerStrategy


# TODO: Test the hell out of this function.
def default_training_job(
    node: Node,
    parent: Node,
    model: FlightModule,
    data: DataLoadable,
    worker_strategy: WorkerStrategy,
    trainer_strategy: TrainerStrategy,
):
    from datetime import datetime

    from torch.utils.data import DataLoader

    hparams = trainer_strategy.trainer_hparams()

    training_start = datetime.now()

    state = worker_strategy.work_start()

    data = {
        "train": data.load(node, "train"),
        "valid": data.load(node, "valid"),
    }

    train_dataloader = DataLoader(
        data["train"],
        **{key: val for (key, val) in hparams if key.startswith("dataloader.train.")},
    )

    trainer = Trainer(trainer_strategy)
    trainer.fit(
        local_model,
        optimizer,
        train_dataloader,
        node_state,
        **{key: val for (key, val) in hparams if key.startswith("trainer.")},
    )

    state = worker_strategy.work_end()

    training_end = datetime.now()

    history = {
        "node_idx": node.idx,
        "node_kind": node.kind,
        "parent_idx": parent.idx,
        "parent_kind": parent.kind,
        "training_start": training_start,
        "training_end": training_end,
    }
