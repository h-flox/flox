import typing as t


if t.TYPE_CHECKING:
    from flight.federation.jobs.types import Result, TrainJobArgs


# TODO: Test the hell out of this function.
def default_training_job(args: TrainJobArgs) -> Result:
    from datetime import datetime

    from torch.utils.data import DataLoader

    hparams = trainer_strategy.trainer_hparams()

    training_start = datetime.now()

    state = worker_strategy.start_work()

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

    state = worker_strategy.end_work()

    training_end = datetime.now()

    history = {
        "node_idx": node.idx,
        "node_kind": node.kind,
        "parent_idx": parent.idx,
        "parent_kind": parent.kind,
        "training_start": training_start,
        "training_end": training_end,
    }
