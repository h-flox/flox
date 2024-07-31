from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.jobs.types import Result, TrainJobArgs


# TODO: Test the hell out of this function.
def default_training_job(args: TrainJobArgs) -> Result:
    from datetime import datetime

    from torch.utils.data import DataLoader

    from flight.learning.trainers.torch import TorchTrainer

    hparams = args.trainer_strategy.trainer_hparams()

    training_start = datetime.now()

    state = args.worker_strategy.start_work()

    data = {
        "train": args.data.load(args.node, "train"),
        "valid": args.data.load(args.node, "valid"),
    }

    train_dataloader = DataLoader(
        data["train"],
        **{key: val for (key, val) in hparams if key.startswith("dataloader.train.")},
    )

    trainer = TorchTrainer(args.trainer_strategy)
    local_model = args.model.copy()
    optimizer = args.model.configure_optimizers()
    trainer.fit(
        args.node_state,
        local_model,
        optimizer,
        train_dataloader,
        **{key: val for (key, val) in hparams if key.startswith("trainer.")},
    )

    state = args.worker_strategy.end_work()

    training_end = datetime.now()

    history = {
        "node_idx": args.node.idx,
        "node_kind": args.node.kind,
        "parent_idx": args.parent.idx,
        "parent_kind": args.parent.kind,
        "training_start": training_start,
        "training_end": training_end,
    }

    return Result(
        node=...,
        node_state=...,
        params=...,
        records=...,
        extra=...,
    )
