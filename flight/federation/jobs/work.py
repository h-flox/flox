from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.jobs.types import Result, TrainJobArgs


# TODO: Test the hell out of this function.
def default_training_job(args: TrainJobArgs) -> Result:
    """
    Default implementation of a local training job that is run on worker nodes in a
    federation.

    Args:
        args (TrainJobArgs):

    Returns:

    """
    import copy

    from datetime import datetime
    from flight.learning.trainers.torch import TorchTrainer
    from flight.federation.jobs.types import Result

    hparams = args.trainer_strategy.hparams()

    training_start = datetime.now()

    node = args.node
    node_state = args.worker_strategy.start_work(args.node_state)
    trainer = TorchTrainer(
        node,
        args.trainer_strategy,
        max_epochs=3,
        progress_bar=False,
    )
    local_model = copy.deepcopy(args.model)

    args.worker_strategy.before_training(args.node_state, args.data)  # TODO: Reconsider

    records = trainer.fit(
        args.node_state,
        local_model,
        args.data,
    )

    # result = args.worker_strategy.end_work()  # TODO: re-include

    training_end = datetime.now()

    history = {
        "node_idx": args.node.idx,
        "node_kind": args.node.kind,
        "parent_idx": args.parent.idx,
        "parent_kind": args.parent.kind,
        "training_start": training_start,
        "training_end": training_end,
    }

    result = Result(
        node=args.node,
        node_state=node_state,
        params=local_model.get_params(),
        records=records,
        extra={},
    )
    return result
