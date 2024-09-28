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
        Result of local training job completed by a worker node.
    """

    import datetime

    from flight.federation.jobs.types import Result

    ####################################################################################

    node = args.node
    parent = args.parent
    node_state = args.worker_strategy.start_work(args.node_state)
    local_model = args.model
    data = args.data
    worker_strategy = args.worker_strategy
    trainer_strategy = args.trainer_strategy

    ####################################################################################

    worker_strategy.start_work(node_state)

    training_start = datetime.datetime.now()

    match local_model.kind():
        case "lightning":
            raise ValueError

        case "scikit":
            from flight.learning.scikit import (
                ScikitDataModule,
                ScikitModule,
                ScikitTrainer,
            )

            assert isinstance(local_model, ScikitModule)
            assert isinstance(data, ScikitDataModule)

            trainer_init_params = dict()  # TODO: Add this as an attr. of TrainArgJobs.
            trainer = ScikitTrainer(node, **trainer_init_params)
            records = trainer.fit(local_model, data)

        case "torch":
            from flight.learning.torch import TorchDataModule, TorchModule, TorchTrainer

            assert isinstance(local_model, TorchModule)
            assert isinstance(data, TorchDataModule)

            # TODO: Add this as an attr. of TrainArgJobs.
            trainer_init_params = dict(progress_bar=False)
            trainer_fit_params = dict()
            trainer = TorchTrainer(**trainer_init_params)
            records = trainer.fit(node_state, local_model, data, **trainer_fit_params)

        case _:
            raise ValueError(
                f"Illegal literal string returned by {local_model.kind()=}. "
                f"Default implementations of this model are given by the framework-"
                f"specific module classes provided by Flight. Users should NOT "
                f"override this. If you have, then please remove the implementation "
                f"of this method."
            )

    # worker_strategy.before_training(node_state, data)
    # TODO: These needed calls (↑↑ and ↓↓) to be included in the trainers!!
    # state, optimizer = worker_strategy.after_training(node_state)

    training_end = datetime.datetime.now()

    ####################################################################################

    history = {
        "node_idx": node.idx,
        "node_kind": node.kind,
        "parent_idx": parent.idx,
        "parent_kind": parent.kind,
        "training_start": training_start,
        "training_end": training_end,
    }

    ####################################################################################

    return Result(
        # should there be a from/to type of dynamic here?
        node=node,
        node_state=node_state,
        params=local_model.get_params(),
        records=records,
        extra={},
    )
