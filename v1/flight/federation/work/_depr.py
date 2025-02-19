from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from v1.flight.federation.types import Result, TrainJobArgs


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

    # import datetime

    from v1.flight.federation.types import Result

    ####################################################################################

    node = args.node
    # parent = args.parent
    node_state = args.worker_strategy.start_work(args.node_state)
    local_model = args.model
    data = args.data
    worker_strategy = args.worker_strategy
    # trainer_strategy = args.trainer_strategy

    ####################################################################################

    worker_strategy.start_work(node_state)

    # training_start = datetime.datetime.now()

    match local_model.kind():
        case "lightning":
            raise ValueError

        case "scikit":
            from v1.flight.federation.work._scikit import scikit_local_train

            records = scikit_local_train(data, local_model, node)

        case "torch":
            pass

            # records = torch_local_train(args, data, local_model, node_state)

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

    # training_end = datetime.datetime.now()

    ####################################################################################

    # history = {
    #     "node_idx": node.idx,
    #     "node_kind": node.kind,
    #     "parent_idx": parent.idx,
    #     "parent_kind": parent.kind,
    #     "training_start": training_start,
    #     "training_end": training_end,
    # }

    ####################################################################################

    return Result(
        # should there be a from/to type of dynamic here?
        node=node,
        node_state=node_state,
        params=local_model.get_params(),
        module=local_model,
        records=records,
        extra={},
    )


def ignite_training_job(args: TrainJobArgs) -> Result:
    from ignite.engine import Engine

    from v1.flight.learning.torch import TorchModule

    assert isinstance(args.model, TorchModule)

    model: TorchModule = args.model

    optimizer = args.ignite_config.optimizer_cls(
        model.parameters(), **args.ignite_config.optimizer_args
    )
    loss_fn = args.ignite_config.loss_fn

    train_step = args.train_step

    if train_step is None:
        if not args.supervised:
            raise ValueError(
                "Unsupervised training requires that users "
                "provide a custom `train_step`."
            )

        from ignite.engine import supervised_training_step

        train_step = supervised_training_step(
            # args without defaults
            model,
            optimizer,
            loss_fn,
            # args with defaults
            device=args.ignite_config.device,
            non_blocking=args.ignite_config.non_blocking,
            prepare_batch=args.ignite_config.prepare_batch,
            model_transform=args.ignite_config.model_transform,
            output_transform=args.ignite_config.output_transform,
            gradient_accumulation_steps=args.ignite_config.gradient_accumulation_steps,
            model_fn=args.ignite_config.model_fn,
        )

    trainer = Engine(args.train_step)

    """

    if there is validation_data:
        if users do not provide `valid_step`:
            valid_step = make_supervised_step_from_default(...)

        validator = Engine(valid_step)

        @trainer.on(Events.EPOCH_COMPLETED):
        def log_validation_results(validator):
            ...
    """

    trainer.run(args.data.train_data(), max_epochs=5)

    # if args.valid_step:
    #     validator = Engine(args.valid_step)
    #     valid_loader = ...
    #
    #     @trainer.on(Events.EPOCH_COMPLETED)
    #     def log_training_results(trainer):
    #         args.valid_step()

    # @trainer.on(Events.)
