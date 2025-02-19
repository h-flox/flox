from __future__ import annotations

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from v1.flight.federation_v2.jobs.ignite import Engine
    from torch import nn, optim

    from v1.flight.federation.aggr import Result, TrainJobArgs
    from v1.flight.types import Record


if t.TYPE_CHECKING:
    ProcessFn: t.TypeAlias = t.Callable[[Engine, t.Any], t.Any]
    """
    As defined by Ignite (see [Engine][ignite.engine.Engine]), this is a function
    that receives a handle to the engine and the current batch in each iteration,
    and returns data to be stored in the engine's state.

    Simply put, this function defines how an `Engine` processes a batch of data during
    training, testing, and evaluation.
    """

    UnwrappedProcessFn: t.TypeAlias = t.Callable[
        [
            tuple[nn.Module, optim.Optimizer, nn.Module],
            Engine,
            t.Any,
        ],
        t.Any,
    ]
    """
    A function that has access to the model, optimizer, and loss function, and processes
    in a broader scope than a `ProcessFn`. This function type is used to wrap a
    `ProcessFn` with these dependencies in context.
    """


@dataclass
class TrainingJobState:
    ...


def training_job(args: TrainJobArgs) -> Result:
    import functools

    from v1.flight.federation_v2.jobs.ignite import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset

    from v1.flight.federation.aggr import Result
    from v1.flight.learning.torch import TorchDataModule, TorchModule

    ####################################################################################

    assert isinstance(args.model, TorchModule)
    model = args.model
    optimizer = model.configure_optimizers()
    loss_fn = model.configure_criterion()
    ignite_cfg = args.ignite_config

    ####################################################################################
    # Setup the PyTorch-Ignite trainer `Engine`.
    ####################################################################################

    if args.train_step is not None:
        wrapped_train_step: ProcessFn = functools.partial(
            args.train_step,
            (model, optimizer, loss_fn),
        )
        trainer = Engine(wrapped_train_step)

    elif args.supervised:
        trainer = create_supervised_trainer(
            model,
            optimizer,
            loss_fn,
            device=ignite_cfg.device,
            non_blocking=ignite_cfg.non_blocking,
            prepare_batch=ignite_cfg.prepare_batch,
            output_transform=ignite_cfg.output_transform,
            gradient_accumulation_steps=ignite_cfg.gradient_accumulation_steps,
        )

    else:
        raise ValueError(
            "Unsupervised training can only be done by providing a `train_step` "
            "function that implements the `UnwrappedProcessFn` interface."
        )

    ####################################################################################

    records: list[Record] = []
    extra: dict[str, t.Any] = {}

    validator: Engine | None = None  # TODO: Implement this.
    tester: Engine | None = None  # TODO: Implement this.

    # local_state = TrainingJobState(**locals())
    local_state = locals()

    for event, handler in args.train_handlers:
        # handler_with_state = functools.partial(handler, local_state)
        # trainer.add_event_handler(event, handler_with_state)
        trainer.add_event_handler(event, handler, local_state)

    if validator:
        for event, handler in args.valid_handlers:
            handler_with_state = functools.partial(handler, local_state)
            validator.add_event_handler(event, handler_with_state)

    if tester:
        for event, handler in args.test_handlers:
            handler_with_state = functools.partial(handler, local_state)
            tester.add_event_handler(event, handler_with_state)

    ####################################################################################

    # assert isinstance(args.data, TorchDataModule)

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, batch_size=64)
    else:
        raise ValueError

    # train_loader = args.data.train_data()
    #
    # print(next(iter(train_loader)))
    trainer.run(train_loader, max_epochs=5)

    return Result(
        node=args.node,
        node_state=args.node_state,
        params=model.get_params(),
        module=model,
        records=records,
        extra={},
    )
