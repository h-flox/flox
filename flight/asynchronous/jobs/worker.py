from __future__ import annotations
import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from flight.learning.module import TorchDataModule, TorchModule
    from flight.strategies.strategy import Strategy
    from flight.system.node import Node
    from flight.learning.parameters import Params
    from flight.jobs.protocols import ProcessFn, Result

@dataclass
class AsyncWorkerJobArgs:
    strategy: Strategy
    model: TorchModule
    data: TorchDataModule | DataLoader | Dataset
    params: Params
    node: Node | None = None
    train_step: t.Any | None = None
    supervised: bool = field(default=True)
    dataset_cfg: dict[str, t.Any] = field(default_factory=lambda: {"batch_size": 64})

def async_worker_job(args: AsyncWorkerJobArgs, send_result: t.Callable[[t.Any], None]):
    import functools
    from ignite.engine import Engine, Events, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset
    from flight.jobs.protocols import Result
    from flight.learning.module import TorchDataModule, TorchModule

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, **args.dataset_cfg)
    else:
        raise ValueError("`data` must be a TorchDataModule, DataLoader, or Dataset.")

    optimizer = args.model.configure_optimizers()
    loss_fn = args.model.configure_criterion()

    if args.train_step is not None:
        wrapped_train_step: ProcessFn = functools.partial(
            args.train_step,
            args.model,
            optimizer,
            loss_fn,
        )
        trainer = Engine(wrapped_train_step)
    elif args.supervised:
        trainer = create_supervised_trainer(
            args.model,
            optimizer,
            loss_fn,
            device="cpu",
            non_blocking=True,
        )
    else:
        raise ValueError("Unsupervised training requires a custom train_step.")

    # Send result to aggregator after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        result = Result(
            node=args.node,
            state=None,
            params=args.model.get_params(),
            module=args.model,
            extra={"epoch": engine.state.epoch},
        )
        send_result(result)

    max_epochs = 3  
    trainer.run(train_loader, max_epochs=max_epochs)