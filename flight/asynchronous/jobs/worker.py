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


def _default_dataset_cfg() -> dict[str, t.Any]:
    return {"batch_size": 64}


class WorkerJobProto(t.Protocol):
    @staticmethod
    async def __call__(args: WorkerJobArgs) -> Result:
        """
        This method is called when the WORKER job is launched.
        """

@dataclass
class WorkerJobArgs:
    strategy: Strategy
    model: TorchModule
    data: TorchDataModule | DataLoader | Dataset
    params: Params
    node: Node | None = None
    train_step: t.Any | None = None
    supervised: bool = field(default=True)
    dataset_cfg: dict[str, t.Any] = field(default_factory=_default_dataset_cfg)


def _default_prepare_batch(
    batch: t.Sequence["torch.Tensor"],
    device: t.Optional[str | "torch.device"] = None,
    non_blocking: bool = False,
) -> tuple["torch.Tensor" | t.Sequence | t.Mapping | str | bytes, ...]:
    from ignite.utils import convert_tensor
    
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


@dataclass
class IgniteConfig:
    supervised: bool = True
    device: str | "torch.device" | None = "cpu"
    non_blocking: bool = True
    prepare_batch: t.Callable = _default_prepare_batch
    model_transform: t.Callable[[t.Any], t.Any] = lambda output: output
    output_transform: t.Callable[
        [t.Any, t.Any, t.Any, "torch.Tensor"], t.Any
    ] = lambda x, y_true, y_pred, loss: (y_true, y_pred)
    gradient_accumulation_steps: int = 1
    model_fn: t.Callable[["torch.nn.Module", t.Any], t.Any] = lambda model, x: model(x)


async def worker_job(args: WorkerJobArgs):
    import functools
    from ignite.engine import Engine, create_supervised_trainer
    from torch.utils.data import DataLoader, Dataset
    from flight.events import IgniteEvents, WorkerEvents
    from flight.jobs.protocols import Result
    from flight.learning.module import TorchDataModule, TorchModule

    context: dict[str, t.Any] = {}

    # Await event handler if async
    if hasattr(args.strategy, "fire_event_handler") and asyncio.iscoroutinefunction(args.strategy.fire_event_handler):
        await args.strategy.fire_event_handler(WorkerEvents.STARTED, context)
    else:
        args.strategy.fire_event_handler(WorkerEvents.STARTED, context)

    assert isinstance(args.model, TorchModule)

    optimizer = args.model.configure_optimizers()
    loss_fn = args.model.configure_criterion()
    ignite_cfg = IgniteConfig()

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
            device=ignite_cfg.device,
            non_blocking=ignite_cfg.non_blocking,
            prepare_batch=ignite_cfg.prepare_batch,
            output_transform=ignite_cfg.output_transform,
            gradient_accumulation_steps=ignite_cfg.gradient_accumulation_steps,
        )
    else:
        raise ValueError()

    validator: Engine | None = None  # TODO: Implement this.
    tester: Engine | None = None  # TODO: Implement this.

    train_handlers = args.strategy.get_event_handlers_by_genre(IgniteEvents)
    for event, handler in train_handlers:
        handler_with_state = functools.partial(handler, trainer, context)
        trainer.add_event_handler(event, handler_with_state)

    if isinstance(args.data, TorchDataModule):
        train_loader = args.data.train_data()
    elif isinstance(args.data, DataLoader):
        train_loader = args.data
    elif isinstance(args.data, Dataset):
        train_loader = DataLoader(args.data, **args.dataset_cfg)
    else:
        raise ValueError("`data` must be a TorchDataModule, DataLoader, or Dataset.")

    context = locals()
    if hasattr(args.strategy, "fire_event_handler") and asyncio.iscoroutinefunction(args.strategy.fire_event_handler):
        await args.strategy.fire_event_handler(WorkerEvents.BEFORE_TRAINING, context)
    else:
        args.strategy.fire_event_handler(WorkerEvents.BEFORE_TRAINING, context)

    max_epochs = 3  # TODO: Parameterize
    epoch_length = None  # TODO: Parameterize

    # Run the blocking trainer in a thread to avoid blocking the event loop
    trainer_state = await asyncio.to_thread(
        trainer.run,
        train_loader,
        max_epochs=max_epochs,
        epoch_length=epoch_length,
    )

    context = locals()
    if hasattr(args.strategy, "fire_event_handler") and asyncio.iscoroutinefunction(args.strategy.fire_event_handler):
        await args.strategy.fire_event_handler(WorkerEvents.AFTER_TRAINING, context)
        await args.strategy.fire_event_handler(WorkerEvents.COMPLETED, context)
    else:
        args.strategy.fire_event_handler(WorkerEvents.AFTER_TRAINING, context)
        args.strategy.fire_event_handler(WorkerEvents.COMPLETED, context)

    return Result(
        node=args.node,
        state=None,
        params=args.model.get_params(),
        module=args.model,
        extra={},
    )