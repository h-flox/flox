import datetime
import typing as t

from pandas import DataFrame

import flox.strategies as strats
from flox.data import FloxDataset
from flox.flock import Flock
from flox.nn import FloxModule
from flox.nn.typing import Kind
from flox.runtime.launcher import (
    GlobusComputeLauncher,
    Launcher,
    LocalLauncher,
    ParslLauncher,
)
from flox.runtime.process.process import Process
from flox.runtime.process.process_async import AsyncProcess
from flox.runtime.process.process_sync import SyncProcess
from flox.runtime.runtime import Runtime
from flox.runtime.transfer import BaseTransfer


def create_launcher(kind: str, **launcher_cfg) -> Launcher:
    match kind:
        case "thread":
            return LocalLauncher(
                pool="thread", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "process":
            return LocalLauncher(
                pool="process", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "globus-compute":
            return GlobusComputeLauncher()
        case "parsl":
            return ParslLauncher()
        case _:
            raise ValueError("Illegal value for argument `kind`.")


def federated_fit(
    flock: Flock,
    module: FloxModule,
    datasets: FloxDataset,
    num_global_rounds: int,
    # Strategy arguments.
    strategy: strats.Strategy | str | None = None,
    client_strategy: strats.ClientStrategy | None = None,
    aggr_strategy: strats.AggregatorStrategy | None = None,
    worker_strategy: strats.WorkerStrategy | None = None,
    trainer_strategy: strats.TrainerStrategy | None = None,
    # Process arguments.
    kind: Kind = "sync",
    launcher_kind: str = "process",
    launcher_cfg: dict[str, t.Any] | None = None,
    debug_mode: bool = False,
) -> tuple[FloxModule, DataFrame]:
    """

    Args:
        flock (Flock):
        module (FloxModule):
        datasets (FloxDataset):
        num_global_rounds (int):
        strategy (Strategy | str | None):
        client_strategy (strats.ClientStrategy): ...
        aggr_strategy (strats.AggregatorStrategy): ...
        worker_strategy (strats.WorkerStrategy): ...
        trainer_strategy (strats.TrainerStrategy): ...
        kind (Kind):
        launcher_kind (str):
        launcher_cfg (dict[str, t.Any] | None):
        debug_mode (bool): ...

    Returns:
        The trained global module hosted on the leader of `flock`.
        The history metrics from training.
    """
    launcher_cfg = dict() if launcher_cfg is None else launcher_cfg
    launcher = create_launcher(launcher_kind, **launcher_cfg)
    transfer = BaseTransfer()
    runtime = Runtime(launcher, transfer)
    parsed_strategy = parse_strategy_args(
        strategy=strategy,
        client_strategy=client_strategy,
        aggr_strategy=aggr_strategy,
        worker_strategy=worker_strategy,
        trainer_strategy=trainer_strategy,
    )

    # runner = runner_factory.build(kind, ...)
    # runner.start()
    process: Process
    match kind:
        case "sync":
            process = SyncProcess(
                runtime=runtime,
                flock=flock,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=parsed_strategy,
            )
        case "async":
            process = AsyncProcess(
                runtime=runtime,
                flock=flock,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=parsed_strategy,
            )
        case _:
            raise ValueError("Illegal value for the strategy `kind` parameter.")

    start_time = datetime.datetime.now()
    module, history = process.start(debug_mode)
    history["train/rel_time"] = history["train/time"] - start_time
    history["train/rel_time"] = history["train/rel_time"].dt.total_seconds()
    return module, history


def parse_strategy_args(
    strategy: strats.Strategy | str | None,
    client_strategy: strats.ClientStrategy | None,
    aggr_strategy: strats.AggregatorStrategy | None,
    worker_strategy: strats.WorkerStrategy | None,
    trainer_strategy: strats.TrainerStrategy | None,
    **kwargs,
) -> strats.Strategy:
    if isinstance(strategy, strats.Strategy):
        return strategy

    if isinstance(strategy, str):
        return strats.load_strategy(strategy, **kwargs)

    if strategy is not None:
        raise ValueError(
            "Argument ``strategy`` is not a legal value. Must be either a ``Strategy``, "
            "a supported string value, or ``None``. "
        )

    # If the user provided each individual strategy implementations, then we must first check and confirm
    # that none of those arguments are ``None``. If they are not, then we can package them as a single
    # ``Strategy`` and return that.
    strategies = [client_strategy, aggr_strategy, worker_strategy, trainer_strategy]
    for _name, _strategy in zip(["client", "aggr", "worker", "trainer"], strategies):
        if _strategy is None:
            cls_name = "aggregator" if _name == "aggr" else _name
            cls_name = cls_name.title()
            raise ValueError(
                f"Argument `{_name}_strategy` must be a class that implements protocol ``{cls_name}``."
            )

    # Explicit asserts to satisfy `mypy`.
    assert client_strategy is not None
    assert aggr_strategy is not None
    assert worker_strategy is not None
    assert trainer_strategy is not None

    return strats.Strategy(
        client_strategy=client_strategy,
        aggr_strategy=aggr_strategy,
        worker_strategy=worker_strategy,
        trainer_strategy=trainer_strategy,
    )
