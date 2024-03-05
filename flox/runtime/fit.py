import datetime
import typing

from pandas import DataFrame

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
from flox.runtime.process.proc import BaseProcess
from flox.runtime.process.proc_async import AsyncProcess
from flox.runtime.process.proc_sync import SyncProcess
from flox.runtime.runtime import Runtime
from flox.runtime.transfer import BaseTransfer
from flox.strategies_depr import Strategy


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
    strategy: Strategy | str | None = None,
    kind: Kind = "sync",
    launcher_kind: str = "process",
    launcher_cfg: dict[str, typing.Any] | None = None,
    debug_mode: bool = False,
) -> tuple[FloxModule, DataFrame]:
    """

    Args:
        flock (Flock):
        module (FloxModule):
        datasets (FloxDataset):
        num_global_rounds (int):
        strategy (Strategy | str | None):
        kind (Kind):
        launcher_kind (str):
        launcher_cfg (dict[str, typing.Any] | None):
        debug_mode (bool): ...

    Returns:
        The trained global module hosted on the leader of `flock`.
        The history metrics from training.
    """
    launcher_cfg = dict() if launcher_cfg is None else launcher_cfg
    launcher = create_launcher(launcher_kind, **launcher_cfg)
    transfer = BaseTransfer()
    runtime = Runtime(launcher, transfer)

    if strategy is None:
        strategy = "fedsgd"
    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    # runner = runner_factory.build(kind, ...)
    # runner.start()
    process: BaseProcess
    match kind:
        case "sync":
            process = SyncProcess(
                runtime=runtime,
                flock=flock,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=strategy,
            )
        case "async":
            process = AsyncProcess(
                runtime=runtime,
                flock=flock,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=strategy,
            )
        case _:
            raise ValueError("Illegal value for the strategy `kind` parameter.")

    start_time = datetime.datetime.now()
    module, history = process.start(debug_mode)
    history["train/rel_time"] = history["train/time"] - start_time
    history["train/rel_time"] = history["train/rel_time"].dt.total_seconds()
    return module, history
