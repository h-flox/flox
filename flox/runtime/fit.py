import datetime
from typing import Any

import numpy as np
from pandas import DataFrame

from flox.data import FloxDataset
from flox.flock import Flock
from flox.nn import FloxModule

# from flox.run.fit_sync import sync_federated_fit
from flox.nn.types import Kind
from flox.runtime.launcher import (
    GlobusComputeLauncher,
    Launcher,
    LocalLauncher,
    ParslLauncher,
)
from flox.runtime.process.proc_async import AsyncProcess
from flox.runtime.process.proc_sync import SyncProcess
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy


def create_launcher(kind: str, **launcher_cfg) -> Launcher:
    if kind == "thread":
        return LocalLauncher(
            pool="thread", n_workers=launcher_cfg.get("max_workers", 3)
        )
    elif kind == "process":
        return LocalLauncher(
            pool="process", n_workers=launcher_cfg.get("max_workers", 3)
        )
    elif kind == "globus-compute":
        return GlobusComputeLauncher()
    elif kind == "parsl":
        return ParslLauncher()
    else:
        raise ValueError("Illegal value for argument `kind`.")


def federated_fit(
    flock: Flock,
    module: FloxModule,
    datasets: FloxDataset,
    num_global_rounds: int,
    strategy: Strategy | str | None = None,
    kind: Kind = "sync",
    launcher: str = "process",
    launcher_cfg: dict[str, Any] | None = None,
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
        launcher (Where):
        launcher_cfg (dict[str, Any] | None):
        debug_mode (bool): ...

    Returns:
        The trained global module hosted on the leader of `flock`.
        The history metrics from training.
    """
    launcher_cfg = dict() if launcher_cfg is None else launcher_cfg
    launcher = create_launcher(launcher, **launcher_cfg)
    transfer = BaseTransfer()

    if strategy is None:
        strategy = "fedsgd"
    if isinstance(strategy, str):
        strategy = Strategy.get_strategy(strategy)()

    # runner = runner_factory.build(kind, ...)
    # runner.start()

    match kind:
        case "sync":
            runner = SyncProcess(
                flock, num_global_rounds, launcher, module, datasets, transfer, strategy
            )
        case "async":
            runner = AsyncProcess(
                flock, num_global_rounds, launcher, module, datasets, transfer, strategy
            )
        case _:
            raise ValueError

    start_time = datetime.datetime.now()
    module, history = runner.start(debug_mode)
    history["train/rel_time"] = history["train/time"] - start_time
    history["train/rel_time"] /= np.timedelta64(1, "s")
    return module, history
