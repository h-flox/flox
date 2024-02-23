from __future__ import annotations

from typing import Any

from pandas import DataFrame

from flox.backends.launcher import (
    GlobusComputeLauncher,
    Launcher,
    LocalLauncher,
    ParslLauncher,
)
from flox.data import FloxDataset
from flox.flock import Flock
from flox.nn import FloxModule
from flox.nn.types import Kind
from flox.run.fit_sync import sync_federated_fit
from flox.strategies import Strategy


def create_launcher(kind: str, **launcher_cfg) -> Launcher:
    if kind == "local-thread":
        return LocalLauncher(
            pool="thread", n_workers=launcher_cfg.get("max_workers", 1)
        )
    elif kind == "local-proc":
        return LocalLauncher(
            pool="process", n_workers=launcher_cfg.get("max_workers", 1)
        )
    elif kind == "globus-compute":
        return GlobusComputeLauncher()
    elif kind == "parsl":
        return ParslLauncher()
    else:
        raise ValueError("Illegal value for argument `kind`.")


def federated_fit(
    flock: Flock,
    module_cls: type[FloxModule],
    datasets: FloxDataset,
    num_global_rounds: int,
    strategy: Strategy | str | None = None,
    kind: Kind = "sync",
    launcher: str = "local-thread",
    launcher_cfg: dict[str, Any] | None = None,
) -> tuple[FloxModule, DataFrame]:
    """

    Args:
        flock (Flock):
        module_cls (type[FloxModule]):
        datasets (FloxDataset):
        num_global_rounds (int):
        strategy (Strategy):
        kind (Kind):
        launcher (Where):
        launcher_cfg (Optional[dict[str, Any]]):

    Returns:

    """
    launcher_cfg = dict() if launcher_cfg is None else launcher_cfg
    # launcher = create_launcher(launcher, **launcher_cfg)  # not used

    strategy = "fedsgd" if strategy is None else strategy

    if kind == "sync":
        # executor = "thread" if where == "local" else "globus_compute"
        executor = "thread"
        return sync_federated_fit(
            flock,
            module_cls,
            datasets,
            num_global_rounds,
            strategy,
            executor
            # , where=where
        )
    elif kind == "async":
        raise NotImplementedError("Asynchronous FL is not yet implemented.")
    else:
        raise ValueError(
            "Illegal value for argument `kind`. Must be either 'sync' or 'async'."
        )
