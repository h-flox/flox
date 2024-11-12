from __future__ import annotations

import pathlib

import numpy as np

from .engine.controllers.base import AbstractController
from .engine.controllers.local import LocalController
from .engine.controllers.serial import SerialController
from .federation import SyncFederation, Topology
from .learning.base import AbstractDataModule, AbstractModule
from .strategies import Strategy
from .strategies.impl import FedSGD
from .types import Record


def load_topology(raw_data: Topology | pathlib.Path | str | dict):
    # Load and parse a `Topology` class instance based on the data type provided here.
    match raw_data:
        case Topology():
            return raw_data
        case pathlib.Path() | str():
            return Topology.from_file(raw_data)
        case list() | np.ndarray():
            return Topology.from_adj_matrix(raw_data)
        case dict():
            return Topology.from_dict(raw_data)


def load_controller(
    controller_type: str,
    **controller_config,
) -> AbstractController:
    match controller_type:
        case "serial":
            return SerialController(**controller_config)
        case "local":
            kind = controller_config.get("kind", "thread")
            return LocalController(kind, **controller_config)
        case _:
            raise ValueError


def federated_fit(
    topology: Topology | pathlib.Path | str | dict,
    module: AbstractModule,
    data: AbstractDataModule,
    rounds: int = 1,
    strategy: Strategy | str = "fedsgd",
    mode: str = "sync",
    fast_dev_run: bool = False,
) -> tuple[AbstractModule, list[Record]]:
    if strategy == "fedsgd":
        strategy = FedSGD()
    else:
        raise ValueError("Fix this later.")

    shared_federation_args = dict(
        topology=load_topology(topology),
        strategy=strategy,
        module=module,
        data=data,
        logger=None,  # TODO
        debug=False,  # TODO
    )
    match mode:
        case "async":
            # args = dict(a="a", **shared_federation_args)
            # federation = AsyncFederation(**args)
            raise NotImplementedError(
                "`AsyncFederation` is not yet implemented in Flight."
            )
        case "sync":
            args = dict(**shared_federation_args)
            federation = SyncFederation(**args)
        case _:
            raise ValueError("Illegal value for argument `mode`.")

    records = federation.start(rounds)
    return module, records
