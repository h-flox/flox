from __future__ import annotations

import pathlib

import numpy as np

from v1.flight.topologies import TopologyLike

from .engine.controllers.base import AbstractController
from .engine.controllers.local import LocalController
from .engine.controllers.serial import SerialController
from .federation import SyncFederation, Topology
from .learning.base import AbstractDataModule, AbstractModule
from .strategies import Strategy
from .strategies.impl import FedSGD
from .types import Record


def load_topology(raw_data: TopologyLike):
    """
    Load a `Topology` instance based on the data type provided here.

    Args:
        raw_data (TopologyLike): The raw data to load as a `Topology`.

    Returns:
        A `Topology` instance.

    Raises:
        - `ValueError`: If the value of `raw_data` is not a valid type.
    """
    match raw_data:
        case Topology():
            return raw_data
        case pathlib.Path() | str():
            return Topology.from_file(raw_data)
        case list() | np.ndarray():
            return Topology.from_adj_matrix(raw_data)
        case dict():
            return Topology.from_dict(raw_data)
        case _:
            raise ValueError("Illegal value for argument `raw_data`.")


def load_controller(
    controller_type: str,
    **controller_config,
) -> AbstractController:
    """
    Load a controller based on the type provided.

    Args:
        controller_type (str): The type of `Controller` to load.
        **controller_config: The configuration keyword arguments to pass to the
            controller.

    Returns:
        An instance of a `Controller`.
    """
    match controller_type:
        case "serial":
            return SerialController(**controller_config)
        case "local":
            kind = controller_config.get("kind", "thread")
            return LocalController(kind, **controller_config)
        case _:
            raise ValueError


def federated_fit(
    topology: TopologyLike,
    module: AbstractModule,
    data: AbstractDataModule,
    rounds: int = 1,
    strategy: Strategy | str = "fedsgd",
    mode: str = "sync",
    fast_dev_run: bool = False,
) -> tuple[AbstractModule, list[Record]]:
    """
    Run a federated learning experiment.

    Args:
        topology (Topology | pathlib.Path | str | dict): The topology to use for
            federation. This can be a `Topology` instance, a path to a file containing
            a topology, an adjacency matrix, or a dictionary representation of a
            topology.
        module (AbstractModule): The global module to train. This module is typically
            untrained.
        data (AbstractDataModule): The data module to use for training. Specifically,
            this module defines how data are loaded by the worker nodes to do local
            training.
        rounds (int): The number of federation rounds. Defaults to 1.
        strategy (Strategy | str): The `Strategy` to use. If the value is a `str`,
            the registry of existing `Strategy` implementations will be indexed.
            Defaults to `fedsgd`.
        mode (str): The type of federation to run (i.e., sync or async).
            Defaults to `sync`.
        fast_dev_run (bool): If `True`, this will run a quick federation to ensure
            most dependencies run and load as expected. In this case, training is
            skipped, so results are not meaningful. This is helpful to quickly test
            if custom `Strategy` implementations do not cause simple errors
            (e.g., import errors). Defaults to `False`.

    Returns:
        A tuple containing:

            1. the trained global module and
            2. a list of result records.
    """
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
