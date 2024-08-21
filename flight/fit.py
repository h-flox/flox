from __future__ import annotations

import pathlib

import numpy as np

from .federation import SyncFederation, Topology
from .federation.jobs.types import Result
from .learning.modules import HasParameters
from .learning.modules.prototypes import DataModuleProto
from .strategies import Strategy
from .strategies.impl import FedSGD


def load_topology(raw_data: Topology | pathlib.Path | str | dict):
    # Load and parse a `Topology` class instance based on the data type provided here.
    match raw_data:
        case Topology():
            return Topology
        case pathlib.Path() | str():
            return Topology.from_file(raw_data)
        case list() | np.ndarray():
            return Topology.from_adj_matrix(raw_data)
        case dict():
            return Topology.from_dict(raw_data)


def federated_fit(
    topology: Topology | pathlib.Path | str | dict,
    module: HasParameters,
    data: DataModuleProto,
    rounds: int = 1,
    strategy: Strategy | str = "fedsgd",
    mode: str = "sync",
) -> tuple[HasParameters, list[Result]]:
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

    results = federation.start(rounds)
    return module, results
