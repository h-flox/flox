from __future__ import annotations

import typing

from torch.utils.data import Dataset, Subset

from flox.flock import FlockNode
from flox.nn import FloxModule
from flox.runtime.result import Result
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy

if typing.TYPE_CHECKING:
    from flox.data import FloxDataset
    from flox.nn.typing import StateDict


# TODO: Debug training job should have the same signature.
def local_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
    module_state_dict: StateDict,
    dataset: FloxDataset,  # TODO: Cannot be `None`.
    **train_hyper_params,
) -> Result:
    """Perform local training on a worker node.

    Args:
        node (FlockNode):
        transfer (BaseTransfer): ...
        parent (FlockNode):
        strategy (Strategy):
        module (FloxModule):
        module_state_dict (StateDict):
        dataset (Dataset | Subset | None):
        **train_hyper_params ():

    Returns:
        Local fitting results.
    """
    from copy import deepcopy
    from flox.flock.states import FloxWorkerState
    from flox.nn.trainer import Trainer
    from torch.utils.data import DataLoader
    from flox.runtime import JobResult

    # if isinstance(dataset, LocalDatasetV2):
    #     data = dataset.load()
    # elif isinstance(dataset, FederatedSubsets):
    #     data = dataset[node.idx]

    # match dataset:
    #     case LocalDataset():
    #         data = ...
    #     case FederatedSubsets():
    #         data = ...
    #     case _:
    #         raise ValueError("...")

    global_model = module
    global_state_dict = module.state_dict()
    local_model = deepcopy(module)
    global_model.load_state_dict(module_state_dict)
    local_model.load_state_dict(module_state_dict)

    node_state = FloxWorkerState(
        node.idx, pre_local_train_model=global_model, post_local_train_model=local_model
    )

    strategy.wrk_on_recv_params(node_state, global_state_dict)

    train_loader = DataLoader(
        dataset.load(node),
        batch_size=train_hyper_params.get("batch_size", 32),
        shuffle=train_hyper_params.get("shuffle", True),
    )

    strategy.wrk_before_train_step(node_state, dataset=dataset)
    trainer = Trainer()
    history = trainer.fit(
        local_model,
        train_loader,
        # TODO: Include `trainer_params` as an argument to
        #       this so users can easily customize Trainer.
        num_epochs=train_hyper_params.get("num_epochs", 2),
        node_state=node_state,
        strategy=strategy,
    )

    local_params = strategy.wrk_before_submit_params(node_state)
    history["node/idx"] = node.idx
    history["node/kind"] = node.kind.to_str()
    history["parent/idx"] = parent.idx
    history["parent/kind"] = parent.kind.to_str()

    result = JobResult(node_state, node.idx, node.kind, module.state_dict(), history)
    return transfer.report(result)


def debug_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
):  # -> Result:
    """

    Args:
        node ():
        transfer ():
        parent ():
        strategy ():
        module (FloxModule): ...

    Returns:

    """
    import datetime
    import numpy as np
    import pandas
    from flox.flock.states import FloxWorkerState
    from flox.runtime import JobResult

    local_module = module
    node_state = FloxWorkerState(
        node.idx,
        pre_local_train_model=local_module,
        post_local_train_model=local_module,
    )
    history = {
        "node/idx": [node.idx],
        "node/kind": [node.kind.to_str()],
        "parent/idx": [parent.idx],
        "parent/kind": [parent.kind.to_str()],
        "train/loss": [np.nan],
        "train/epoch": [np.nan],
        "train/batch_idx": [np.nan],
        "train/time": [datetime.datetime.now()],
        "mode": "debug",
    }
    history_df = pandas.DataFrame.from_dict(history)
    result = JobResult(node_state, node.idx, node.kind, module.state_dict(), history_df)
    return transfer.report(result)
