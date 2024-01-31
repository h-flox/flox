from torch.utils.data import Dataset, Subset

from flox.flock import FlockNode
from flox.nn import FloxModule
from flox.runtime.result import Result
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy
from flox.typing import StateDict


# TODO: Debug training job should have the same signature.
def local_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
    module_state_dict: StateDict,
    dataset: Dataset | Subset | None = None,  # TODO: Cannot be `None`.
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

    global_model = module
    global_state_dict = module.state_dict()
    local_model = deepcopy(module)
    global_model.load_state_dict(module_state_dict)
    local_model.load_state_dict(module_state_dict)

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

    node_state = FloxWorkerState(
        node.idx, pre_local_train_model=global_model, post_local_train_model=local_model
    )
    train_loader = DataLoader(
        dataset,
        batch_size=train_hyper_params.get("batch_size", 32),
        shuffle=train_hyper_params.get("shuffle", True),
    )

    strategy.wrk_on_before_train_step(node_state, dataset=dataset)
    trainer = Trainer()
    history = trainer.fit(
        local_model,
        train_loader,
        # TODO: Include `trainer_params` as an argument to this so users can easily customize Trainer.
        num_epochs=train_hyper_params.get("num_epochs", 2),
        node_state=node_state,
        strategy=strategy,
    )

    history["node/idx"] = node.idx
    history["node/kind"] = node.kind.to_str()
    history["parent/idx"] = parent.idx
    history["parent/kind"] = parent.kind.to_str()

    return transfer.report(
        node_state, node.idx, node.kind, local_model.state_dict(), history
    )


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
    local_module = module
    node_state = dict(
        idx=node.idx,
        pre_local_train_model=local_module,
        post_local_train_module=local_module,
    )
    return transfer.report(node_state, node.idx, node.kind, module.state_dict(), {})
