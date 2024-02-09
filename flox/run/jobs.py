# NOTE: These import statements are *just* for type hints. Each 'job' function must be
#       a PURE function with all their dependencies imported within them.
from __future__ import annotations

from flox.flock import FlockNode, FlockNodeID, FlockNodeKind
from flox.nn.model import FloxModule
from flox.reporting import Result
from flox.nn.logger import TensorboardLogger
from flox.strategies import Strategy
from flox.typing import StateDict
from typing import Optional
from torch.utils.data import Dataset, Subset
from flox.backends.transfer.base import BaseTransfer

Transfer: BaseTransfer

def local_training_job(
    node: FlockNode,
    transfer: Transfer, 
    parent: FlockNode,
    strategy: Strategy,
    module_cls: type[FloxModule],
    module_state_dict: StateDict,
    round: int,
    dataset: Optional[Dataset | Subset] = None,
    logger: str = "csv",
    **train_hyper_params,
) -> Result:
    """Perform local training on a worker node.

    Args:
        node (FlockNode):
        parent (FlockNode):
        strategy (Strategy):
        module_cls (type[FloxModule]):
        module_state_dict (StateDict):
        dataset (Optional[Dataset | Subset]):
        **train_hyper_params ():

    Returns:
        Local fitting results.
    """
    from flox.flock.states import FloxWorkerState
    from flox.nn.trainer import Trainer
    from torch.utils.data import DataLoader

    global_model = module_cls()
    local_model = module_cls()
    global_model.load_state_dict(module_state_dict)
    local_model.load_state_dict(module_state_dict)

    node_state = FloxWorkerState(
        node.idx, pre_local_train_model=global_model, post_local_train_model=local_model
    )
    train_loader = DataLoader(
        dataset,
        batch_size=train_hyper_params.get("batch_size", 32),
        shuffle=train_hyper_params.get("shuffle", True),
    )

    strategy.wrk_on_before_train_step(node_state, dataset=dataset)
    trainer = Trainer(node, logger=logger, metadata={
        "node/idx": node.idx,
        "node/kind": node.kind.to_str(),
        "parent/idx": parent.idx,
        "parent/kind": parent.kind.to_str(),
        "round": round,
        "strategy": strategy.get_label()
    })
    history = trainer.fit(
        local_model,
        train_loader,
        # TODO: Include `trainer_params` as an argument to this so users can easily customize Trainer.
        num_epochs=train_hyper_params.get("num_epochs", 1),
        node_state=node_state,
        strategy=strategy,
        round=round,
    )

    return transfer.report(node_state, node.idx, node.kind, local_model.state_dict(), history)


def aggregation_job(
    node: FlockNode, transfer: Transfer, strategy: Strategy, results: list[Result]
) -> Result:
    """Aggregate the state dicts from each of the results.

    Args:
        node (FlockNode): The aggregator node.
        strategy (Strategy): ...
        results (list[JobResult]): Results from children of ``node``.

    Returns:
        Aggregation results.
    """
    import pandas as pd
    from flox.flock.states import FloxAggregatorState

    child_states, child_state_dicts = {}, {}
    for result in results:
        idx = result.node_idx
        child_states[idx] = result.node_state
        child_state_dicts[idx] = result.state_dict

    node_state = FloxAggregatorState(node.idx)
    avg_state_dict = strategy.agg_param_aggregation(
        node_state, child_states, child_state_dicts
    )

    # history = extend_dicts(*(res.history for res in results))
    history = pd.concat([res.history for res in results])
    return transfer.report(node_state, node.idx, node.kind, avg_state_dict, history)
