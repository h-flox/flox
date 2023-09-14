from __future__ import annotations

# NOTE: These import statements are *just* for type hints. Each 'job' function must be
#       a PURE function with all there dependencies imported within them.
from flox.flock import FlockNode
from flox.learn.nn.model import FloxModule
from flox.run.update import TaskUpdate
from flox.strategies import Strategy
from flox.typing import StateDict
from typing import Optional
from torch.utils.data import Dataset, Subset


def local_fitting_job(
    node: FlockNode,
    parent: FlockNode,
    strategy: Strategy,
    module_cls: type[FloxModule],
    module_state_dict: StateDict,
    dataset: Optional[Dataset | Subset] = None,
    **train_hyper_params,
):
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

    """
    from flox.flock.states import FloxWorkerState
    from flox.learn.nn.trainer import Trainer
    from flox.run.update import TaskUpdate
    from torch.utils.data import DataLoader

    global_model = module_cls()
    local_model = module_cls()
    global_model.load_state_dict(module_state_dict)
    local_model.load_state_dict(module_state_dict)

    node_state = FloxWorkerState(
        pre_local_train_model=global_model, post_local_train_model=local_model
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
        num_epochs=train_hyper_params.get("num_epochs", 1),
        node_state=node_state,
        strategy=strategy,
    )

    history["node/idx"] = node.idx
    history["node/kind"] = node.kind.to_str()
    history["parent/idx"] = parent.idx
    history["parent/kind"] = parent.kind.to_str()

    return TaskUpdate(
        node_state, node.idx, node.kind, local_model.state_dict(), history
    )


def aggregation_job(
    node: FlockNode, strategy: Strategy, updates: list[TaskUpdate]
) -> TaskUpdate:
    """Aggregate the state dicts from each of the results.

    Args:
        node (FlockNode): The aggregator node.
        strategy (Strategy): ...
        updates (list[TaskUpdate]): Results from children of ``node``.

    Returns:
        dict[str, Any]: Aggregation results.
    """
    import pandas as pd

    from flox.flock.states import FloxAggregatorState
    from flox.run.update import TaskUpdate

    child_states, child_state_dicts = {}, {}
    for update in updates:
        idx = update.node_idx
        child_states[idx] = update.node_state
        child_state_dicts[idx] = update.state_dict

    node_state = FloxAggregatorState()
    avg_state_dict = strategy.agg_on_param_aggregation(
        node_state, child_states, child_state_dicts
    )

    # history = extend_dicts(*(update.history for update in updates))
    history = pd.concat([update.history for update in updates])
    return TaskUpdate(node_state, node.idx, node.kind, avg_state_dict, history)
