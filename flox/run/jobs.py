# NOTE: These import statements are *just* for type hints. Each 'job' function must be
#       a PURE function with all their dependencies imported within them.
from __future__ import annotations

from dataclasses import dataclass

from pandas import DataFrame

from flox.flock import FlockNode, FlockNodeID, FlockNodeKind
from flox.flock.states import NodeState
from flox.nn.model import FloxModule
from flox.strategies import Strategy
from flox.typing import StateDict
from typing import Optional
from torch.utils.data import Dataset, Subset


@dataclass
class JobResult:
    """A simple dataclass that is returned by jobs executed on Aggregator and Worker nodes in a ``Flock``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState
    """The state of the ``Flock`` node based on its kind."""

    node_idx: FlockNodeID
    """The ID of the ``Flock`` node."""

    node_kind: FlockNodeKind
    """The kind of the ``Flock`` node."""

    state_dict: StateDict
    """The ``StateDict`` of the PyTorch module (either aggregated or trained locally)."""

    history: DataFrame
    """The history of results."""


def local_training_job(
    node: FlockNode,
    parent: FlockNode,
    strategy: Strategy,
    module_cls: type[FloxModule],
    module_state_dict: StateDict,
    dataset: Optional[Dataset | Subset] = None,
    **train_hyper_params,
) -> JobResult:
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
    trainer = Trainer()
    history = trainer.fit(
        local_model,
        train_loader,
        # TODO: Include `trainer_params` as an argument to this so users can easily customize Trainer.
        num_epochs=train_hyper_params.get("num_epochs", 1),
        node_state=node_state,
        strategy=strategy,
    )

    history["node/idx"] = node.idx
    history["node/kind"] = node.kind.to_str()
    history["parent/idx"] = parent.idx
    history["parent/kind"] = parent.kind.to_str()

    return JobResult(node_state, node.idx, node.kind, local_model.state_dict(), history)


def aggregation_job(
    node: FlockNode, strategy: Strategy, results: list[JobResult]
) -> JobResult:
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
    return JobResult(node_state, node.idx, node.kind, avg_state_dict, history)
