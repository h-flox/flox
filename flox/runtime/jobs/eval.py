from torch.utils.data import Dataset, Subset

from flox.flock import FlockNode
from flox.nn import FloxModule
from flox.runtime import Result
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy
from flox.typing import StateDict


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
    pass


def aggr_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
    module_state_dict: StateDict,
    dataset: Dataset | Subset | None = None,  # TODO: Cannot be `None`.
    **train_hyper_params,
) -> Result:
    pass
