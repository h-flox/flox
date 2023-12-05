from pandas import DataFrame

from flox.flock import FlockNodeID, FlockNodeKind
from flox.flock.states import NodeState
from flox.reporting import JobResult, Result
from flox.typing import StateDict


class BaseTransfer:
    def report(
        self,
        node_state: NodeState,
        node_idx: FlockNodeID,
        node_kind: FlockNodeKind,
        state_dict: StateDict,
        history: DataFrame,
    ) -> Result:
        return JobResult(
            node_state=node_state,
            node_idx=node_idx,
            node_kind=node_kind,
            state_dict=state_dict,
            history=history,
        )
