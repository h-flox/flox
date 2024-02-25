from typing import Any

from pandas import DataFrame

from flox.flock import FlockNodeID, FlockNodeKind
from flox.flock.states import NodeState
from flox.runtime.result import JobResult, Result
from flox.typing import StateDict


class BaseTransfer:
    def report(
        self,
        node_state: NodeState | dict[str, Any] | None,
        node_idx: FlockNodeID | None,
        node_kind: FlockNodeKind | None,
        state_dict: StateDict | None,
        history: DataFrame | None,
    ) -> Result:
        return JobResult(
            node_state=node_state,
            node_idx=node_idx,
            node_kind=node_kind,
            state_dict=state_dict,
            history=history,
        )

    def proxy(self, data: Any) -> Any:
        return data
