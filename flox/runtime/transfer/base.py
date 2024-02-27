from typing import Any


class BaseTransfer:
    # def report(
    #     self,
    #     node_state: NodeState | dict[str, Any] | None,
    #     node_idx: FlockNodeID | None,
    #     node_kind: FlockNodeKind | None,
    #     state_dict: StateDict | None,
    #     history: DataFrame | None,
    # ) -> Result:
    #     return JobResult(
    #         node_state=node_state,
    #         node_idx=node_idx,
    #         node_kind=node_kind,
    #         state_dict=state_dict,
    #         history=history,
    #     )

    def report(self, data: Any) -> Any:
        return data

    def proxy(self, data: Any) -> Any:
        return data
