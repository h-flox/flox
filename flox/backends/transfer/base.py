from flox.reporting import Result
from flox.reporting import JobResult

class BaseTransfer:

    def report(self, node_state, node_idx, node_kind, state_dict, history) -> Result:
        return JobResult(
            node_state=node_state,
            node_idx=node_idx,
            node_kind=node_kind,
            state_dict=state_dict,
            history=history
        )