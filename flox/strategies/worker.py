import typing as t

from flox.flock.states import NodeState

if t.TYPE_CHECKING:
    pass


class WorkerStrategy(t.Protocol):
    def work_start(self):
        pass

    def before_training(self, state: NodeState, data: t.Any):
        pass

    def after_training(self, node_state: NodeState):
        pass

    def work_end(self):
        pass
