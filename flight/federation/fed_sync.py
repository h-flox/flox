import typing as t
from concurrent.futures import Future

from ..learning.datasets.core import DataLoadable
from ..learning.module import Trainable
from ..strategies.base import Strategy
from .fed_abs import Federation
from .topologies.topo import Topology

if t.TYPE_CHECKING:
    from .jobs.types import Result

    Engine: t.TypeAlias = t.Any


def log(msg: str):
    print(msg)


class SyncFederation(Federation):
    def __init__(
        self,
        module: Trainable,
        data: DataLoadable,
        topology: Topology,
        strategy: Strategy,
        engine: Engine,
        #
        logger=None,
        debug=None,
    ):
        super().__init__(topology, strategy)
        self.module = module
        self.data = data
        self.engine = engine
        self.exceptions = []
        self.global_model = None

    def start(self, rounds: int):
        for round_no in range(rounds):
            self.federation_round()

    def federation_round(self):
        log("Starting round")
        global_params = self.global_model.get_params()
        # NOTE: Be sure to wrap `result` calls to handle errors.
        try:
            round_future = self.federation_step()
            round_results = round_future.result()
        except Exception as exc:
            self.exceptions.append(exc)
            raise exc

    def federation_step(self) -> Future:
        self.params = self.global_model.state_dict()
        step_result = self.traverse_step().result()
        step_result.history["round"] = round_num

        if not debug_mode:
            test_acc, test_loss = test_model(self.global_model)
            step_result.history["test/acc"] = test_acc
            step_result.history["test/loss"] = test_loss

        histories.append(step_result.history)
        self.global_model.load_state_dict(step_result.params)

        if self.pbar:
            self.pbar.update()

    def traverse_step(
        self,
        node: t.Optional[Node] = None,
        parent: t.Optional[Node] = None,
    ) -> Future[Result]:
        node = Federation._resolve_node(node)
