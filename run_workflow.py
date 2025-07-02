import matplotlib.pyplot as plt
import seaborn as sns
import torch

from flight.events import TrainProcessFnEvents, WorkerEvents, on
from flight.learning import TorchModule
from flight.strategies.strategy import DefaultStrategy
from flight.system.utils import flat_topology
from flight.utils.fed_data import federated_split
from flight.workflow import Federation, FederationWorkflow


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        shape: tuple[int, ...] = (28, 28),
        size: int = 1000,
        labels: int = 10,
        seed: int | None = None,
    ):
        if seed:
            torch.manual_seed(seed)

        self._inputs = torch.randn(tuple([size, *shape]))
        self._targets = torch.randint(0, labels, (size,))
        self._data = torch.utils.data.TensorDataset(
            self._inputs,
            self._targets,
        )

        self.size = size
        self.labels = labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._data[idx]


class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        # Define your layers here, for example:
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, 10),
        )

    def forward(self, x):
        # Define your forward pass here
        return self.model(x)

    def configure_optimizers(self):
        # Define your optimizer here
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def configure_criterion(self):
        # Define your loss function here
        return torch.nn.CrossEntropyLoss()


class MyStrategy(DefaultStrategy):
    @on(TrainProcessFnEvents.BACKWARD_COMPLETED)
    def observe_loss(self, context):
        _loss = context["loss"]  # noqa: F841

    @on(WorkerEvents.AFTER_TRAINING)
    def record_output(self, context):
        import datetime

        records = context["records"]
        trainer_state = context["trainer_state"]
        # print(trainer_state.output)

        loss = trainer_state.output[-1]
        records.append(
            {
                "loss": loss,
                "time": datetime.datetime.now(),
                "node": context["node"].idx,
                "round": context["args"].round_num,
            }
        )


def main():
    topo = flat_topology(5)
    data = federated_split(
        topo,
        MyDataset(),
        10,
        10.0,
        10.0,
    )
    print(topo)

    workflow = FederationWorkflow(topo, MyStrategy())
    workflow = Federation(topo, MyStrategy())

    results = workflow.start(
        MyModule(),
        data,
    )

    print(results.head())

    results["node"] = results.node.astype(str)
    sns.lineplot(results, x="round", y="loss", hue="node")
    plt.show()


if __name__ == "__main__":
    # from flight.system.utils import flat_topology, hierarchical_topology

    main()
    # topo = hierarchical_topology(aggr_shape=(2,), n=5)
    # for node in topo.nodes():
    #     print(node, type(node.idx))
    #
    # pprint(dict(topo._nodes))
    # for key in topo._nodes:
    #     print(key, type(key))
    # rel = get_relevant_nodes(topo, [1, 2])
    # print(rel)
