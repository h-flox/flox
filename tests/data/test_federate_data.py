import os

from flox.flock import Flock
from flox.utils.data.beta import randomly_federate_dataset
from numpy.random import RandomState
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def test_randomly_federate_dataset():
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    mnist = MNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )
    fed_data = randomly_federate_dataset(
        flock,
        mnist,
        shuffle=True,
        random_state=RandomState(1),
    )
    indices = set()
    flag = True
    for node, node_indices in fed_data.items():
        if flag is False:
            break
        for ind in node_indices:
            if ind in indices:
                flag = False
                break
            indices.add(ind)

    assert flag, "There were duplicate indices when there should be none."
