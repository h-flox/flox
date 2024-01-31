import os
from pathlib import Path

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from flox.data.utils import federated_split
from flox.flock import Flock
from flox.run import federated_fit

if __name__ == "__main__":
    flock = Flock.from_yaml("examples/flocks/tutorial-endpoint.yaml")
    mnist = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )
    fed_data = federated_split(mnist, flock, 10, 1.0, 1.0)
    assert len(fed_data) == len(list(flock.workers))

    _, history = federated_fit(
        flock,
        None,  # NOTE: Only valid because of `test_mode=True`
        fed_data,
        5,
        strategy="fedsgd",
        # launcher="globus-compute",
        # runtime=Runtime(),
        debug_mode=True,
        # NOTE: Set the run to testing mode so it just ensures that the launching of jobs is correct.
    )

    history.to_feather(Path("out/demo_testing_history.feather"))
    print(">>> Finished!")
