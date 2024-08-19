import logging
import warnings

logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import os

    import pandas as pd
    from models import *
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor

    import flox
    from flox.data.utils import federated_split
    from flox.federation.topologies import create_hier_flock
    from flox.strategies import load_strategy


def main():
    flock = create_hier_flock(5, 2)
    logging.info(f"Flock is created with {flock.number_of_workers} workers")
    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )
    logging.info("FashionMNIST data is loaded")
    fed_data = federated_split(
        data,
        flock,
        num_classes=10,
        labels_alpha=0.1,
        samples_alpha=2.0,
    )
    logging.info("Data is federated")

    results = []
    for strategy in ["fedprox", "fedavg", "fedsgd"]:
        logging.info(f"Starting federated learning with strategy '{strategy}'.")
        _, res = flox.federated_fit(
            flock,
            SmallConvModel(device="cpu"),
            fed_data,
            num_global_rounds=25,
            strategy=load_strategy(strategy, participation=0.25),
            launcher_cfg={"max_workers": 10},
        )
        res["strategy"] = strategy
        results.append(res)

    df = pd.concat(results)
    df.to_feather("my_hier_results.feather")


if __name__ == "__main__":
    main()
