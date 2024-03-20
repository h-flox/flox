import argparse
import datetime
import json
import logging
import warnings

from pathlib import Path


logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import flox
    import os
    import pandas as pd

    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor

    from flox.data.utils import federated_split
    from flox.flock.factory import create_standard_flock
    from flox.strategies import load_strategy
    from models import *


def main(**kwargs):
    config = argparse.Namespace(**kwargs)
    flock = create_standard_flock(num_workers=config.num_workers)
    logging.info(f"Flock is created with {flock.number_of_workers} workers")
    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )
    logging.info("FashionMNIST data is loaded")
    fed_data = federated_split(
        data,
        flock,
        num_classes=10,
        labels_alpha=config.labels_alpha,
        samples_alpha=config.samples_alpha,
    )
    logging.info("Data is federated")

    results = []
    for strategy in ["fedprox", "fedavg", "fedsgd"]:
        logging.info(f"Starting federated learning with strategy '{strategy}'.")
        _, res = flox.federated_fit(
            flock,
            SmallConvModel(),
            # SmallModel(),
            fed_data,
            num_global_rounds=50,
            strategy=load_strategy(strategy, participation=config.participation),
            launcher_cfg={"max_workers": config.max_workers},
        )
        res["strategy"] = strategy
        results.append(res)

    # Save data results and the config.
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split(".")[0]
    out_dir = Path(f"experiments/aggr_comparison/{timestamp}/")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    df = pd.concat(results)
    df.to_feather(out_dir / "results.feather")
    with open(out_dir / "config.json", "w") as file:
        json.dump(dict(**kwargs), file)


if __name__ == "__main__":
    import caffeine

    caffeine.on(display=False)
    main(
        num_workers=50,
        labels_alpha=0.1,
        samples_alpha=2.0,
        max_workers=10,
        participation=1.0,
    )
    caffeine.off()