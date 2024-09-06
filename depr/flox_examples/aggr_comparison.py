import argparse
import datetime
import json
import logging
import warnings
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import os

    import pandas as pd
    import torchvision.transforms as transforms
    from models import *
    from torchvision.datasets import FashionMNIST

    import flox
    from flox import Topology
    from flox.data import FloxDataset
    from flox.data.utils import federated_split
    from flox.federation.topologies import hierarchical_topology
    from flox.strategies import load_strategy


def train_experiment(
    strategy_name: str,
    flock: Topology,
    fed_data: FloxDataset,
    config: argparse.Namespace,
    use_small_model: bool = True,
) -> pd.DataFrame:
    logging.info(f"Starting federated learning with strategy '{strategy_name}'.")

    if "async" in strategy_name:
        strategy = load_strategy(strategy_name, alpha=config.alpha)
        kind = "async"
    else:
        strategy = load_strategy(strategy_name, participation=config.participation)
        kind = "sync"

    if use_small_model:
        module = SmallModel()
    else:
        module = SmallConvModel()

    _, result = flox.federated_fit(
        flock,
        module,
        fed_data,
        num_global_rounds=config.num_global_rounds,
        strategy=strategy,
        kind="sync-v2",
        debug_mode=False,
        launcher_kind="federation",
        # launcher_kind="thread",
        launcher_cfg={"max_workers": config.max_workers},
    )
    result["strategy"] = strategy_name
    logging.info(f"FINISHED learning with strategy '{strategy_name}'.")
    return result


def main(**kwargs):
    config = argparse.Namespace(**kwargs)
    # topologies = create_standard_flock(num_workers=config.num_worker_nodes)
    flock = hierarchical_topology(config.num_worker_nodes, [2, 3])
    # topologies.draw()
    # plt.show()
    logging.info(f"Flock is created with {flock.number_of_workers} workers.")

    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        # train=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )
    logging.info("FashionMNIST data is loaded.")

    # strategies = ["fedasync", "fedprox", "fedavg", "fedsgd"]
    # strategies = ["fedavg"]
    # strategies = ["fedprox", "fedavg", "fedsgd"]
    # strategies = ["fedprox", "fedavg"]  # NOTE: This one.
    strategies = ["fedavg"]

    results = []
    label_alpha_list = [0.01, 1000.0]
    sample_alpha_list = [2.0, 1000.0]
    label_alpha_list = [1000.0]
    sample_alpha_list = [1000.0]

    for i, l_alpha in enumerate(label_alpha_list):
        for j, s_alpha in enumerate(sample_alpha_list):
            seed = 2**i
            np.random.seed(seed)
            torch.manual_seed(seed)
            fed_data = federated_split(
                data,
                flock,
                num_classes=10,
                labels_alpha=l_alpha,
                samples_alpha=s_alpha,
            )
            logging.info(f"Data is federated with {l_alpha=} and {s_alpha=}.")
            for strategy in strategies:
                res = train_experiment(strategy, flock, fed_data, config)
                res["labels_alpha"] = l_alpha
                res["samples_alpha"] = s_alpha
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
    worker_nodes = 100  # 000
    main(
        num_global_rounds=5,  # 200,
        num_worker_nodes=worker_nodes,
        # labels_alpha=0.1,
        # samples_alpha=1000.0,  # 1.0,
        max_workers=10,
        participation=0.1,
        # participation=0.01,
        # participation=0.1,  # Param for sync Strategies
        alpha=1 / worker_nodes,  # FedAsync Param
    )
    caffeine.off()
