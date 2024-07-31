import argparse
import datetime
import json
import logging
import pickle
import warnings
from pathlib import Path

logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import os

    import torchvision.transforms as transforms
    from models import *
    from torchvision.datasets import FashionMNIST

    import flox
    from flox.data import federated_split
    from flox.federation.topologies import hierarchical_topology


def load_data() -> FashionMNIST:
    return FashionMNIST(
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


def main(**kwargs):
    config = argparse.Namespace(**kwargs)

    aggr_shapes = [None, (2,), (4,), (6,), (8,), (10,)]
    #     create_hierarchical_flock(config.num_worker_nodes, None),
    #     create_hierarchical_flock(config.num_worker_nodes, [2]),
    #     create_hierarchical_flock(config.num_worker_nodes, [4]),
    #     create_hierarchical_flock(config.num_worker_nodes, [6]),
    #     create_hierarchical_flock(config.num_worker_nodes, [8]),
    #     create_hierarchical_flock(config.num_worker_nodes, [10]),
    # ]
    data = load_data()
    logging.info("FashionMNIST data is loaded.")

    iid = 1000.0
    non_iid = 1.0
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split(".")[0]
    out_dir = Path(f"experiments/topo_comparison/{timestamp}/")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for shape in aggr_shapes:
        logging.info(f"Creating and using flock with aggr_shape {shape}.")
        flock = hierarchical_topology(config.num_worker_nodes, shape)
        fed_data = federated_split(data, flock, 10, iid, iid)
        _, df = flox.federated_fit(
            flock,
            # SmallConvModel(),
            SmallModel(),
            fed_data,
            num_global_rounds=config.num_global_rounds,
            strategy="fedavg",
            kind="sync-v2",
            debug_mode=False,
            launcher_kind="federation",
            launcher_cfg={"max_workers": config.max_workers},
        )

        try:
            df["aggr_shape"] = shape
        except ValueError:
            df["aggr_shape"] = [shape] * len(df)
        df.to_feather(out_dir / f"results_{shape=}.feather")

        with open(out_dir / "config.json", "w") as file:
            json.dump(dict(**kwargs), file)

        with open(out_dir / f"topo_{shape=}.pkl", "wb") as file:
            pickle.dump(flock.topo, file)


if __name__ == "__main__":
    import caffeine

    caffeine.on(display=False)
    worker_nodes = 1000
    main(
        num_global_rounds=2,  # 200,
        num_worker_nodes=worker_nodes,
        max_workers=10,
        participation=0.01,  # Param for sync Strategies
        alpha=1 / worker_nodes,  # FedAsync Param
    )
    caffeine.off()
