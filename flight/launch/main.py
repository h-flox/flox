import sys
import typing as t

from flight.federation.app import federated_fit

# from src.run.config import ExperimentConfig


# def parse_args_to_config(argv: t.Sequence[str]) -> ExperimentConfig:
#     return ExperimentConfig(model="resnet", num_clients=10)


def main(argv: t.Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    # print("[FLIGHT] Hello, world!")
    federated_fit()
    return 0
    # config = parse_args_to_config(argv)
    # print(config)
    # print("Experiment done!")
    # return 0
