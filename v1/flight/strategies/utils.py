from v1.flight.strategies import DefaultStrategy, Strategy


def load_strategy(strategy_name: str, **kwargs) -> Strategy:
    """Function used to grab the users preferred 'Strategy'.

    Args:
        strategy_name (str): The name of the 'Strategy' to be grabbed.
        **kwargs: Additional keyword arguments to be passed through to the 'Strategy'.

    Raises:
        ValueError: If an unknown 'Strategy' type is passed through.

    Returns:
        Strategy: The selected 'Strategy' type.
    """
    assert isinstance(strategy_name, str), "`strategy_name` must be a string."
    match strategy_name.lower():
        case "default":
            return DefaultStrategy()

        case "fedasync" | "fed-async":
            from v1.flight.strategies.impl.fedasync import FedAsync

            return FedAsync(**kwargs)

        case "fedavg" | "fed-avg":
            from v1.flight.strategies.impl.fedavg import FedAvg

            return FedAvg(**kwargs)

        case "fedprox" | "fed-prox":
            from v1.flight.strategies.impl.fedprox import FedProx

            return FedProx(**kwargs)

        case "fedsgd" | "fed-sgd":
            from v1.flight.strategies.impl.fedsgd import FedSGD

            return FedSGD(**kwargs)
        case _:
            raise ValueError(f"Strategy '{strategy_name}' is not recognized.")
