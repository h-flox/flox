def get_strategy(name: str) -> tuple:
    strategy_map = {
        "fedavg": (None, None),
    }
    if name not in strategy_map:
        raise ValueError("...")  # TODO
    return strategy_map[name]