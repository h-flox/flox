import typing as t


class AggregatorStrategy(t.Protocol):
    def round_start(self):
        pass

    def aggregate_params(self):
        pass

    def round_end(self):
        pass
