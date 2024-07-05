import typing as t

if t.TYPE_CHECKING:
    Params: t.TypeAlias = t.Any


class AggrStrategy(t.Protocol):
    def start_round(self):
        pass

    def aggregate_params(self) -> Params:
        pass

    def end_round(self):
        pass
