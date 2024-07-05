import functools
import typing as t

import pydantic as pyd


@pyd.dataclasses.dataclass(frozen=True, repr=False)
class Strategy:
    coord_strategy: str = pyd.field()
    aggr_strategy: str = pyd.field()
    worker_strategy: str = pyd.field()
    trainer_strategy: str = pyd.field()

    def __iter__(self) -> t.Iterator[tuple[str, t.Any]]:
        yield from (
            ("coord_strategy", self.coord_strategy),
            ("aggr_strategy", self.aggr_strategy),
            ("worker_strategy", self.worker_strategy),
            ("trainer_strategy", self.trainer_strategy),
        )

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def __str__(self) -> str:
        name = self.__class__.__name__
        inner = ", ".join(
            [
                f"{strategy_key}={strategy_value.__class__.__name__}"
                for (strategy_key, strategy_value) in iter(self)
                if strategy_value is not None
            ]
        )
        return f"{name}({inner})"
