from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    pass


class WorkerStrategy(t.Protocol):
    def start_work(self):
        pass

    def before_training(self):
        pass

    def after_training(self):
        pass

    def end_work(self):
        pass
