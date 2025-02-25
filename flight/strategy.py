from __future__ import annotations

import typing as t


class AggregationPolicy(t.Protocol):
    def __call__(*args, **kwargs):
        pass


class WorkerSelectionPolicy(t.Protocol):
    def __call__(*args, **kwargs):
        pass


class Strategy:
    def __init__(self):
        pass
