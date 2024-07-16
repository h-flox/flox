import typing as t


class MetricLogger(t.Protocol):
    def log(self):
        pass

    def log_dict(self):
        pass


class InMemoryRecordLogger:
    pass


class DiscRecordLogger:
    pass
