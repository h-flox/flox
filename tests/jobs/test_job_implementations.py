from flox.federation.jobs.aggr import AggregateJob, DebugAggregateJob
from flox.federation.jobs.protocols import AggrJob, TrainJob
from flox.federation.jobs.train import DebugLocalTrainJob, LocalTrainJob


def test_protocol_implementations():
    assert isinstance(AggregateJob(), AggrJob)
    assert isinstance(DebugAggregateJob(), AggrJob)
    assert isinstance(LocalTrainJob(), TrainJob)
    assert isinstance(DebugLocalTrainJob(), TrainJob)
