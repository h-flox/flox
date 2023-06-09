import pytorch_lightning as pl

from flox.aggregator.base import AggregatorLogic
from flox.trainer.base import TrainerLogic


def fit(
        endpoint_ids: list[str],
        module: pl.LightningModule,
        aggr_logic: AggregatorLogic,
        trainer_logic: TrainerLogic,
        **kwargs
):
    pass
