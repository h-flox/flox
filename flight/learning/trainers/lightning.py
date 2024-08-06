import lightning.pytorch as PL


class LightningTrainer:
    def __init__(self, *args, **kwargs) -> None:
        self.trainer = PL.Trainer(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.trainer.fit(*args, **kwargs)
        _ = self.trainer.logged_metrics
        return []
