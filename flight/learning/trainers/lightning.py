import lightning.pytorch as L


class LightningTrainer:
    def __init__(self, *args, **kwargs) -> None:
        self.trainer = L.Trainer(*args, **kwargs)

    def fit(self, *args, **kwargs):
        results = self.trainer(*args, **kwargs)
        return results
