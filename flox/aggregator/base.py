import lightning as L


class AbstractAggregatorLogic:
    def __init__(self):
        pass

    def on_model_init(self, module: L.LightningModule):
        return module()

    def on_model_broadcast(self):
        pass

    def on_model_receive(self):
        pass

    def on_model_aggregate(self):
        pass

    def on_model_evaluate(self):
        pass
