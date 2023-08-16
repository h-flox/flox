from depr.aggregator.asynch.base import AsynchAggregatorLogic


class SimpleAsynchAggregatorLogic(AsynchAggregatorLogic):
    def __init__(self):
        super().__init__()

    def on_model_broadcast(self):
        pass

    def on_model_receive(self):
        pass

    def on_model_aggregate(
        # self, global_module, worker_id, update, update_history
        self,
        global_module,
        worker_update,
    ):
        avg_weights = {}
        for name, param in global_module.state_dict().items():
            avg_weights[name] = (param + worker_update[name]) / 2
        return avg_weights

    def on_model_evaluate(self):
        pass
