class TestValidFedSGD:
    def test_fedsgd_class_hierarchy(self):
        """
        Test that the associated node strategy types follow the correct protocols.

        TODO: Re-implement from scratch.
        """
        # fedsgd = FedSGD(1, False, True)
        #
        # assert isinstance(fedsgd.aggr_strategy, (AggrStrategy, FedSGDAggr))
        # assert isinstance(fedsgd.coord_strategy, (CoordStrategy, FedSGDCoord))
        # assert isinstance(
        #     fedsgd.trainer_strategy, (TrainerStrategy, DefaultTrainerStrategy)
        # )
        # assert isinstance(
        #     fedsgd.worker_strategy, (WorkerStrategy, DefaultWorkerStrategy)
        # )

    def test_default_fedsgd_coord(self):
        """
        Tests the usability of the coordinator strategy for 'FedSGD'

        TODO: Re-implement from scratch.
        """
        # fed_sgd = FedSGD(1, False, True)
        # coord_strategy: CoordStrategy = fed_sgd.coord_strategy
        # gen = default_rng()
        # workers = create_children(num_workers=10)
        #
        # selected = coord_strategy.select_workers("foo", workers, gen)
        #
        # for worker in workers:
        #     assert worker in selected

    def test_fedsgd_aggr(self):
        """
        Tests the usability of the aggregator strategy for 'FedSGD'

        TODO: Re-implement from scratch.
        """
        # fed_sgd = FedSGD(1, False, True)
        # aggr_strategy: AggrStrategy = fed_sgd.aggr_strategy
        #
        # state = "foo"
        # children = {1: "foo1", 2: "foo2"}
        #
        # children_state_dicts_pt = {
        #     1: {
        #         "train/loss": torch.tensor(2.3, dtype=torch.float32),
        #         "train/acc": torch.tensor(1.2, dtype=torch.float32),
        #     },
        #     2: {
        #         "train/loss": torch.tensor(3.1, dtype=torch.float32),
        #         "train/acc": torch.tensor(1.4, dtype=torch.float32),
        #     },
        # }
        #
        # avg = aggr_strategy.aggregate_params(state, children, children_state_dicts_pt)
        #
        # assert isinstance(avg, dict)
        #
        # expected_avg = {
        #     "train/loss": 2.7,
        #     "train/acc": 1.3,
        # }
        #
        # epsilon = 1e-6
        # for key, value in avg.items():
        #     expected = expected_avg[key]
        #     assert abs(expected - value.item()) < epsilon
