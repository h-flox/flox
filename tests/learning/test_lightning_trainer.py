# NOTE: Commented out because it kept creating a `lightning_logs` directory in the root.
#       When we re-integrate Lightning as a feature, use a *temporary directory*.
#
#
# from lightning.pytorch import LightningModule, LightningDataModule
#
# from flight.learning.lightning import LightningTrainer
# from testing.fixtures import lightning_data_module, worker_node, lightning_module
#
#
# def test_lightning_trainer(worker_node, lightning_module, lightning_data_module):
#     """
#     Tests a basic setup of using the `LightningTrainer` class for
#     PyTorch-based models.
#     """
#     kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
#     trainer = LightningTrainer(worker_node, **kwargs)
#
#     assert isinstance(lightning_module, LightningModule)
#     assert isinstance(trainer, LightningTrainer)
#     assert isinstance(lightning_data_module, LightningDataModule)
#
#     results = trainer.fit(lightning_module, lightning_data_module)
#     assert isinstance(results, dict)
#
#
# def test_test_process(worker_node, lightning_module, lightning_data_module):
#     """
#     Tests that no errors occur during basic testing.
#     """
#     kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
#     trainer = LightningTrainer(worker_node, **kwargs)
#
#     trainer.fit(lightning_module, lightning_data_module)
#     records = trainer.test(lightning_module, lightning_data_module)
#
#     assert isinstance(records, list)
#
#
# def test_val_process(worker_node, lightning_module, lightning_data_module):
#     """
#     Tests that no errors occur during basic validation.
#     """
#     kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
#     trainer = LightningTrainer(worker_node, **kwargs)
#
#     trainer.fit(lightning_module, lightning_data_module)
#     records = trainer.validate(lightning_module, lightning_data_module)
#
#     assert isinstance(records, list)
