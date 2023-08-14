import pytorch_lightning as pl

from mecvae.lit import BrainMeasuresDataModule, LitCVAE

# Instantiate the LitCVAE model and BrainMeasuresDataModule
n_features = 5
n_batches = 4
n_samples = 400
batch_size = 32

data_module = BrainMeasuresDataModule(n_samples, n_features, n_batches, batch_size=batch_size)
data_module.prepare_data()

# After one_hot encoding
lit_cvae = LitCVAE(num_dim=data_module.num_dim, cat_dim=data_module.cat_dim, n_batches=n_batches)

# Train the LitCVAE model using PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=100, logger=False)
trainer.fit(lit_cvae, data_module)