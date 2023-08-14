# Create a conditional variational autoencoder (CVAE) model using pytorch lightning.
# The model shoud identify automatically which columns contain categorical variables and compute a BCE loss on it.
# Number of layers and activation functions should be configurable for further optimization.

import pytorch_lightning as pl
import torch
import torch.nn as nn


def _repameterize(mu, logvar):
    """Sample from latent space using the re-parameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class LitFlexCVAE(pl.LightningModule):
    """A more flexible implementation of a multilayer CVAE."""

    def __init__(self, data_dim: int, conditioning_dim: int, z_dim=64, hidden_dim=[256, 128], lr=1e-3,
                 activation=nn.ReLU(), optimizer='adam'):
        super().__init__()
        self.save_hyperparameters(ignore=['activation'])
        self.data_dim = data_dim
        self.conditioning_dim = conditioning_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.optimizer = optimizer

        # Create the conditional encoder using the add_module method
        self.encoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip([data_dim + conditioning_dim] + hidden_dim[:-1], hidden_dim)):
            self.encoder.add_module(f"layer_{i}", nn.Linear(in_dim, out_dim))
            self.encoder.add_module(f"activation_{i}", activation)
            # self.encoder.add_module(f"batchnorm_{i}", nn.BatchNorm1d(out_dim))
            # self.encoder.add_module(f"dropout_{i}", nn.Dropout(0.2))
        self.fc_mean = nn.Linear(hidden_dim[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dim[-1], z_dim)

        # Create the conditional decoder
        self.decoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(
                zip([z_dim + conditioning_dim] + hidden_dim[::-1][:-1], hidden_dim[::-1])):
            self.decoder.add_module(f"layer_{i}", nn.Linear(in_dim, out_dim))
            self.decoder.add_module(f"activation_{i}", activation)
            # self.decoder.add_module(f"batchnorm_{i}", nn.BatchNorm1d(out_dim))
            # self.decoder.add_module(f"dropout_{i}", nn.Dropout(0.2))
        self.decoder.add_module("final_layer", nn.Linear(hidden_dim[::-1][-1], data_dim))

        # Initialize the weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights in each layer using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        # Concatenate the data and conditioning
        x = torch.cat([x, y], dim=1)

        # Encode the data
        x = self.encoder(x)

        # Compute the latent space
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = _repameterize(mu, logvar)

        # Concatenate the latent space and conditioning
        z = torch.cat([z, y], dim=1)

        # Decode the latent space
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Compute the loss function."""
        # Identify the categorical variables (they're all one-hot encoded) so basically all columns with unique
        # values <= 2 This is a very naive approach and should be improved Get a mask for categorical variables
        mask = torch.tensor([x[:, i].unique().numel() <= 2 for i in range(x.shape[1])]).to(self.device)

        # Compute the MSE loss for continuous variables relying on the mas.x is a tensor so it cannot be indexed as a
        # DataFrame.
        mse_loss = nn.MSELoss(reduction="sum")
        mse = mse_loss(recon_x[:, ~mask], x[:, ~mask])
        mse = mse / x.shape[0]  # Normalize the MSE loss

        # Compute the BCE loss for categorical variables
        bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        bce = bce_loss(recon_x[:, mask], x[:, mask])
        bce = bce / x.shape[0]  # Normalize the BCE loss

        # Compute the KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return {'loss': mse + bce + kl, 'mse': mse, 'bce': bce, 'kl': kl}

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self(x, y)
        losses = self.loss_function(recon_x, x, mu, logvar)

        # Logg all losses
        for loss_name, loss_value in losses.items():
            self.log(f'Train/{loss_name}', loss_value, prog_bar=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self(x, y)
        losses = self.loss_function(recon_x, x, mu, logvar)

        # Logg all losses
        for loss_name, loss_value in losses.items():
            self.log(f'Validation/{loss_name}', loss_value, prog_bar=True)

        return losses['loss']

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer {self.optimizer}')

        # Add a scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Validation/loss'}
