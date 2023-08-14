import os
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import MeanSquaredError

from .model import CVAE
from .data import simulate_brain_measures

import pandas as pd
import patsy


class LitCVAE(pl.LightningModule):
    def __init__(self, num_dim, cat_dim, n_batches, hidden_dim=128, z_dim=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.cvae = CVAE(num_dim, cat_dim, n_batches, z_dim=z_dim, hidden_dim=hidden_dim)
        self.lr = lr

    def forward(self, x, y):
        return self.cvae(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self.cvae(x, y)
        losses = self.cvae.loss_function(recon_x, x, mu, logvar, as_dict=True)

        for key, val in losses.items():
            prog_bar = key == 'Loss'
            self.log(f"Train/{key}", val, prog_bar=prog_bar)

        return losses['Loss']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self.cvae(x, y)
        losses = self.cvae.loss_function(recon_x, x, mu, logvar, as_dict=True)

        for key, val in losses.items():
            prog_bar = key == 'Loss'
            self.log(f"Validation/{key}", val, prog_bar=prog_bar)

        return losses['Loss']

    def test_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self.cvae(x, y)
        losses = self.cvae.loss_function(recon_x, x, mu, logvar, as_dict=True)

        for key, val in losses.items():
            prog_bar = key == 'Loss'
            self.log(f"Test/{key}", val, prog_bar=prog_bar)

        return losses['Loss']

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        recon_x = self.cvae.predict(x, y=None, n_samples=30)
        assert recon_x.shape == x.shape
        return recon_x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Validation/Loss",
            },
        }


class BrainMeasuresDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class BrainMeasuresDataModule(pl.LightningDataModule):
    def __init__(self, n_samples, n_features, n_batches, batch_size=32):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_batches = n_batches
        self.batch_size = batch_size

    def prepare_data(self):
        self.df, self.df_unbiased = simulate_brain_measures(self.n_samples, self.n_features, self.n_batches,
                                                            random_state=21)

        # Standardize numerical variables
        numerical_vars = self.df.columns[:self.n_features].tolist() + ['Age', 'Scanner']
        standardized_formula = " + ".join([f'standardize({var})' for var in numerical_vars])

        # One-hot encode categorical variables
        categorical_vars = ['Sex']
        one_hot_formula = " + ".join([f'C({var}, Treatment)' for var in categorical_vars])

        # Create the full design matrix using patsy
        design_formula = f'C(Batch) ~ {standardized_formula} + {one_hot_formula} - 1'
        self.batch_design, self.design_matrix = patsy.dmatrices(design_formula, data=self.df, return_type='dataframe')

        # Reorder variables (patsy way of ordering is not convenient)
        columns_to_keep = standardized_formula.split(' + ')
        columns_to_move = self.design_matrix.columns.difference(columns_to_keep).tolist()

        self.num_dim = len(columns_to_keep)
        self.cat_dim = len(columns_to_move)

        self.design_matrix = self.design_matrix[columns_to_keep + columns_to_move]

        self.X = torch.tensor(self.design_matrix.values, dtype=torch.float32)
        self.y = torch.tensor(self.batch_design.values, dtype=torch.long)

    def setup(self, stage=None):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, stratify=self.y, test_size=0.8)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

        self.full_dataset = BrainMeasuresDataset(torch.tensor(self.design_matrix.values, dtype=torch.float32),
                                                 torch.tensor(self.batch_design.values, dtype=torch.long))
        self.train_dataset = BrainMeasuresDataset(X_train, y_train)
        self.val_dataset = BrainMeasuresDataset(X_val, y_val)
        self.test_dataset = BrainMeasuresDataset(X_test, y_test)

        # Create DataFrames for checking
        self.df_test = pd.DataFrame(X_test[:, :self.n_features], columns=self.df.columns[:self.n_features]).join(
            self.df.iloc[:, self.n_features:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size)


# Now lwt's create a DataModule for real data that takes a CSV file containing FreeSurfer measure and covariates as input and performs
# the same preprocessing steps as above as well as the train/val/test split

class RealBrainMeasuresDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=32, cat_cols=('Sex', 'DX')):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.cat_cols = list(cat_cols)

        # verify file exists
        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError(f"{self.csv_file} not found!")

    def prepare_data(self):
        # read csv file
        print(f"Reading {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file, index_col=0)

        # Create design matrices using patsy.dmatrices:
        # Y for site and X with standardized numerical variables and one-hot encoded categorical variables
        # Get categorical variables
        cat_cols = self.cat_cols
        batch_col = 'site'
        num_cols = [col for col in self.df.columns if col not in cat_cols + [batch_col]]
        self.n_num_cols = len(num_cols)
        self.n_cat_cols = len(cat_cols)
        print(f"Number of numerical variables: {self.n_num_cols=}")
        print(f"Number of categorical variables: {self.n_cat_cols=}")

        # Create formuyla for patsy
        formula = 'site ~ '
        for col in num_cols:
            formula += f'standardize(Q("{col}")) + '
        for col in cat_cols:
            formula += f'C({col}) + '
        formula = formula[:-3]  # remove last ' + '
        formula += ' - 1'  # remove intercept (identifiability)
        # print(f"Formula: {formula}")

        # Create design matrices
        self.y, self.X = patsy.dmatrices(formula, data=self.df, return_type='dataframe')
        print(f"Number of batches: {self.y.shape[1]}")

        self.n_features = self.X.shape[1]
        self.n_batches = self.y.shape[1]

    def setup(self, stage=None):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, stratify=self.y, test_size=0.8)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

        self.full_dataset = BrainMeasuresDataset(torch.tensor(self.X.values, dtype=torch.float32),
                                                 torch.tensor(self.y.values, dtype=torch.long))
        self.train_dataset = BrainMeasuresDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                                  torch.tensor(y_train.values, dtype=torch.long))
        self.val_dataset = BrainMeasuresDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                                torch.tensor(y_val.values, dtype=torch.long))
        self.test_dataset = BrainMeasuresDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                                 torch.tensor(y_test.values, dtype=torch.long))

        # Create DataFrames for checking
        self.df_test = pd.DataFrame(X_test, columns=self.X.columns).join(self.df.iloc[:, self.n_features:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
