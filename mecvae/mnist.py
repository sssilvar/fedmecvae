import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import torch.nn.functional as F

from torchvision.transforms.functional import adjust_brightness, rotate

class MNISTBatchEffectDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='~/data/', batch_size=64, n_batches=4, batch_effect_std=0.3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.batch_effect_std = batch_effect_std

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def brightness_shift(self, data, batch_indices):
        data = data.to(torch.float)
        brightness_shift = torch.normal(torch.tensor(0.0), torch.tensor(self.batch_effect_std), size=(1,))
        data[batch_indices] = torch.clamp(data[batch_indices] + brightness_shift * 255, 0, 255)
        data = data.to(torch.uint8)
        return data

    def brightness_shift(self, data, batch_indices):
        data = data.to(torch.float)
        brightness_shift = torch.normal(torch.tensor(0.0), torch.tensor(self.batch_effect_std), size=(1,))
        data[batch_indices] = torch.clamp(data[batch_indices] + brightness_shift * 255, 0, 255)
        data = data.to(torch.uint8)
        return data

    def contrast_scaling(self, data, batch_indices):
        data = data.to(torch.float)
        contrast_scale = 1 + torch.normal(torch.tensor(0.0), torch.tensor(self.batch_effect_std), size=(1,))
        data[batch_indices] = torch.clamp(data[batch_indices] * contrast_scale, 0, 255)
        data = data.to(torch.uint8)
        return data

    def rotation(self, data, batch_indices):
        data = data.to(torch.float)
        angle = torch.normal(torch.tensor(0.0), torch.tensor(self.batch_effect_std), size=(1,)) * 100
        data[batch_indices] = torch.clamp(
            rotate(data[batch_indices].unsqueeze(1), angle.item()).squeeze(1), 0, 255
        )
        data = data.to(torch.uint8)
        return data


    def apply_batch_effect(self, data, bias_type='brightness_shift'):
        batch_assignments_tensor = torch.tensor(self.batch_assignments)
        for b in range(self.n_batches):
            batch_indices = (batch_assignments_tensor == b).nonzero(as_tuple=True)
            
            if bias_type == 'brightness_shift':
                data = self.brightness_shift(data, batch_indices)
            elif bias_type == 'contrast_scaling':
                data = self.contrast_scaling(data, batch_indices)
            elif bias_type == 'rotation':
                data = self.rotation(data, batch_indices)
            elif bias_type == 'translation':
                data = self.translation(data, batch_indices)
            else:
                raise ValueError(f"Invalid bias_type: {bias_type}")
        
        return data

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def _get_one_hot(self, labels, num_classes):
        return F.one_hot(labels, num_classes=num_classes).to(torch.float)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.batch_assignments = np.random.choice(np.arange(self.n_batches), len(train_dataset))
            train_dataset.data = self.apply_batch_effect(train_dataset.data)
            train_dataset.targets = self._get_one_hot(train_dataset.targets, num_classes=10)
            self.batch_one_hot = self._get_one_hot(torch.tensor(self.batch_assignments), num_classes=self.n_batches)
            self.train_dataset = train_dataset

            # Store batch_assignments for each digit
            self.digit_batch_assignments = {}
            for digit in range(10):
                digit_indices = np.where(train_dataset.targets.argmax(dim=1) == digit)[0]
                self.digit_batch_assignments[digit] = self.batch_assignments[digit_indices]

        if stage == 'test' or stage is None:
            test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = test_dataset

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.cat([self.train_dataset.data.view(-1, 28 * 28), self.train_dataset.targets], dim=1),
            self.batch_one_hot
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


import matplotlib.pyplot as plt

def visualize_samples_and_histograms(data_module, digit, n_rows=2, n_cols=None):
    if n_cols is None:
        n_cols = data_module.n_batches

    # Get the digit_indices for the specified digit
    digit_indices = np.where(data_module.train_dataset.targets == digit)[0]

    # Get the batch_assignments for the samples of the specified digit
    digit_batch_assignments = data_module.digit_batch_assignments[digit]

    fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(2 * n_cols * 3, n_rows * 3))

    for batch in range(n_cols):
        # Find the indices of the samples within the current batch
        batch_indices = np.where(digit_batch_assignments == batch)

        # Choose a random sample index from the batch
        sample_index = np.random.choice(digit_indices[batch_indices])

        # Get the sample image and plot it
        img, _ = data_module.train_dataset[sample_index]
        img = img.squeeze().numpy()
        print(img.min(), img.max())

        axes[0, 2 * batch].imshow(img, cmap='gray', vmin=0)
        axes[0, 2 * batch].set_title(f"Batch {batch}")

        # Plot the intensity histogram for the current batch
        batch_intensity_values = data_module.train_dataset.data[digit_indices[batch_indices]].numpy().ravel()
        axes[0, 2 * batch + 1].hist(batch_intensity_values, bins=30, density=True,range=[0,255])
        axes[0, 2 * batch + 1].set_title(f"Batch {batch} Intensity Histogram")

    plt.tight_layout()
    plt.show()
