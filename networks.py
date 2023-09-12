import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd

# Basic fully connected neural network with MSE loss
class NN4vi(pl.LightningModule):
    """
     Creates a fully connected neural network
    :param input_dim: int, dimension of input data
    :param hidden_widths: list of size of hidden widths of network
    :param output_dim: dimension of output
    :param activation: activation function for hidden layers (defaults to ReLU)
    :return: A network
     """


    def __init__(self,
                 input_dim: int,
                 hidden_widths: list,
                 output_dim: int,
                 activation = nn.ReLU):
        super().__init__()
        structure = [input_dim] + list(hidden_widths) + [output_dim]
        layers = []
        for j in range(len(structure) - 1):
            act = activation if j < len(structure) - 2 else nn.Identity
            layers += [nn.Linear(structure[j], structure[j + 1]), act()]

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = nn.MSELoss()(y_hat, y)
        # Logging to TensorBoard by default
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.MSELoss()(self.net(x), y)
        # self.log('val_loss', loss)
        return loss

# Basic fully connected neural network with cross-entropy loss
class BinClass(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network with binary cross-entropy loss and ReLU activation
    :param p: int, dimension of input data
    :param hidden_width: int, size of hidden width of network

    :return: A network
     """
    def __init__(self, p=4, hidden_width=50):
        super().__init__()

        self.layer_1 = nn.Linear(p, hidden_width)
        self.layer_2 = nn.Linear(hidden_width, 1)


    def forward(self, x):
        # Pass the tensor through the layers
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)

        # Softmax the values to get a probability
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Pass through the forward function of the network
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Very very simple network for MNIST classification
class UltraLiteMNIST(pl.LightningModule):
    """
     Creates a fully connected 2-layer neural network for images with binary cross-entropy loss and ReLU activation
    :param hidden_width: int, size of hidden width of network
    :param weight_decay: float, l2 regularization parameter

    :return: A network
     """
    def __init__(self, hidden_width=4, weight_decay=0):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, hidden_width)
        self.layer_3 = nn.Linear(hidden_width, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.tensor(x.view(batch_size, -1), dtype=torch.float32)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = nn.BCELoss()(logits, y)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)

# Slightly less simple network for MNIST classification
class LiteMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        logits = self(x)
        loss = nn.BCELoss()(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        probs = self(x)
        loss = nn.BCELoss()(probs, y)
        y_hat = torch.round(probs)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)