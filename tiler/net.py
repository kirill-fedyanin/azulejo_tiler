import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    """Generative network for tiles"""
    def __init__(self, in_size, hidden_size):
        super(Network, self).__init__()
        self.in_size = in_size

        coef = 5

        self.fc1 = nn.Linear(in_size, coef*hidden_size)
        self.fc2 = nn.Linear(coef*hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, coef*hidden_size)
        self.fc4 = nn.Linear(coef*hidden_size, in_size)

    def forward(self, images):
        images = images.view(-1, self.in_size)

        x = torch.tanh(self.fc1(images))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x
