import torch
import torch.nn as nn
import numpy as np


class ConvNetwork(nn.Module):
    """Generative network for tiles"""
    def __init__(self, dimensions, hidden_size):
        super(ConvNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.dims = dimensions

        base = 16
        self.base = base
        self.internal = 3136

        self.conv1 = nn.Conv2d(3, base, 3, stride=2)
        self.conv2 = nn.Conv2d(base, 2*base, 2)
        self.fc1 = nn.Linear(self.internal, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.internal)
        self.deconv1 = nn.ConvTranspose2d(2*base, base, 2)
        self.deconv2 = nn.ConvTranspose2d(base, 3, 4, stride=2)

    def forward(self, images=None):
        images = images.permute(0, 3, 1, 2)
        x = torch.tanh(self.conv1(images))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.fc1(x.view((-1, self.internal))))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.deconv1(x.view((-1, 2*self.base, 14, 14))))
        x = torch.tanh(self.deconv2(x))

        x = x.view((-1, self.dims[2], self.dims[0], self.dims[1]))
        x = x.permute((0, 2, 3, 1))

        return x



class Network(nn.Module):
    """Generative network for tiles"""
    def __init__(self, in_size, hidden_size):
        super(Network, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        coef = 5

        self.fc1 = nn.Linear(in_size, coef*hidden_size)
        self.fc2 = nn.Linear(coef*hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, coef*hidden_size)
        self.fc4 = nn.Linear(coef*hidden_size, in_size)

    def forward(self, images=None):
        if images is None:
            x = torch.rand((1, self.hidden_size))
        else:
            images = images.view(-1, self.in_size)
            x = torch.tanh(self.fc1(images))
            x = torch.tanh(self.fc2(x))

        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return x
