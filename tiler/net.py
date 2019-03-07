import torch
import torch.nn as nn
import numpy as np


class ConvNetwork(nn.Module):
    """Generative network for tiles"""
    def __init__(self, dims, hidden_size):
        super(ConvNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.dims = dims

        base = 16
        KERNEL_1 = 4
        KERNEL_2 = 2
        self.base = base
        self.conv_side = (dims[0] - KERNEL_1 + 1) - KERNEL_2 + 1

        self.internal = 2 * self.base * self.conv_side**2

        self.conv1 = nn.Conv2d(3, base, KERNEL_1)
        self.conv2 = nn.Conv2d(base, 2*base, KERNEL_2)
        self.fc1 = nn.Linear(self.internal, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.internal)
        self.deconv1 = nn.ConvTranspose2d(2*base, base, KERNEL_2)
        self.deconv2 = nn.ConvTranspose2d(base, 3, KERNEL_1)

    def forward(self, images=None):
        if images is None:
            x = 2*torch.rand((1, self.hidden_size)) - 1
        else:
            images = images.permute(0, 3, 1, 2)
            x = torch.tanh(self.conv1(images))
            x = torch.tanh(self.conv2(x))
            x = torch.tanh(self.fc1(x.view((-1, self.internal))))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.deconv1(x.view((-1, 2*self.base, self.conv_side, self.conv_side))))
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
            x = 2*torch.rand((1, self.hidden_size)) - 1
        else:
            images = images.view(-1, self.in_size)
            x = torch.tanh(self.fc1(images))
            x = torch.tanh(self.fc2(x))

        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return x
