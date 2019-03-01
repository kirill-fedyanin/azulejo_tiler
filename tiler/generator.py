import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .net import Network
from .helpers import normalize, denormalize


class Generator:
    """Generate new random tiles"""
    def __init__(self, config):
        self.config = config
        self._init_net(config)

    def _init_net(self, config):
        in_size = np.prod(config['dimensions'])
        self.net = Network(in_size, config['hidden_size'])
        self.net.load_state_dict(torch.load(config['model_file']))
        self.net.eval()

    def generate(self):
        image = self.net().detach().view(self.config['dimensions'])
        return denormalize(image, self.config)
