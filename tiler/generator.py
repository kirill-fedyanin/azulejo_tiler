import torch

from .net import GeneratorNet
from .helpers import denormalize


class Generator:
    """Generate new random tiles"""
    def __init__(self, config):
        self.config = config
        self._init_net(config)

    def _init_net(self, config):
        self.net = GeneratorNet()
        self.net.load_state_dict(torch.load(config['g_model_file']))
        self.net.eval()

    def generate(self):
        image = self.net().detach().view(self.config['dimensions'])
        return denormalize(image, self.config)
