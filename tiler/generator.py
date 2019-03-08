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
        random_vector = torch.rand((1, 100, 1, 1))
        image = self.net(random_vector).detach()
        dims = self.config['dimensions']
        image = image.view((dims[2], dims[0], dims[1])).permute(1, 2, 0)
        return denormalize(image, self.config)
