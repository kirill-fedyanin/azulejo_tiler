from .net import Network


class Trainer:
    """Train net on images"""
    def __init__(self, images, config):
        self.net = Network()
        self.config = config
        self.images = images

    def train(self):
        print(len(self.images))

    def validate(self):
        pass
