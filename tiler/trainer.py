import torch
import numpy as np
import matplotlib.pyplot as plt

from .net import Network


class Trainer:
    """Train net on images"""
    def __init__(self, images, config):
        in_size = np.prod(config['dimensions'])
        self.net = Network(in_size, config['hidden_size'])
        self.config = config
        self.images = images

    def train(self):
        image_batch = self._prepare_images(self.images[:self.config['batch_size']])
        restored = self.net(image_batch)

        self._show(image_batch, restored.detach().view((-1, *self.config['dimensions'])))


    def validate(self):
        pass

    def _prepare_images(self, images):
        image_batch = torch.tensor(np.stack(images), dtype=torch.float)
        image_batch /= 255
        return image_batch

    def _show(self, source, result):
        num = len(source)
        plt.figure(figsize=(12, 6))
        for i in range(num):
            plt.subplot(2, num, i+1)
            plt.imshow(source[i])
            plt.subplot(2, num, num+i+1)
            plt.imshow(result[i])
        plt.show()
