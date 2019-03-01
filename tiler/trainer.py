import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .net import Network


class Trainer:
    """Train net on images"""
    def __init__(self, images, config):
        self._init_net(config)
        self.config = config
        self.images = self._normalize(np.stack(images))
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

    def train(self):
        train_num = len(self.images) - self.config['validation_size']

        for e in range(self.config['epochs']):
            self.optimizer.zero_grad()

            indices = np.random.choice(train_num, self.config['batch_size'])
            image_batch = self._prepare_images(self.images[indices])
            restored = self.net(image_batch)

            loss = self.loss(restored.view((-1, *self.config['dimensions'])), image_batch)
            loss.backward()
            self.optimizer.step()
            print(e, loss.item())

        self._save_net()

    def validate(self):
        image_batch = self._prepare_images(self.images[-self.config['validation_size']:])
        restored = self.net(image_batch)
        loss = self.loss(restored.view((-1, *self.config['dimensions'])), image_batch)
        print('validation loss', loss.item())
        self._show(image_batch, restored.detach().view((-1, *self.config['dimensions'])))

    def _init_net(self, config):
        in_size = np.prod(config['dimensions'])
        self.net = Network(in_size, config['hidden_size'])
        if config['restore']:
            self.net.load_state_dict(torch.load(config['model_file']))
            self.net.eval()

    def _save_net(self):
        torch.save(self.net.state_dict(), self.config['model_file'])

    def _prepare_images(self, images):
        image_batch = torch.tensor(images, dtype=torch.float)
        return image_batch

    def _show(self, source, result):
        source = torch.clamp(self._denormalize(source) / 255, 0, 1)
        result = torch.clamp(self._denormalize(result) / 255, 0, 1)
        num = len(source)
        plt.figure(figsize=(22, 6))
        for i in range(num):
            plt.subplot(2, num, i+1)
            plt.imshow(source[i])
            plt.subplot(2, num, num+i+1)
            plt.imshow(result[i])
        plt.show()

    def _normalize(self, images):
        self.norm = {'mean': np.mean(images), 'variation': 2*np.std(images)}
        return (images - self.norm['mean']) / self.norm['variation']

    def _denormalize(self, images):
        return images * self.norm['variation'] + self.norm['mean']
