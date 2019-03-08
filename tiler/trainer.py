import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .net import Network, ConvNetwork, GeneratorNet, DiscriminatorNet
from .helpers import normalize, denormalize


class Trainer:
    def __init__(self, images, config):
        self.config = config
        self.images = normalize(np.stack(images), config)

    def train(self):
        # net_generator = GeneratorNet()
        net_discriminator = DiscriminatorNet()

        # optimizer_generator = torch.optim.Adam(net_generator.parameters(), lr=2e-4, betas=(.5, .999))
        optimizer_discriminator = torch.optim.Adam(net_discriminator.parameters(), lr=2e-4, betas=(.5, .999))

        criterion = nn.BCELoss()
        train_num = len(self.images) - self.config['validation_size']

        for e in range(self.config['epochs']):
            indices = np.random.choice(train_num, self.config['batch_size'])
            image_batch = torch.tensor(self.images[indices], dtype=torch.float)

            optimizer_discriminator.zero_grad()
            real_img_evaluation = net_discriminator(image_batch)
            ones = torch.ones_like(real_img_evaluation)
            real_img_loss = criterion(real_img_evaluation, ones)

            fake_images = 2*torch.rand_like(image_batch) - 1
            fake_img_evaluation = net_discriminator(fake_images)
            zeros = torch.zeros_like(fake_img_evaluation)
            fake_img_loss = criterion(fake_img_evaluation, zeros)

            print('---')
            print(real_img_evaluation)
            print(fake_img_evaluation)

            loss = real_img_loss + fake_img_loss
            loss.backward()
            optimizer_discriminator.step()

    def validate(self):
        pass


class InitialTrainer:
    """Train autoencoder on images"""
    def __init__(self, images, config):
        self._init_net(config)
        self.config = config
        self.images = normalize(np.stack(images), config)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

    def train(self):
        train_num = len(self.images) - self.config['validation_size']

        for e in range(self.config['epochs']):
            self.optimizer.zero_grad()

            indices = np.random.choice(train_num, self.config['batch_size'])
            image_batch = self._prepare_images(self.images[indices])
            restored = self.net(image_batch)

            loss = self.criterion(restored.view((-1, *self.config['dimensions'])), image_batch)
            loss.backward()
            self.optimizer.step()
            print(e, loss.item())

        self._save_net()

    def validate(self):
        image_batch = self._prepare_images(self.images[-self.config['validation_size']:])
        restored = self.net(image_batch)
        loss = self.loss(restored.view((-1, *self.config['dimensions'])), image_batch)
        print('validation loss', loss.item())
        self._show(image_batch, restored.detach().view((-1, *self.config['dimensions'])), self.config['show_size'])

    def _init_net(self, config):
        if config['convolution']:
            self.net = ConvNetwork(config['dimensions'], config['hidden_size'])
        else:
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

    def _show(self, source, result, num):
        source = denormalize(source, self.config)
        result = denormalize(result, self.config)
        plt.figure(figsize=(22, 6))
        for i in range(num):
            plt.subplot(2, num, i+1)
            plt.imshow(source[i])
            plt.subplot(2, num, num+i+1)
            plt.imshow(result[i])
        plt.show()
