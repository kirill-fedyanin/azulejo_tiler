import torch


def normalize(images, config):
    return (images - config['mean']) / config['variation']


def denormalize(images, config):
    rescaled = images * config['variation'] + config['mean']
    return torch.clamp(rescaled/255, 0, 1)
