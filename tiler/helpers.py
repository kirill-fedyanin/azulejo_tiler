import torch
import matplotlib.pyplot as plt


def normalize(images, config):
    return (images - config['mean']) / config['variation']


def denormalize(images, config):
    rescaled = images * config['variation'] + config['mean']
    return torch.clamp(rescaled/255, 0, 1)


def show(source, result, config):
    source = denormalize(source, config)
    result = denormalize(result, config)
    plt.figure(figsize=(22, 6))
    num = config['show_size']
    for i in range(num):
        plt.subplot(2, num, i+1)
        plt.imshow(source[i])
        plt.subplot(2, num, num+i+1)
        plt.imshow(result[i])
    plt.show()
