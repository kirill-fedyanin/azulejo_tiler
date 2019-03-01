import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tiler import Generator, Trainer

TILE_SIZE = (32, 32)
DATA_DIR = "data/"
CACHE_DIR = "preprocessed/"


def generate_tile():
    """Generate new random tile"""
    config = {
        'dimensions': (TILE_SIZE[0], TILE_SIZE[1], 3),
        'hidden_size': 64,
        'model_file': 'model/temp.pt'
    }

    generator = Generator(config)
    num = 10
    for i in range(num):
        image = generator.generate()
        plt.subplot(1, num, i+1)
        plt.imshow(image)
    print(image)
    plt.show()






def train_tiler():
    """Train tiler on given images"""
    config = {
        'dimensions': (TILE_SIZE[0], TILE_SIZE[1], 3),
        'hidden_size': 64,
        'batch_size': 16,
        'epochs': 1,
        'lr': 1e-3,
        'validation_size': 16,
        'model_file': 'model/temp.pt',
        'restore': True
    }
    images = _load_images()
    trainer = Trainer(images, config)
    trainer.train()
    trainer.validate()


def _load_images():
    images = []
    for image_name in os.listdir(CACHE_DIR):
        image = np.array(Image.open(CACHE_DIR + image_name))
        images.append(image)
    return images


if __name__ == '__main__':
    # train_tiler()
    generate_tile()
