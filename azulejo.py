import os

from PIL import Image
import numpy as np

from tiler import Generator, Trainer

TILE_SIZE = (32, 32)
DATA_DIR = 'data/'
CACHE_DIR = 'preprocessed/'


def generate_tile():
    """Generate new random tile"""


def train_tiler():
    """Train tiler on given images"""
    config = {}
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
    train_tiler()

