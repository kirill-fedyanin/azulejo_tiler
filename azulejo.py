import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tiler import Generator, Trainer

TILE_SIZE = (32, 32)
DATA_DIR = "data/"
CACHE_DIR = "preprocessed/"
MEAN = 139.
VARIATION = 152.

parser = argparse.ArgumentParser(description="Tiler generator")
parser.add_argument('--experiment', '-x', default=None)
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--validation-size', type=int, default=32)
parser.add_argument('--show-size', type=int, default=8)
parser.add_argument('--epochs', '-e', type=int, default=1)
parser.add_argument('--restore-model', default=False, action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

config = {
    'dimensions': (TILE_SIZE[0], TILE_SIZE[1], 3),
    'hidden_size': args.hidden_size,
    'batch_size': args.batch_size,
    'show_size': args.show_size,
    'epochs': args.epochs,
    'lr': args.lr,
    'mean': MEAN,
    'variation': VARIATION,
    'validation_size': args.validation_size,
    'model_file': 'model/temp.pt',
    'restore': args.restore_model,
    'convolution': True
}


def generate_tile():
    """Generate new random tile"""
    generator = Generator(config)
    num = args.show_size
    plt.figure(figsize=(22, 6))
    for i in range(num):
        image = generator.generate()
        plt.subplot(1, num, i+1)
        plt.imshow(image)
    plt.show()


def train_tiler():
    """Train tiler on given images"""
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
    if args.experiment == 'train':
        train_tiler()
    else:
        generate_tile()
