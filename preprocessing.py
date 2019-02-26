import os
from PIL import Image
from PIL import ImageChops
import PIL
import numpy as np

TARGET_SIZE = (236, 236)
DATA_DIR = 'data/'


def preprocess():
    """Resize all images to same size and save it to """


def remove_duplicates():
    """Remove duplicates based on euristic rule"""
    prev_image = None
    for image_name in reversed(sorted(os.listdir(DATA_DIR))):
        image = Image.open(DATA_DIR + image_name)

        if prev_image is None:
            prev_image = image
            continue

        if image.size == prev_image.size:
            diff = np.array(image) - np.array(prev_image)
            if not np.any(diff):
                os.remove(DATA_DIR + image_name)
                print(image_name, 'deleted')

        prev_image = image


def show_dimensions():
    """Print biggest and smallest image dimensions"""
    min_d = (10e6, 10e6)
    max_d = (0, 0)

    for image_name in os.listdir(DATA_DIR):
        image = Image.open(DATA_DIR + image_name)
        height, width = image.size

        if height*width > max_d[0]*max_d[1]:
            max_d = (height, width)
        if height*width < min_d[0]*min_d[1]:
            min_d = (height, width)

    print("Minimal dimensions", min_d)
    print("Maximal dimensions", max_d)


if __name__ == "__main__":
    remove_duplicates()
