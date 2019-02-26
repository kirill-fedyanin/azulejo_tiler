from os import listdir
from PIL import Image
import PIL

TARGET_SIZE = (236, 236)
DATA_DIR = 'data/'


def preprocess():
    """Resize all images to same size and save it to """


def remove_duplicates():
    """Remove duplicates based on euristic rule"""


def show_dimensions():
    """Print dimensions of biggest and smallest images"""
    min_d = (10e6, 10e6)
    max_d = (0, 0)

    for image_name in listdir(DATA_DIR):
        image = Image.open(DATA_DIR + image_name)
        height, width = image.size

        if height*width > max_d[0]*max_d[1]:
            max_d = (height, width)
        if height*width < min_d[0]*min_d[1]:
            min_d = (height, width)

    print("Minimal dimensions", min_d)
    print("Maximal dimensions", max_d)


if __name__ == "__main__":
    show_dimensions()
