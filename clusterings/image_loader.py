from PIL import Image
from pylab import *
from clusterings.k_means_and_k_medoids import k_means, k_medoids
import numpy as np


def load_image(path):
    image_array = array(Image.open(path))
    return np.array(image_array).reshape(-1, 3)


def draw_image(arr):
    image = Image.fromarray(np.uint8(arr))
    image.show()


if __name__ == "__main__":
    path = r"lena.png"
    pixels = load_image(path)
    num_of_pixels = pixels.shape[0]
    assigning, centroids = k_means(pixels, 2)
    K = centroids.shape[0]
    binarized_pixels = np.zeros((num_of_pixels, 3), dtype=np.int)
    for cla in range(K):
        binarized_pixels[assigning == cla] = centroids[cla]
    draw_image(binarized_pixels.reshape((512, 512, -1)))
