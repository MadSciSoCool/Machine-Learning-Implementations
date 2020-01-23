import numpy as np


def k_means(pixels, K, max_iterations=10000):
    num_of_pixels = len(pixels)
    centroids = np.random.randint(256, size=(K, 3))
    assigning = np.zeros(num_of_pixels)
    for k in range(max_iterations):
        # assign samples
        swapped = False
        for i in range(num_of_pixels):
            current_norm = np.inf
            for cla in range(K):
                this_norm = np.linalg.norm(pixels[i] - centroids[cla])
                if this_norm < current_norm:
                    current_norm = this_norm
                    assigning[i] = cla
                    swapped = True
        # update centroids
        for cla in range(K):
            centroids[cla] = np.average(pixels[np.round(np.where(assigning == cla))])
        # stop iteration if centroids not change
        if not swapped:
            break
    return assigning, centroids


def k_medoids(pixels, K, max_iterations=10000):
    num_of_pixels = len(pixels)
    centroids = pixels[np.random.choice(num_of_pixels, K, replace=False)]
    assigning = np.zeros(num_of_pixels)
    for k in range(max_iterations):
        swapped = False
        # assign samples
        for i in range(num_of_pixels):
            current_norm = np.inf
            for cla in range(K):
                this_norm = np.linalg.norm(pixels[i] - centroids[cla])
                if this_norm < current_norm:
                    current_norm = this_norm
                    assigning[i] = cla
                    swapped = True
        # update centroids
        for cla in range(K):
            pixels_this_class = pixels[assigning == cla]
            centroids[cla] = pixels_this_class[
                np.argmin([np.sum(np.abs(pixels_this_class - pixel)) for pixel in pixels_this_class])]
        if not swapped:
            break
    return assigning, centroids
