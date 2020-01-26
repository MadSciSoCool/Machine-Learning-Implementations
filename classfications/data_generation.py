import numpy as np
import matplotlib.pyplot as plt
from classfications.SVM import SVM


def data_generation_2D(scale, size, linear_separable=True):
    x, y = scale
    data = np.hstack((x * np.random.rand(size, 1), y * np.random.rand(size, 1)))
    x0, y0 = np.array([x / 2, y / 2]) + np.random.randn() * np.linalg.norm([x, y]) / 10
    k = np.tan(np.pi * (np.random.rand() - 0.5))
    classification = np.array([1. if k * (x - x0) - (y - y0) < 0 else -1. for x, y in data])
    return [data, classification]


def visualize(data, size, classifier):
    x, y = size
    data, classification = data
    red = data[np.where(classification == 1)]
    blue = data[np.where(classification == -1)]
    plt.scatter(red.T[0], red.T[1], c="red")
    plt.scatter(blue.T[0], blue.T[1], c="blue")
    # plot decision boundary
    test = np.array([[xx, yy] for xx in np.arange(0, x, x / 10) for yy in np.arange(0, y, y / 10)])
    pred = np.array([classifier.predict(t) for t in test])
    test_red = test[np.where(pred == 1)]
    test_blue = test[np.where(pred == -1)]
    plt.scatter(test_red.T[0], test_red.T[1], c="red", marker="x")
    plt.scatter(test_blue.T[0], test_blue.T[1], c="blue", marker="x")
    plt.show()


if __name__ == "__main__":
    x, y = 1., 1.
    data = data_generation_2D((x, y), 50)
    classifier = SVM(data)
    visualize(data, (x, y), classifier)
