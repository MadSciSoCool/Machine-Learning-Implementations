import numpy as np


class SvmClassifier:
    def __init__(self, kernel, b, alphas, x, y):
        self.kernel = kernel
        self.b = b
        self.alphas = alphas
        self.x = x
        self.y = y

    def decision(self, x0):
        return np.sum(self.alphas * self.y * self.kernel(self.x, x0)) + self.b

    def predict(self, x0):
        return 1 if self.decision(x0) > 0 else -1


def SVM(data, c=np.inf, kernel=None, max_iteration=50, tol=0.0001):
    """
    :param tol: tolerance
    :param max_iteration: number of max iterations in SMO
    :param data:
    :param c: the factor for relaxation
    :param kernel:
    :return:
    """
    x, y = data
    if kernel is None:
        def k(m, n):
            return np.dot(m, n)

        kernel = k
    size = len(x)
    # SMO, here just pick i and j randomly
    alphas = np.zeros(size, dtype=float)
    b = 0.
    iteration = 0
    while iteration < max_iteration:
        changed = 0
        for i in range(size):
            # check KKT
            x1, y1 = x[i], y[i]
            e1 = np.sum(alphas * y * kernel(x, x1)) + b - y1
            if (y1 * e1 < -tol and alphas[i] < c) or (y1 * e1 > tol and alphas[i] > 0):
                perm = np.random.permutation(size)
                j = perm[perm != i][0]
                x2, y2 = x[j], y[j]
                # first calculate the valid domain for alpha_j
                if y1 == y2:
                    low, high = max(0., alphas[i] + alphas[j] - c), min(c, alphas[i] + alphas[j])
                else:
                    low, high = max(0., alphas[j] - alphas[i]), min(c, c + alphas[j] - alphas[i])
                if low == high:
                    continue
                e2 = np.sum(alphas * y * kernel(x, x2)) + b - y2
                eta = 2 * kernel(x1, x2) - kernel(x1, x1) - kernel(x2, x2)
                # update alpha2 (delta2 = alpha2_new - alpha2_old)
                delta2 = - y2 * (e1 - e2) / eta
                # truncate the solution for alpha2
                delta2 = max(low - alphas[j], delta2)
                delta2 = min(high - alphas[j], delta2)
                if np.abs(delta2) < tol:
                    continue
                delta1 = - y1 * y2 * delta2
                # update alpha1 and alpha2
                alphas[i] = alphas[i] + delta1
                alphas[j] = alphas[j] + delta2
                # update b
                b1 = b - e1 - y1 * delta1 * kernel(x1, x1) - y2 * delta2 * kernel(x1, x2)
                b2 = b - e2 - y1 * delta1 * kernel(x1, x2) - y2 * delta2 * kernel(x2, x2)
                if 0. < alphas[i] < c:
                    b = b1
                elif 0. < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.
                changed += 1
        if changed == 0:
            iteration += 1
    return SvmClassifier(kernel, b, alphas, x, y)
