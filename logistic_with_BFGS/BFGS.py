import numpy as np


def norm(x):
    return np.dot((x-1).T, (x-1))


def norm_grad(x):
    return 2 * (x-1)


def BFGS(f, dif_f, x0, threshold=1e-6, max_iters=1e6):
    """
    implementation of an optimization
    :param f: f(x) is the function to be minimized
    :param dif_f: dif_f(x) is the derivative(gradient) of f
    :param x0: initialization
    :param max_iters: the max number of iterations
    :param threshold: the algorithms stops when the difference between two iterations are smaller than the threshold
    :return: x when f(x) is minimized
    """
    dim = len(x0)
    B = np.eye(dim)  # inverse Hessian
    i = 0
    while i < max_iters:
        # find newton direction
        d = - np.dot(B, dif_f(x0))
        # update x0 with a line search along Newton direction
        alpha, new_x0 = wolfe_line_search(f, dif_f, x0, d)
        # keep a record of s(k)=x(k+1)-x(k) and y(k)=dif_f(x(k+1))-dif_f(x(k))
        s = new_x0 - x0
        y = dif_f(new_x0) - dif_f(x0)
        # update inverse hessian matrix
        B = B + np.dot(y, y.T) / np.dot(y.T, s) + B.dot(s).dot(s.T).dot(B.T) / s.T.dot(B).dot(s)
        if np.dot(s.T, s) < threshold:
            break
        x0 = new_x0
        i += 1
    return x0


def wolfe_line_search(f, dif_f, x0, d, rho=0.25, sigma=0.75, max_iter=10000):
    """
    an inexact line search
    :param max_iter: max number of iterations
    :param sigma: parameter in wolfe condition
    :param rho: parameter in wolfe condition
    :param f: target function
    :param dif_f: gradient of target function
    :param x0: the initial argument
    :param d: search direction
    :return: the step length
    """
    a, b, alpha = 0, np.inf, 1
    rho_criteria = rho * np.dot(d.T, dif_f(x0))
    sigma_criteria = np.abs(sigma * np.dot(d.T, dif_f(x0)))
    i = 0
    while i < max_iter:
        new_x0 = x0 + alpha * d
        if f(new_x0) <= f(x0) + alpha * rho_criteria:
            if np.abs(np.dot(d.T, dif_f(new_x0))) <= sigma_criteria:
                break
            else:
                a = alpha
                alpha = np.min([2 * alpha, (alpha + b) / 2])
        else:
            b = alpha
            alpha = (a + alpha) / 2
        i += 1
    return alpha, new_x0


if __name__ == "__main__":
    print(BFGS(norm, norm_grad, x0=np.random.randn(5, 1)))
