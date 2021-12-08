"""
svm, data generation:
    https://github.com/cperales/SupportVectorMachine
svm optimization reference:
    https://www.baeldung.com/cs/svm-hard-margin-vs-soft-margin
    https://cvxopt.org/userguide/coneprog.html#quadratic-programming
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
from cvxopt import matrix, solvers

cur_dir, _ = os.path.split(__file__)
solvers.options['show_progress'] = False


def fit(x, y):
    num = x.shape[0]
    K = y * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    G = matrix(-np.eye(num))
    h = matrix(np.zeros(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def fit_soft(x, y, C):
    num = x.shape[0]
    K = y * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    g = np.concatenate((-np.eye(num), np.eye(num)))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


class LinearSVM(object):
    """
    Implementation of the linear support vector machine.
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None

    def fit(self, X, y, soft=True):
        y = y[..., None] if len(y.shape) == 1 else y

        if soft is True:
            C = 1.0
            alphas = fit_soft(X, y, C)
        else:
            alphas = fit(X, y)

        w = np.sum(alphas * y * X, axis=0)
        b_vector = y - np.dot(X, w.T)
        b = b_vector.sum() / b_vector.size

        norm = np.linalg.norm(w)
        w, b = w / norm, b / norm

        self.w = w
        self.b = b

    def predict(self, X):
        y = np.sign(np.dot(self.w, X.T) + self.b * np.ones(X.shape[0]))
        return y


def load_data_2d(num=50,
                 overwrite=False):
    """
    Generate 2D random binary data and save it into a pickle.
    """
    file_name = os.path.join(cur_dir, 'clf_data.dat')
    if os.path.exists(file_name) or not overwrite:
        with open(file_name, 'rb') as f:
            x, y = pickle.load(f)
        return x, y

    else:
        dim = 2
        mu = -4 + 8 * np.random.rand(2, dim)
        sigma = 0.5 * np.ones([2, dim])

        # generate points for class 1
        x1 = np.random.multivariate_normal(mu[0], np.diag(sigma[0]), num)
        # generate points for class 2
        x2 = np.random.multivariate_normal(mu[1], np.diag(sigma[1]), num)
        # labels
        y1 = np.ones(num)
        y2 = -np.ones(num)
        # join
        x = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        with open(file_name, 'wb') as f:
            pickle.dump((x, y), f)

        return x, y


def plot_clf(X, y, title, w, b):
    _, ax = plt.subplots()
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b')
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, -1], c='r')
    plt.title(title)
    xy_lim = np.array(ax.viewLim).T

    pt0 = [min(xy_lim[0]), - (w[0] * min(xy_lim[0]) + b) / w[1]]
    pt1 = [max(xy_lim[0]), - (w[0] * max(xy_lim[0]) + b) / w[1]]
    pt2 = [- (w[1] * min(xy_lim[1]) + b) / w[0], min(xy_lim[1])]
    pt3 = [- (w[1] * max(xy_lim[1]) + b) / w[0], max(xy_lim[1])]

    pts = np.array([pt0, pt1, pt2, pt3])
    pts = np.delete(pts, np.argmax(pts[:, 1]), axis=0)
    pts = np.delete(pts, np.argmin(pts[:, 1]), axis=0)
    ax.plot(pts[:, 0], pts[:, 1], c='k')

    plt.show()


def test_classifier_hard_svm():
    clf = LinearSVM()
    X, y = load_data_2d()
    clf.fit(X=X, y=y, soft=False)
    y_pred = clf.predict(X=X)

    plot_clf(X, y, 'data', clf.w, clf.b)
    plot_clf(X, y_pred, 'classified', clf.w, clf.b)


def test_classifier_soft_svm():
    clf = LinearSVM()
    X, y = load_data_2d()
    clf.fit(X=X, y=y, soft=True)
    y_pred = clf.predict(X=X)

    plot_clf(X, y, 'data', clf.w, clf.b)
    plot_clf(X, y_pred, 'classified', clf.w, clf.b)


def test_classifier():
    X, y = load_data_2d()
    X, y = torch.tensor(X).float(), torch.tensor(y).float()

    w = torch.autograd.Variable(torch.randn(2), requires_grad=True)
    b = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
    opt = torch.optim.SGD(params=[w, b], lr=1e-3)

    losses = []
    for itr in range(3000):
        target = X @ w.T + b
        loss = torch.mean(torch.clamp(1.0 - y * target, min=0))
        # loss += torch.norm(w)
        # loss = torch.mean(- y * target)
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        # print(f'# {itr}, loss: {loss.item():.2f}')

    plt.plot(losses)
    plt.title('learning curve')
    plt.show()

    w = w.detach().numpy()
    b = b.detach().numpy()
    y_pred = np.sign(target.detach().numpy())

    plot_clf(X, y, 'data', w, b)
    plot_clf(X, y_pred, 'classified', w, b)
