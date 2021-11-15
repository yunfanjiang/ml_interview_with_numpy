"""
Covariance-based PCA implemented purely with numpy

Pseudo Algorithm
Requires: data X (n, m) with n instances, each instance has m features.
1. normalize data by subtracting mean: X = X - mean(X, axis=0)
2. normalize data by dividing std: X = X / std(x, axis=0)
3. find covariance matrix between features: C = cov(X), shape: (m, m)
4. find eigenvalues and eigenvector: eigenvalue, eigenvector = eig(C). eigenvalue shape (m, ), eigenvector shape (m, m)
5. select largest k eigenvectors: D = sort(eigenvector, according=eigenvalue)[:, :k], shape (m, k)
6. return the encoding matrix D, the input data can be encoded through dot(X, D), resulted shape (n, k)
"""


import numpy as np
import matplotlib.pyplot as plt  # for visualization only


def pca(x: np.ndarray, n_components: int = 2):
    assert x.ndim == 2
    n, m = x.shape
    assert n_components < m

    # normalize data by subtracting
    x = x - np.mean(x, axis=0)  # (n, m)
    # normalize data by dividing std
    x = x / np.std(x, axis=0)  # (n, m)
    # find covariance matrix
    cov_mat = np.cov(x.T)  # (m, m)
    # find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    # descending sort eigenvectors according to eigenvalues
    sort_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sort_idx]
    # select `n_components` largest eigenvectors so we have the encoding matrix
    encode_mat = sorted_eigenvectors[:, :n_components]  # (m, k)
    # encode the input data
    encoded_data = np.dot(x, encode_mat)  # (n, k)
    return encoded_data, encode_mat


if __name__ == "__main__":
    # make up some test data
    data = np.array([np.random.randn(8) for k in range(150)])
    data[:50, 2:4] += 5
    data[50:, 2:5] += 5

    encoded_data, encode_mat = pca(data, n_components=2)

    # visualization
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(data[:50, 0], data[:50, 1], c="r")
    ax1.scatter(data[50:, 0], data[50:, 1], c="b")
    ax2.scatter(encoded_data[:50, 0], encoded_data[:50, 1], c="r")
    ax2.scatter(encoded_data[50:, 0], encoded_data[50:, 1], c="b")
    plt.show()
