import random

import numpy as np
import matplotlib.pyplot as plt


K = 3
ITERATIONS = 1000

# (N, 2)
points = np.vstack(
    (
        (np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
        (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
        (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5])),
    )
)


def initiate_k_centroids(all_points, k):
    all_points_copy = all_points.copy()
    random.shuffle(all_points_copy)
    return all_points_copy[:k]


def step(all_points, centroids, k):
    """
    all_points: (N, 2)
    centroids: (k, 2)
    """
    # make arrays can be broadcast
    all_points_copy = all_points.copy()
    all_points = all_points[..., np.newaxis]  # (N, 2, 1)
    centroids_copy = centroids.copy()
    centroids = np.transpose(centroids)[np.newaxis, ...]  # (1, 2, k)
    diff = np.linalg.norm(all_points - centroids, axis=1)  # (N, k)
    nearest_neighbor = np.argmax(diff, axis=-1)  # (N)

    # now calculate new centroids
    new_centroids = []
    for i in range(k):
        points_same_class = all_points_copy[nearest_neighbor == i]  # (N, 2)
        if points_same_class.shape[0] == 0:
            new_centroids.append(centroids_copy[i])
        else:
            new_centroids.append(np.mean(points_same_class, axis=0))
    return np.array(new_centroids, dtype=np.float32)


if __name__ == "__main__":
    centroids = initiate_k_centroids(points, k=K)
    for _ in range(ITERATIONS):
        centroids = step(points, centroids, k=K)

    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c="r", s=100)
    plt.show()
