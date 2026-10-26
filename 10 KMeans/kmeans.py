import matplotlib.pyplot as plt
import numpy as np


def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.centroids = None

    def predict(self, X):

        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iters):

            for idx, x in enumerate(X):
                labels[idx] = np.argmin([euclidean_distance(x, c) for c in self.centroids])

            if self.plot_steps:
                self.plot(X, labels)

            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.K)])
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        return labels

    def plot(self, X, labels):
        fig, ax = plt.subplots(figsize=(12, 8))

        for k in range(self.K):
            point = X[labels == k].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()