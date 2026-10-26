import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, c=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.c = c
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y > 0, 1, -1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.c * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.c * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
