import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eig_val, eig_vec = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eig_vec = eig_vec.T

        # sort eigenvectors
        idxs = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idxs]
        eig_vec = eig_vec[idxs]

        self.components = eig_vec[:self.n_components]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)
