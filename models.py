import numpy as np


class GaussianNaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9

    def gaussian_log_likelihood(self, x, mean, var):
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_likelihood

    def predict_one(self, x):
        scores = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self.gaussian_log_likelihood(
                x,
                self.means[c],
                self.vars[c]
            )
            score = log_prior + log_likelihood
            scores.append(score)

        return self.classes[np.argmax(scores)]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class LDAScratch:
    """
    GDA with shared covariance matrix.
    This is equivalent to LDA for classification.
    """

    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.sigma = None
        self.sigma_inv = None
        self.sigma_det = None

    def fit(self, X, y):
        m, n = X.shape
        self.classes = np.unique(y)

        sigma = np.zeros((n, n))

        for c in self.classes:
            X_c = X[y == c]

            self.priors[c] = X_c.shape[0] / m
            self.means[c] = np.mean(X_c, axis=0)

        for i in range(m):
            c = y[i]
            diff = (X[i] - self.means[c]).reshape(-1, 1)
            sigma += diff @ diff.T

        sigma = sigma / m
        sigma += 1e-6 * np.eye(n)

        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma)
        self.sigma_det = np.linalg.det(sigma)

    def multivariate_gaussian_log_pdf(self, x, mean):
        n = x.shape[0]
        diff = (x - mean).reshape(-1, 1)

        log_pdf = -0.5 * n * np.log(2 * np.pi)
        log_pdf -= 0.5 * np.log(self.sigma_det)
        log_pdf -= 0.5 * (diff.T @ self.sigma_inv @ diff)[0, 0]

        return log_pdf

    def predict_one(self, x):
        scores = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self.multivariate_gaussian_log_pdf(
                x,
                self.means[c]
            )

            score = log_prior + log_likelihood
            scores.append(score)

        return self.classes[np.argmax(scores)]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class QDAScratch:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.sigmas = {}
        self.sigma_invs = {}
        self.sigma_dets = {}

    def fit(self, X, y):
        m, n = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            self.priors[c] = X_c.shape[0] / m
            self.means[c] = np.mean(X_c, axis=0)

            sigma_c = np.cov(X_c, rowvar=False, bias=True)
            sigma_c += 1e-6 * np.eye(n)

            self.sigmas[c] = sigma_c
            self.sigma_invs[c] = np.linalg.inv(sigma_c)
            self.sigma_dets[c] = np.linalg.det(sigma_c)

    def multivariate_gaussian_log_pdf(self, x, mean, sigma_inv, sigma_det):
        n = x.shape[0]
        diff = (x - mean).reshape(-1, 1)

        log_pdf = -0.5 * n * np.log(2 * np.pi)
        log_pdf -= 0.5 * np.log(sigma_det)
        log_pdf -= 0.5 * (diff.T @ sigma_inv @ diff)[0, 0]

        return log_pdf

    def predict_one(self, x):
        scores = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = self.multivariate_gaussian_log_pdf(
                x,
                self.means[c],
                self.sigma_invs[c],
                self.sigma_dets[c]
            )

            score = log_prior + log_likelihood
            scores.append(score)

        return self.classes[np.argmax(scores)]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])