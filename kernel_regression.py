# Copyright (c) 2006-2021 Tampere University.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes prohibited.

# !/usr/bin/env python
import numpy as np


class KernelRegression:
    def __init__(self, X=None, Y=None, sigmas=None, periodic=False):
        """
        :param X: pose samples of N x 6 where each 6D vector is [x(m), y(m), z(m), rx(rad), ry(rad), rz(rad)]
        :param Y: task outcome N x 1 where y=0 is failed and y=1 is succeed
        :param sigmas: standard deviation
        :param periodic:
        """
        self._Y = Y
        self._X = X
        self._sigmas = sigmas
        self._periodic = periodic

    @staticmethod
    def gaussian_kernel(x, mu, sigma):
        """
        :param x: data point of D x 1
        :param mu: center 1 x 1
        :param sigma: standard deviation 1 x 1
        :return:
        """
        res = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2.0) * \
              np.exp(-((x - mu) ** 2.0) / (2.0 * sigma ** 2.0))
        return res

    @staticmethod
    def multivariate_gaussian_kernel(x, mu, cov):
        """
        :param x: data points dimension of D x N
        :param mu: center D x 1
        :param cov: covariance matrix D x D (elements are variances e.g. std**2)
        :return:
        """
        part1 = 1.0 / (((2.0 * np.pi) ** (len(mu) / 2.0)) * (np.linalg.det(cov) ** (1.0 / 2.0)))
        if x.ndim > 1:
            part2 = (-1.0 / 2.0) * ((x - mu).dot(np.linalg.inv(cov)) * (x - mu)).sum(axis=1)
        else:
            part2 = (-1.0 / 2.0) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
        return part1 * np.exp(part2)

    def nadaraya_watson_regression(self, Y, X, sigma, center):
        """
        :param Y: samples N x 1
        :param X: sample coordinates N x D
        :param sigma: standard deviation (not variance which is std**2) 1 x D
        :param center: gaussian center point
        :return:
        """
        N, D = X.shape
        assert len(center) == D, 'Number of center coordinates have to match sample dimensions'
        assert len(sigma) == D, "Sigma dimension does not match to data dimension"
        assert N == len(Y)

        if D < 2:
            kernel_res = self.gaussian_kernel(X, mu=center, sigma=sigma)
            res = np.sum((kernel_res * Y)) / np.sum(kernel_res)
        else:
            sigma = np.diag(sigma) ** 2
            kernel_res = self.multivariate_gaussian_kernel(X, mu=center, cov=sigma)
            if np.sum(kernel_res) == 0.0:
                return 0
            res = np.sum((kernel_res.reshape(N, 1) * Y)) / np.sum(kernel_res)
        return res

    def periodic_nadaraya_watson_regression(self, Y, X, sigma, center):
        """
        Assumes rotational components are already projected to 0..2*PI interval
        :param Y: Data samples N x 1
        :param X: Data coordinates N x D
        :param sigma: Sigmas D x 1 !Note standard deviation, not variance (std**2)
        :param center: Gaussian center
        :return:
        """
        N, D = X.shape
        assert len(center) == D, 'Number of center coordinates have to match sample dimensions'
        assert sigma.ndim == 1 and sigma.shape[0] == D, "Incorrect sigmas"
        assert N == len(Y)

        gaussian_kernel = np.ones(N)
        for i in range(D):
            if i > 2:

                gaus_temp = self.gaussian_kernel(X[:, i], mu=center[i], sigma=sigma[i]) + \
                            self.gaussian_kernel(X[:, i], mu=center[i] + (2 * np.pi), sigma=sigma[i]) + \
                            self.gaussian_kernel(X[:, i], mu=center[i] - (2 * np.pi), sigma=sigma[i])
                gaussian_kernel *= gaus_temp
            else:
                gaussian_kernel *= self.gaussian_kernel(X[:, i], mu=center[i], sigma=sigma[i])

        if np.sum(gaussian_kernel) == 0:
            return 0
        res = np.sum((gaussian_kernel.reshape(N, 1) * Y)) / np.sum(gaussian_kernel)
        return res

    @staticmethod
    def nadaraya_watson_regression_fast(Y, X, sigma, center):
        """
        :param Y: data samples N x 1
        :param X: data coordinates N x D
        :param sigma: standard deviation D x 1
        :param center: gaussian center D x 1
        :return:
        """
        N, D = X.shape
        assert len(center) == D, 'Number of center coordinates have to match sample dimensions'
        assert len(sigma) == D, 'Sigma dimension does not match to data dimension'
        assert N == len(Y), 'Number of samples does not match to number of points'

        sigma = np.diag(sigma) ** 2
        gauss_sum = np.exp((-1.0 / 2.0) * ((X - center).dot(np.linalg.inv(sigma)) * (X - center)).sum(axis=1))
        if np.sum(gauss_sum) == 0.0:
            return 0
        res = np.sum((gauss_sum.reshape(N, 1) * Y)) / np.sum(gauss_sum)
        return res

    def estimate(self, test_point):
        assert len(self._sigmas) > 0, "No sigmas"
        assert len(self._Y) > 0 and len(self._X) > 0, "No training data"
        if not self._periodic:
            res = self.nadaraya_watson_regression(self._Y, self._X, self._sigmas, test_point)
        else:
            pos = self._X.copy()
            pos[:, 3:] = pos[:, 3:] % (2 * np.pi)
            tp = test_point.copy()
            tp[3:] = tp[3:] % (2 * np.pi)
            res = self.periodic_nadaraya_watson_regression(self._Y, pos, np.sqrt(np.diag(self._sigmas)), tp)
        return res


def main():
    pass


if __name__ == "__main__":
    main()
