# Copyright (c) 2006-2021 Tampere University.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes prohibited.

# !/usr/bin/env python
import argparse
import numpy as np
from kernel_regression import KernelRegression

np.set_printoptions(precision=10, suppress=True)


def LOOCV(X, Y, init_sigmas, save_path):
    """
    Simple Leave-One-Out-Cross-Validation training procedure
    :param X: data coordinates N x 6
    :param Y: data labels N x 1
    :param init_sigmas: 6 x 1 initial values for sigmas from which the training is started
    :param save_path: path to a file where the sigmas are stored
    :return:
    """
    steps = np.arange(0.001, 2.001, 0.001)
    sigmas = init_sigmas
    kr = KernelRegression()

    # iterate over data dimensions
    for s in range(len(sigmas)):
        step_results = np.ones(len(steps)) * float("-inf")
        # iterate over sigma test values
        for v in range(len(steps)):
            error = 0.0
            # iterate over training samples
            for idx in range(len(X)):
                I = np.ones(len(X), dtype=bool)
                I[idx] = False

                test_sigmas = sigmas.copy()
                test_sigmas[s] = sigmas[s] * steps[v]
                res = kr.nadaraya_watson_regression_fast(Y[I].reshape(-1, 1), X[I].reshape(-1, 6), test_sigmas, X[idx])

                if Y[idx]:
                    error = error + np.log(res + 1e-100)
                else:
                    error = error + np.log(1 - res + 1e-100)
            print("Running sigma %i, step %i" % (s, v))
            print("Step, Error = (%i/%i, %f)" % (v, len(steps), error))
            step_results[v] = error

            print("Error %f" % error)
            step_results[v] = error
        max_id = np.argmax(step_results)
        sigmas[s] = sigmas[s] * steps[max_id]
        print("Best sigma (value, id): %f, %u" % (sigmas[s], max_id))
        np.savetxt(save_path, sigmas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-training-samples", type=str, help="Path to a file containing the training samples")
    parser.add_argument("--path-save-file", type=str, help="Path to a file where the sigmas are stored")
    parser.add_argument("--init-sigma-values", nargs="+",
                        help="Initial values for sigmas from which the training is started")
    args = parser.parse_args()

    data_path = args.path_training_samples
    save_path = args.path_save_file
    data = np.loadtxt(data_path)
    X = np.array([d[:-1] for d in data])
    Y = np.array([d[-1] for d in data])
    Y = Y.reshape(-1, 1)
    LOOCV(X, Y, args.init_sigma_values, save_path=save_path)


if __name__ == "__main__":
    main()
