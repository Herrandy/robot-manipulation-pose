# Copyright (c) 2006-2021 Tampere University.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes prohibited.

# !/usr/bin/env python
import numpy as np
import os
import glob
import argparse

from kernel_regression import KernelRegression
from math_tools import Tf2pose


def calculate_grasp_success_rate(estimates, gts, kr):
    """
    Calculates the grasp success rate
    :param estimates: Nx4x4 of estimated object poses
    :param gts: Nx4x4 of ground truth object poses
    :param kr: model to estimate the grasp success probability
    :return: Nx1 estimated probabilities
    """
    assert estimates.shape == gts.shape
    probs = np.zeros(len(estimates))
    for e in range(len(estimates)):
        tf_est = estimates[e]

        if np.isnan(tf_est).any():
            tf_est = np.eye(4)

        tf_gt = gts[e]
        T = np.dot(np.linalg.inv(tf_gt), tf_est)
        assert np.sum(np.dot(tf_gt, T) - tf_est) < 1e-12
        pose_vec = Tf2pose(T)
        probs[e] = kr.estimate(pose_vec)
    return probs


def run(model_data_dir, estimate_dir):
    path_sigmas = os.path.join(model_data_dir, 'sigmas/sigmas.txt')
    path_training_data = os.path.join(model_data_dir, 'sigmas/training_samples.txt')
    data = np.loadtxt(path_training_data)
    X = [d[:-1] for d in data]
    Y = [d[-1] for d in data]
    X = np.array(X).reshape(-1, 6)
    Y = np.array(Y).reshape(-1, 1)
    sigmas = np.loadtxt(path_sigmas)
    sigmas = np.sqrt(np.diag(sigmas))

    kr = KernelRegression(X=X, Y=Y, sigmas=sigmas)

    # load ground truth and estimates transformation matrices
    estimates = np.sort(glob.glob(os.path.join(estimate_dir, '*.txt')))
    gts = np.sort(glob.glob(os.path.join(model_data_dir, 'gt', '*_gt_full.txt')))
    probs = calculate_grasp_success_rate(estimates, gts, kr)
    for p_idx, p in enumerate(probs):
        print(f'Calculated success probability for estimate {p_idx}: {p}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate-dir", type=str, required=False,
                        help="Path to a directory containing the estimated object poses.")
    parser.add_argument("--model-data-dir", type=str, required=False,
                        help="Path to a directory containing the model data.")
    args = parser.parse_args()
    run(args.model_data_dir, args.estimate_dir)


if __name__ == '__main__':
    main()
