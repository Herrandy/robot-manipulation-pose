# Copyright (c) 2006-2021 Tampere University.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes prohibited.

# !/usr/bin/env python
import numpy as np


def Tf2pose(T):
    """
    Converts transformation matrix into a 6D pose vector
    :param T: 4x4 transformation matrix
    :return: 6x1 pose  vector
    """
    output = np.zeros(6)
    output[:3] = T[:3, 3]
    angle, direction, _ = rotm2axis_ang(T)
    output[3:] = direction * angle
    return output


def rotm2axis_ang(matrix):
    """
    Converts rotation matrix into rotation angle and axis
    :param matrix: 3x3 rotation matrix
    :return: rotation angle and axis
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = np.atan2(sina, cosa)
    return angle, direction, point
