# Copyright (c) 2006-2021 Tampere University.
# All rights reserved.
# This work (software, material, and documentation) shall only
# be used for nonprofit noncommercial purposes.
# Any unauthorized use of this work for commercial or for-profit purposes prohibited.

# !/usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import open3d as o3d
from kernel_regression import KernelRegression


def ticks_to_degrees(x, pos):
    return "{:.1f}".format(np.rad2deg(x))


def color_cloud(pcd, color):
    npoints = len(np.array(pcd.points))
    color_points = np.ones((npoints, 3)) * color
    pcd.colors = o3d.utility.Vector3dVector(color_points)
    return pcd


def model_1D(data_dir):
    """
    Plots the estimated and true success probabilities along each pose axis
    :param data_dir: Path to the robot manipulation dataset
    :return:
    """
    path_sigmas = os.path.join(data_dir, 'motor_cap2/sigmas/sigmas.txt')
    path_eval = os.path.join(data_dir, 'motor_cap2/sigmas/robot_eval.txt')
    path_training_data = os.path.join(data_dir, 'motor_cap2/sigmas/training_samples.txt')

    data = np.loadtxt(path_training_data)
    eval_data = np.loadtxt(path_eval)
    X = [d[:-1] for d in data]
    Y = [d[-1] for d in data]
    X = np.array(X).reshape(-1, 6)
    Y = np.array(Y).reshape(-1, 1)
    sigmas = np.loadtxt(path_sigmas)
    sigmas = np.sqrt(np.diag(sigmas))

    pw = KernelRegression()
    all_results = []
    xs = []
    for dim in range(6):
        if dim < 3:
            x = np.linspace(-0.02, 0.02, 251)
        else:
            x = np.linspace(-np.pi, np.pi, 251)
            ind = np.argsort(x, axis=0)
            x = x[ind]

        res = []
        xs.append(x)
        for idx in range(x.shape[0]):
            vec = np.zeros(6)
            vec[dim] = x[idx]
            res.append(pw.nadaraya_watson_regression_fast(Y.reshape(-1, 1), X.reshape(-1, 6), sigmas, vec))
        all_results.append(np.array(res))

    # Translation plot
    colors = ['-r', '-g', '-b', '-r', '-g', '-b']
    f_trans, axes_trans = plt.subplots(3, sharey=True)
    C = axes_trans.shape[0]
    for c in range(C):
        axes_trans[c].plot(xs[c], all_results[c], colors[c], linewidth=2)
        axes_trans[c].set_xlim(-0.03, 0.03)
        axes_trans[c].set_ylim(0.0, 1.4)
        # evaluation data
        dim_data = np.array(eval_data[eval_data[:, 0] == c])
        dim_data = dim_data[np.argsort(dim_data[:, 1])]
        axes_trans[c].plot(dim_data[:, 1], dim_data[:, 2], '-y', linewidth=2)
    axes_trans[0].set_title('Translation', fontsize=17, fontweight="bold")
    f_trans.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f_trans.axes[:-1]], visible=False)
    plt.xlabel('Meters', fontsize=13)

    # Rotation plot
    f_rot, axes_rot = plt.subplots(3, sharey=True)
    C = axes_rot.shape[0]
    for c in range(C):
        # data to -pi..pi range
        ind = xs[c + 3] > np.pi
        xs[c + 3][ind] = xs[c + 3][ind] % (np.pi * -2.0)
        sort_id = np.argsort(xs[c + 3])
        xs[c + 3] = xs[c + 3][sort_id]
        all_results[c + 3] = all_results[c + 3][sort_id]
        axes_rot[c].plot(xs[c + 3], all_results[c + 3], colors[c + 3], linewidth=2)
        axes_rot[c].set_xlim(-0.3, 0.3)
        axes_rot[c].set_ylim(0.0, 1.4)

        # evaluation data
        dim_data = np.array(eval_data[eval_data[:, 0] == (c + 3)])
        dim_data = dim_data[np.argsort(dim_data[:, 1])]
        axes_rot[c].plot(dim_data[:, 1], dim_data[:, 2], '-y', linewidth=2)
    axes_rot[0].set_title('Rotation', fontsize=17, fontweight="bold")
    f_rot.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f_rot.axes[:-1]], visible=False)
    axes_rot[-1].xaxis.set_major_formatter(mticker.FuncFormatter(ticks_to_degrees))
    plt.xlabel('Degrees', fontsize=13)
    plt.show()


def model_2D(data_dir):
    """
    Plots the estimated success probabilities on a XY-plane
    :param data_dir: Path to the robot manipulation dataset
    :return:
    """
    path_sigmas = os.path.join(data_dir, 'motor_cap2/sigmas/sigmas.txt')
    path_training_data = os.path.join(data_dir, 'motor_cap2/sigmas/training_samples.txt')

    data = np.loadtxt(path_training_data)
    X = [d[:-1] for d in data]
    Y = [d[-1] for d in data]
    X = np.array(X).reshape(-1, 6)
    Y = np.array(Y).reshape(-1, 1)
    sigmas = np.loadtxt(path_sigmas)
    sigmas = np.sqrt(np.diag(sigmas))

    pw = KernelRegression(X=X, Y=Y, sigmas=sigmas)

    X = np.linspace(-0.02, 0.02, 201)
    Y = np.linspace(-0.012, 0.012, 201)
    coords_grid = np.array(np.meshgrid(X, Y))
    coords = coords_grid.T.reshape(-1, 2)

    grid_res = np.zeros((len(coords), 1))
    for i in range(len(grid_res)):
        test_point = np.zeros((6,))
        test_point[:2] = coords[i]
        grid_res[i] = pw.estimate(test_point.reshape(-1, ))
    grid_res = grid_res.reshape(coords_grid.T.shape[:-1]).T
    plt.imshow(grid_res)

    x_ticks = np.around(X, decimals=3)
    y_ticks = np.around(Y, decimals=3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.yticks(range(201)[::25], y_ticks[::25], fontsize=15)
    plt.xticks(range(201)[::25], x_ticks[::25], fontsize=15)
    plt.show()


def align_model_to_scene(data_dir):
    """
    Aling the model to the scene point cloud
    :param data_dir: Path to the robot manipulation dataset
    :return:
    """
    model = os.path.join(data_dir, 'motor_frame/model/model.pcd')
    scene = os.path.join(data_dir, 'motor_frame/scenes/pcd/0145_cloud.pcd')
    gt = os.path.join(data_dir, 'motor_frame/gt/0145_gt_full.txt')

    gt_mat = np.loadtxt(gt)
    model = o3d.io.read_point_cloud(model)
    scene = o3d.io.read_point_cloud(scene)

    model.transform(gt_mat)
    model = color_cloud(model, np.array([0.7, 0.0, 0.0]))
    o3d.visualization.draw_geometries([model + scene])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=False,
                        help="Path to the root folder of the robot manipulation dataset")
    args = parser.parse_args()
    print(f'Running 1D plot ...')
    model_1D(args.data_dir)
    print(f'Running 2D plot ...')
    model_2D(args.data_dir)
    print(f'Aligning model to scene ...')
    align_model_to_scene(args.data_dir)


if __name__ == '__main__':
    main()
