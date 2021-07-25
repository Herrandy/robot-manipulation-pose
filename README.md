# Benchmarking Pose Estimation for Robot Manipulation
Download the robotic manipulation dataset and run simple example code 
```bash
# clone the repository
$ git clone git@github.com:Herrandy/pose-esimation.git && cd pose-esimation

# download and extract the dataset (~2.8GB)
$ wget -O robot-manipulation-dataset.tar.xz https://zenodo.org/record/5136512/files/robot-manipulation-dataset.tar.xz?download=1
$ tar -xvf robot-manipulation-dataset.tar.xz

# run example plots
$ python3 plotting.py --data-dir <path-to-dataset>
```
Calculate grasp success rate for your pose estimates
```bash
$ python3 calculate_grasp_success.py --estimate-dir <path-to-estimates> --model-data-dir <path-to-model-data>
```
* `estimate-dir`: Path to a directory containing the estimated object poses as text files, e.g. `0000_pred.txt`, `0001_pred.txt, ...`. 
  The estimates must be 4x4 transformation matrices.
* `model-data-dir`: Path to a directory containing the model data, e.g. `robot-manipulation-dataset/cranfield`. 

## Robot manipulation dataset
The dataset is stored to Zenodo open-access repository and can be downloaded from here: [link](https://zenodo.org/record/5136512#.YP2NnjpRU5k).
The dataset has the following directory structure 
```
.
├── object_id
│   ├── scenes
│   │   ├── color -- 0000_color.jpg, 0001_color.jpg,...
│   │   ├── depth -- 0000_depth.png, 0001_depth.png,...
│   │   └── pcd -- 0000_cloud.pcd, 0001_cloud.pcd, ...
│   ├── gt -- 0000_gt_full.txt, 0001_gt_full.txt, ...
│   ├── model -- model.pcd
│   └── sigmas -- sigmas.txt, training_samples.txt
├── object_id
│   ├── scenes
:   :
```
* `object_id`: the target object identifier e.g. `motor_frame`.
* `scenes`: the test data in three different data formats: color image (rgb), depth image (depth) and colored point cloud (pcd).
* `gt`: ground truth data to align the model to the scene point cloud.
* `model`: 3D model of the target object.
* `sigmas`: the kernel bandwidth values for each manipulated object, and the collected training samples used for finding the optimal value for the bandwidths.  

## Contact
This is the reference implementation for the paper:

**_Benchmarking Pose Estimation for Robot Manipulation, In Robotics and Autonomous Systems, 2021._** _A. Hietanen, J. Latokartano, A. Foi, R. Pieters, V. Kyrki, M. Lanz and J.-K. Kämäräinen_, ([PDF](https://www.sciencedirect.com/science/article/pii/S0921889021000956))

If you find this code useful in your work, please consider citing:
```tex
@article{hietanen2021pose,
title = {Benchmarking pose estimation for robot manipulation},
journal = {Robotics and Autonomous Systems},
volume = {143},
pages = {103810},
year = {2021},
issn = {0921-8890},
doi = {https://doi.org/10.1016/j.robot.2021.103810},
url = {https://www.sciencedirect.com/science/article/pii/S0921889021000956},
author = {Antti Hietanen and Jyrki Latokartano and Alessandro Foi and Roel Pieters and Ville Kyrki and Minna Lanz and Joni-Kristian Kämäräinen},
keywords = {Object pose estimation, Robot manipulation, Evaluation}
}
```