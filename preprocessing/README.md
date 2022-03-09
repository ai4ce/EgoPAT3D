# Tools for preprocessing the original data

##

## 1. mkv2d2rgbrgb.py

This file aims to split the MKV files into RGB images and depth images. Please note that because of the difference between the resolution of RGB images and depth images, we transform the resolution of Depth to the resolution common to RGB images, i.e., d2rgb.

## 2. rgbd2pointcloud.py

This file aims to transform RGBD images to point clouds.

## 3. realclip2gt.py

This file aims to generate the ground truth (in the first frame) based on the location of the target in the last frame. Besides, it will output the visualization  of the ground truth for our validation.

## imu 

An example about generating imu data from MKV files.

```

cd imu

bash ./build/produce.sh

```