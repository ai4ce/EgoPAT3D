EgoPAT3D: Egocentric Prediction of Action Target in 3D [CVPR 2022]
==========
<p> 
  <b><a href="https://scholar.google.com/citations?user=i_aajNoAAAAJ&hl=en">Yiming Li*</a></b> ,
  <b><a href="https://ziangcao0312.github.io/">Ziang Cao*</a></b> , 
  <b><a href="https://www.linkedin.com/in/andrew-s-liang/">Andrew Liang</a></b>, 
  <b><a href="https://www.linkedin.com/in/benjamin-s-liang/">Benjamin Liang</a></b>, 
  <b><a href="https://www.linkedin.com/in/luoyao-chen/">Luoyao Chen</a></b>, 
  <b><a href="https://scholar.google.com/citations?user=DmahiOYAAAAJ">Hang Zhao</a></b>, 
  <b><a href="https://scholar.google.com/citations?user=YeG8ZM0AAAAJ&hl=en">Chen Feng</a></b>
</p>

<!--  <p style="font-size:small">* denotes equal contribution </p> -->

<p align="center">
  <img width="100%" height="50%" src="https://github.com/ai4ce/EgoPAT3D/blob/gh-pages/img/home/scene.jpg">
</p>

## News

[2022.07] Our dataset EGOPAT3D 1.0 is available [here](https://ai4ce.github.io/EgoPAT3D/).

[2022.03] Our paper is available on [arxiv](https://arxiv.org/pdf/2203.13116.pdf).

[2022.03] EgoPAT3D is accepted at CVPR 2022.

## Abstract

We are interested in anticipating as early as possible the target location of a person's object manipulation action in a 3D workspace from egocentric vision. It is important in fields like human-robot collaboration, but has not yet received enough attention from vision and learning communities. To stimulate more research on this challenging egocentric vision task, we propose a large multimodality dataset of more than 1 million frames of RGB-D and IMU streams, and provide evaluation metrics based on our high-quality 2D and 3D labels from semi-automatic annotation. Meanwhile, we design baseline methods using recurrent neural networks (RNNs) and conduct various ablation studies to validate their effectiveness. Our results demonstrate that this new task is worthy of further study by researchers in robotics, vision, and learning communities.

## Raw Data
<b>EgoPAT3D</b> contains multimodal data generated from RGBD first-person videos recorded using a helmet-mounted Azure Kinect depth camera. In each recording, the camera wearer reaches for, grabs, and moves objects randomly placed in a household scene.

Each recording features a different configuration of household objects within the scene. The dataset also contains scene point clouds for 3D registration, binary masks for hand presence frame detection, and hand pose inference results for each frame detected to contain a hand.  

#### Table of Contents
* [Specifications](#specifications)
* [Access](#access-dataset)
* [Folder hierarchy](#dataset-folder-hierarchy)
* [Scene index](#scene-index)
* [Alternative access methods](#alternative-access)

#### Specifications
* 15 household scenes (see scene index table below)
* 15 point cloud files (one for each scene)
* 150 total recordings (10 recordings in each scene, with different object configurations in each recording)
* 15000 hand-object actions (100 per recording)
* ~600 min of RGBD video (~4 min per video)
* ~1,080,000 RGB frames at 30 fps
* ~900,000 hand action frames (assuming ~2 seconds per hand-object action)

Data modalities:
* RGB (.png), depth, IR, IMU, temperature frames [compressed in multiple channels in Matroska format](https://docs.microsoft.com/en-us/azure/kinect-dk/record-file-format) (.mkv) 
* RGB color frames (.png) extracted from the Matroska file
* RGB videos (.mp4) of the color frames from each recording
* Point clouds (.ply) of each scene, produced using [AzureKinfu](https://github.com/microsoft/Azure-Kinect-Samples/tree/master/opencv-kinfu-samples)  
* Labeled hand/action frames (.txt)
* Hand pose inference results using [Google MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) (.txt)  

Azure Kinect recording specifications:  
|  | Color camera | Depth camera |
| ------------- | ------------- | ------------- |
| Resolution | 3840x2160 (4K) | 512x512 |
| Frames per second | 30 | 30 |
| Recording mode | -- | WFOV 2x2 binned |  

#### Access dataset:
[Dataset without raw .MKV recordings and individual pre-extracted RGB frames](https://drive.google.com/drive/folders/1WHCWQ3dVoBqz6lkJKzgVoDfU_NO0lcFw?usp=sharing)

[Raw .MKV recordings](https://drive.google.com/drive/folders/1cxisgjUK9afV9vr62L_m6Shb7Mfl39zc?usp=sharing)

#### Dataset folder hierarchy
```bash
Dataset/
    ├──1/ # one folder containing multimedia data for each of the 15 scenes (see scene index table below)
        ├── bathroomCabinet_1/
            ├── hand_frames/
                ├── clip_ranges.txt
                ├── hand_landmarks.txt
                └── masks.zip
            └── rgb_video.mp4
        ├── bathroomCabinet_2/ # all sceneName_# directories share the same folder/subdirectory structure
        ├── ...
        ├── bathroomCabinet_10/
        └── bathroomCabinet.ply
    ├──2/ # all 15 scene # directories share the same folder/subdirectory structure
        └── ...
    .
    .
    .
    
    └── 15
```

#### Scene index
| Index | Scene | Index | Scene | Index | Scene |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | bathroomCabinet | 6 | kitchenCounter | 11 | pantryshelf |
| 2 | bathroomCounter | 7 | kitchenCupboard | 12 | smallbins |
| 3 | bin | 8 | kitchenSink | 13 | stovetop |
| 4 | desk | 9 | microwave | 14 | windowsillAC |
| 5 | drawer | 10 | nightstand | 15 | woodenTable |

#### Alternative access
1. Direct download using the links above
2. Manual build using HPC (currently only available on NYU HPC)
<br/>Download the raw multimodal recording data and generate a local copy of the dataset on your NYU HPC /scratch space, following instructions provided in this repository's model_build folder. May take 1-3 days to build on NYU Greene depending on compute resource availability.


## Groundtruth Generation
Our annotation can be semi-automatic with several off-the-shelf machine learning algorithms, thus is quite efficient. Given a recording, we manually divide it into multiple action clips. To localize the 3D target in each clip, we use the following procedures. Firstly, we take the last frame of each clip based on the index provided by the manual division. Secondly, we use an off-the-shelf hand pose estimation model to localize the hand center in the last frame of each clip. Thirdly, we use colored point cloud registration to calculate the transformation matrices between the adjacent frames. Finally, for each clip, we transform the hand location in the last frame to historical frames according to the results of the third step, and the transformed locations can describe the 3D action target location in each frame's coordinate. Detailed procedures are presented in our paper.

## Baseline Method 
To obtain the training data from the raw datastreams, please follow the [instructions](https://github.com/ai4ce/EgoPAT3D/tree/main/preprocessing). To use our processed data, please follow the baseline method [implementation](https://github.com/ai4ce/EgoPAT3D/tree/Predictor-code).


## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{Li_2022_CVPR,
      title = {Egocentric Prediction of Action Target in 3D},
      author = {Li, Yiming and Cao, Ziang and Liang, Andrew and Liang, Benjamin and Chen, Luoyao and Zhao, Hang and Feng, Chen},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2022}
}
```

<!-- <br>
<p align="center">
  <a href="https://www.revolvermaps.com/livestats/5chmzyob7br/">Live Visitor Statistics (via RevolverMaps)</a> 
  <br><br>
  <img align="center" width="50%" height="50%" src="http://rf.revolvermaps.com/h/m/a/0/ff0000/256/10/5chmzyob7br.png">
</p>
 -->
