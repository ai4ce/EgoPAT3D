EgoPAT3D - Egocentric Action Target Prediction Dataset
==========

<p align="center">
  <img width="50%" height="50%" src="https://ai4ce.github.io/EgoPAT3D/img/home/VideoPicture.jpg">
</p>

<b>EgoPAT3D</b> contains multimodal data generated from RGBD first-person videos recorded using a helmet-mounted Azure Kinect depth camera. In each recording, the camera wearer reaches for, grabs, and moves objects randomly placed in a household scene.

Each recording features a different configuration of household objects within the scene. The dataset also contains scene point clouds for 3D registration, binary masks for hand presence frame detection, and hand pose inference results for each frame detected to contain a hand.  


### Table of Contents
* [Specifications](#specifications)
* [Access](#access-dataset)
* [Folder hierarchy](#dataset-folder-hierarchy)
* [Scene index](#scene-index)
* [Alternative access methods](#alternative-access)

### Specifications
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

### Access dataset:
[Dataset without raw .MKV recordings and individual pre-extracted RGB frames](https://drive.google.com/drive/folders/1WHCWQ3dVoBqz6lkJKzgVoDfU_NO0lcFw?usp=sharing)

[Raw .MKV recordings](https://drive.google.com/drive/folders/1cxisgjUK9afV9vr62L_m6Shb7Mfl39zc?usp=sharing)

### Dataset folder hierarchy
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

### Scene index
| Index | Scene | Index | Scene | Index | Scene |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | bathroomCabinet | 6 | kitchenCounter | 11 | pantryshelf |
| 2 | bathroomCounter | 7 | kitchenCupboard | 12 | smallbins |
| 3 | bin | 8 | kitchenSink | 13 | stovetop |
| 4 | desk | 9 | microwave | 14 | windowsillAC |
| 5 | drawer | 10 | nightstand | 15 | woodenTable |

### Alternative access
1. Direct download using the links above
2. Manual build using HPC (currently only available on NYU HPC)
<br/>Download the raw multimodal recording data and generate a local copy of the dataset on your NYU HPC /scratch space, following instructions provided in this repository's model_build folder. May take 1-3 days to build on NYU Greene depending on compute resource availability.

<br>
<p align="center">
  <a href="https://www.revolvermaps.com/livestats/5chmzyob7br/">Live Visitor Statistics (via RevolverMaps)</a> 
  <br><br>
  <img align="center" width="50%" height="50%" src="http://rf.revolvermaps.com/h/m/a/0/ff0000/256/10/5chmzyob7br.png">
</p>
