Egocentric Intent Dataset
===========
### Purpose
### Specifications
### Dataset folder hierarchy
```bash
Dataset/
    ├──1/ # one folder containing multimedia data for each of the 15 scenes (see scene index table below)
        ├── bathroomCabinet_1/
            ├── hand_frames/
                ├── clip_ranges.txt
                ├── hand_landmarks.txt
                └── masks.zip
            ├── bathroomCabinet_1.mkv
            ├── color_frames.zip
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
| Index | Scene |
| ------------- | ------------- |
| 1 | bathroomCabinet |
| 2 | bathroomCounter |
| 3 | bin |
| 4 | desk |
| 5 | drawer |
| 6 | kitchenCounter |
| 7 | kitchenCupboard |
| 8 | kitchenSink |
| 9 | microwave |
| 10 | nightstand |
| 11 | pantryShelf |
| 12 | smallBins |
| 13 | stoveTop |
| 14 | windowsillAC |
| 15 | woodenTable |

### Access
1. Direct download
<br/>Simplest method. Currently unavailable, but a link will be up soon for this option once dataset upload is complete.
2. Manual build using HPC
<br/>Download the raw multimodal recording data and generate a local copy of the dataset on your NYU HPC /scratch space, following instructions provided in this repository's model_build folder. May take 1-3 days to build on NYU Greene depending on compute resource availability.
