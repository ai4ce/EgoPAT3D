# A Baseline for Egocentric Action Target Prediction

## Environment setup
This code has been tested on Ubuntu 20.04, Python 3.7.0, Pytorch 1.9.0, CUDA 11.2.
Please install related libraries before running this code. The detailed information is included in `./requirement.txt`.

## Test and Validate
Download the pre-trained [model](https://drive.google.com/file/d/1u8b4xcLlevOmwXP-GTImAPDzfISAnNUR/view?usp=sharing) and set the checkpoints directory.


```
python test.py 	                          \
	--model_name LSTM-based           \ # tracker_name
	--checkpoint ./experiment/LSTM-based.pth   #model_path
	--datapath ./data_path   #data_path
```

```
python validate.py 	                          \
	--model_name LSTM-based           \ # tracker_name
	--checkpoint ./experiment/LSTM-based.pth   #model_path
	--datapath ./data_path   #data_path
```

The testing and validating result will be saved in the `./results/model_name` directory.

## Train

### Prepare training datasets

Download the datasets from our provided link and put it into `./benchmark/`.

#### Dataset folder hierarchy
```bash
Dataset/
    ├──annotrain/ # The annotation of train set
        ├── bathroomCabinet/ # The name of different scenes
            ├── bathroomCabinet_1.txt/ # The groundtruth of each clip
            ├── bathroomCabinet_2.txt/
            ├── bathroomCabinet_3.txt/
            └── bathroomCabinet_4.txt/
                
        ├── bathroomCounter/ 
        ├── ...
        └── nightstand/
    ├──annonoveltest_final/ # The annotation of test set (unseen scenes)
        └── ...
    ├──annotest_final/ # The annotation of test set (seen scenes)
    ├──annovalidate_final/ # The annotation of validation set
    ├──sequences/ # The pointcloud and imu data of each scene
        ├── bathroomCabinet/ 
            ├── bathroomCabinet_1/ 
                 ├── pointcloud/ # The pointcloud files
                 ├── transformation/ # The odometry files
                 └── data.txt/ # The imu data of bathroomCabinet_1
            .
            .
    
            └── bathroomCabinet_6/
        .
        .
    
        └── woodenTable
```

### Train a model
To train the predictor model, run `train.py` with the desired configs:

```
python train.py 
```




## Acknowledgement
The code of predictor is implemented based on [PointConv](https://github.com/DylanWusee/pointconv_pytorch). We would like to express our sincere thanks to the contributors.