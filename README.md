# [The Easiest Baseline of Egocentric Action Target Predictor]

## Environment setup
This code has been tested on Ubuntu 20.04, Python 3.7.0, Pytorch 1.9.0, CUDA 11.2.
Please install related libraries before running this code. The detailed information is included in `./requirement.txt`.

## Test and Validate
Download the pre-trained model and set the checkpoints directory.


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

Download the datasets from our provided link.


### Train a model
To train the SiamAPN model, run `train.py` with the desired configs:

```
python train.py 
```




## Acknowledgement
The code of predictor is implemented based on [PointConv](https://github.com/DylanWusee/pointconv_pytorch). We would like to express our sincere thanks to the contributors.