# Video-Representation-via-Multi-level-Optimization

Code of ICCV paper [Enhancing Self-supervised Video Representation Learning via Multi-level Feature Optimization](https://arxiv.org/abs/2108.02183). 

We propose a multi-level feature optimization framework to enhance low-level, mid-level and high-level feature learning and motion pattern modeling. 

## Requirements

- Python 3.7
- PyTorch 1.5
- torchvision

## Prepare Dataset

#### Kinetics-400

Download Kinetics-400 video data with download tools, e.g., https://github.com/cvdfoundation/kinetics-dataset, then extract the RGB frames and obtain the files like

```
kinetics-400
|----train
|--------category 1
|------------video 1
|----------------image_00001.jpg
|----------------image_00002.jpg
    		...
|----------------image_0000n.jpg
    	...
|------------video n
    ...
|--------category n
    	...
|----val
|--------category 1
|------------video 1
|----------------image_00001.jpg
|----------------image_00002.jpg
    		...
|----------------image_0000n.jpg
    	...
|------------video n
    ...
|--------category n
    	...
```

Then write a csv file to record the video frame paths and total number of frames of each video in Kinetics-400 like

```
path/to/video/frame,number_of_frames
/root/kinetics-400/train/category_1/video_1,frames_1
```

#### UCF-101

Download data, train/val splits and annotations from the official data provider, extract frames and write the csv file for training and validation set.

When preparing UCF-101 csv file used for pretrain, the csv file structure is the same as Kinetics-400. When preparing for downstream evaluation, the csv file contains one more item label

```
path/to/video/frame,number_of_frames,label
/root/ucf-101/train/category_1/video_1,frames_1,label_1
```

#### HMDB-51

Download data, train/val splits and annotations from the official data provider, extract frames and write the csv file for training and validation set. The csv file is only used for downstream evaluation not for pretrain.

## Training

By changing the data root path in `train.py` to adjust the pretraining dataset. Note that the batchsize and workers hyper-parameters are for each GPU process in distributed training.

```
python train.py [-h]
--seq                           number of frames in each clip
--sample                        number of clips extrated from each video
--img_dim                       spatial dimension of input clip
--cluster                       number of cluster centroids in SK cluster
--rate                          upper bound of frame sampling rate
--train_batch                   training batchsize for each GPU process
--workers                       num_workers for each GPU process
--epoch                         total pretraining epochs
--split                         split epoch for introducing graph constraint
--lr_decay                      epoch for learning rate dacay
--thres                         thershold in graph constraint inference
--csv_file                      csv file path of training data
--multiprocessing-distributed   activate distributed training multiprocessing

For Kinetics-400 pretrain:
python train.py --train_batch 64 --workers 8 --cluster 1000 --epoch 100 --lr_decay 70 --thres 0.05 --csv_file kinetics.csv --multiprocessing-distributed --split 20

For UCF-101 pretrain:
python train.py --train_batch 64 --workers 8 --cluster 200 --epoch 300 --lr_decay 200 -thres 0.05 --csv_file ucf_pretrain.csv --multiprocessing-distributed --split 50
```

## Evaluation

We follow the evaluation steps in previous works on video representation learning, e.g., [CoCLR](https://github.com/TengdaHan/CoCLR).

## Visualization

To visualize how much temporal cues are contained in each spatial area, we use CAM visualization approach in `./visualization`.  Specifically, we first load the pretrained backbone parameters and freeze them, and train a linear classifier head without bias to discriminate the original video clip and temporally reversed one, as shown in the function `train` in `./visualization/main_temporal.py`. After that, we could obtain the CAM visualization result by calling `returncam` function. The CAM results show how each spatial area in the video clip provides evidence to help discriminate whether in normal order or reverse.