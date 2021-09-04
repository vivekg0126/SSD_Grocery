# Single Shot Detection with Grocery Dataset

### Downloading and prepraring data

Follow below steps after downloading the dataset from 

1) Download dataset from below link and store it in directory(dset_root)
``` 
https://github.com/gulvarol/grocerydataset 
https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz
```
2) Download annotations from below link and save as _annotation.txt_ inside dset_root
``` https://github.com/gulvarol/grocerydataset/blob/master/annotation.txt ```

Problem: identify/find the objects kept in the grocery rack of the super mart.

The objective is to run single shot detection with single anchor box.
This poses a different set of challenges to the problem.

``` First of all why do we use multiple anchors per feature map cell? ```

The purpose of using multiple anchors per feature map cell is to capture object of different aspect ratios and sizes. 
For each cell when we draw multiple anchor boxes there is a 
good chance that one of them might overlap with ground truth bounding box. 
Hence it helps in training and detection of object of different
scales and aspect ratios.

```Does this problem require multiple anchors? Please justify your answer. ```

First we do some analysis on the dataset, we can observe that the objects in the dataset have mean aspect ratio of 0.66 with deviation of 0.07, 
this means we know which aspect ratio to choose for our prior boxes. So we don't need multiple aspect ratio for the boxes. 
But we still need to handle the scaling of prior boxes since the objects in dataset are present at different scales.
So we will try to detect the boxes at aspect ratio of 0.66 on different scales

### Single shot detection (SSD) implementation on grocery dataset.

### Prerequisites

Install python version 3.6 along with following packages 

```
numpy
pytorch 1.2.0
torchvision 0.4
tqdm
PIL
cv2
```



### Installing Project

Extract the project and run train.py for training the dataset and eval.py to evaluate the model

#### Argument to train.py

* data : dataset root directory (compulsory input)
* object_json : json file of preprocessed object 
* checkpoint : checkpoint name to resume the training
* batch_size : batch size for training, default is 32
* iterations : number of iterations to train, default is set to 2000
* lr : learning rate, default is 0.001
* use_gpu : True if using gpu for training 
* model_save : to save the model for checkpoint


``` python train.py -<arguement> <arguement value>```

#### Argument to eval.py


``` python eval.py -<arguement> <arguement value>```

For Example:
```
python train.py -data <path to data root directory> -batch_size <batch size for training>
python eval.py -data <path to data directory> -checkpoint <checkpoint path to load the weights>
```
```
python train.py -data /home/vivek/dataset_root -batch_size 32
```


## Dataset Preparation:

### Dataset Description:

The Dataset home directory contains 3 subdirectries
1) BrandImages, 2) ProductImages, and 3) ShelfImages

We are using ShelfImages for this dataset. The directory also contains annotation.txt file which contains bounding box coordinates
for all the images in ShelfImages. ShelfImages contains train and test folders which contains images for corresponding split.

### Preprocessing:


As preprocessing step we take the data directory as input to preprocess_data() method in utils.py file. It generates All_objects.json
file in the project directory. This file contains train and test data information in following format:

```
{
"train": [
    {
      "name": "<jpeg file name>",
      "path": "<fully qualified file path>",
      "objects": 15,
      "label": [],
      "classes": [],
      "boxes": [],
    }
}
```

* train and test dictionary contains images from the training and test dataset respectively
* 'label' is dummy label provided to mention that this bounding box contains an object. This is done to add a condition to output object labels evaluation script, so that image with no object can be identified.
* 'objects' denotes number of objects/bounding boxes present in the image.
* 'boxes' contains bounding box coordinates for objects in image
* 'classes' are the object classes provided by the dataset


### GroceryDataset class in datasets.py

The GroceryDataset class takes preprocessed json file as input. It filters 'train' and 'test' data from the file.
The json file contains bounding boxes in format (x_min, y_min, width, height) format which is convert to (x_min, y_min, x_max, y_max)
format by convert boxes method in the class.
The getitem method of the class extract bounding boxes, convert them and send them with image to apply augmentations.
These augmentations are discussed below in detail. The dataset defines 'detection_collate' method which accumulates the
data from getitem and delivers it as a batch.

#### Transformations

*Normalize* : The image is normalized to the dataset mean and deviation with (x-mu)/sigma

*photometric_distort* : This method changes brightness, contrast, saturation and hue of the image with 50% probability. 

*flip* : The images are horizontally flipped, the bounding boxes are also accordingly changed.

*expand* : Here we perform zooming out of the image over a filler canvas. The attempt was to identify small objects.

*random_crop* : Perform random crop of the images, some objects might get cut off from the image due to this. This
augmentation help in detecting large images. This is implemented as suggested in the SSD paper.

## The Single Shot Detection Network

We have used VGG-16 as base network with weights pretrained on ImageNet. In SSD paper the authors have taked output from 'conv4_3' (38x38) and
used FC6 and FC7 on 'conv5_3' of standard VGG convolutional layers. They used extra convolutions to obtain 'conv8_2'(10x10), 'conv9_2'(5x5), 'conv10_2'(3x3) and 
'conv11_2'(1x1)  from output of 'FC7' feature maps.
Here (d x d) represents dimensions of the feature map from the conv layers.

Instead of using FC6 and FC7, we have used atrous convolutions in VGG base.
For this dataset the feature map dimension of (3x3) and (1x1) is less likely to discover any relavant underlying objects, since most of the
objects in the dataset are small. So we have omitted the output conv10_2 and conv11_2 layers in the dataset.
 
 ### Network Implementation (Model.py)

 The 'SSD' class module encapsulates 4 submodules and carries forward pass for whole of single shot detector.

1) VGG base: this is the base vgg model with pretrained weights providing us with the base feature maps.

2) Auxiliary Convolutions: The extra convolutions to obtain 'conv8_2' and 'conv9_2' feature maps.

3) Prediction Convolutions: Here we define convolutions to predict bounding boxes and their associated classes. 
As per the condition mentioned we are predicting only one bounding box per feature map cell. Since we are working with 
2 classes (object & non-object), number of classes is 2.

4) Creating prior boxes: 

Anchor/Prior boxes are generated using the conept of figuring out width and height of images using scale and aspect ratio.
If 'w' be the width, 'h' be the height, 's' being the scale and 'a' be the aspect ratio. 
Then w*h=s^2 and w/h=a
Solving these two equation yields width and height of the anchor box. 

For each feature map cell we create prior box coordinates in the format (c_x, c_y , w, h). Here c_x and c_y are the center
coordinates of the prior box. 'w' and 'h' are width and height of boxes created via solving above two equations.

### Detect method ( in SSD class)

In Detect object we pass on predicted locations and predicted scores along with minimum_score, max_overlap, and top-k values.

1) Predicted location and scores are passes through non maximal suppression where bounding boxes pointing to same objects are
suppressed. So that each prediction point to different objects.

2) minimum_score is used to consider objects with class score greater than this. Since we are doing two class classification
the minimum score must be greater than 0.5

3) max_overlap is used in NMS to suppress the boxes with IOU greater than max_overlap. We have set this value to 0.8



## Anchor Box Tuning

The anchor boxes are very crucial for training since they are mapped to the ground truth boxes during training.
Since we are allowed to use only one anchor box per feature map cell, this came out to be one of the most challenging task for this assignment. 

Initial stage of dataset analysis shows that the mean aspect ratio for the trainig dataset is 0.66 with standard deviation of 0.03.

For finding correct value for different scale we did empircal analysis with multiple scale values for different feature maps.
We calculated mean IOU for all the ground truth bounding boxes with prior boxes generated via different scaling and aspect ratio values.
We observed that for lower scale values slightly lower aspect ratios generated better IOUs with ground truth boxes.

## Hyperparameter Tuning

We have used following hyper parameters for training the model

* batch size : 32
* learning rate : 0.001
* decay learning rate : 0.1
* momentum : 0.9
* iterations : 2500
* aspect ratio : 0.66
* scale : ```{'conv4_3': 0.1, 'conv7': 0.21, 'conv8_2': 0.255, 'conv9_2': 0.30}```


## Q&A





### Reason for low mAP

We have obtained current mAP of 0.45. We have emperically found the different scaling factors for different feature maps.
We are of the view that the results can be improved by considering one more feature map from the base VGG between 'conv5_3' and 'FC7'.
This will provide some bounding boxes between (38x38) and (19x19), helping us in dealing with the scaling problem.