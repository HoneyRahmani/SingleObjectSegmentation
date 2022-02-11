
## Table of Content
> * [Single Object Segmentation- Segmentation of the Fetal Head in ultrasound images- Pytorch](#SingleObjectSegmentation-SegmentationoftheFetalHeadinultrasoundimages-Pytorch)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)

# Single Object Segmentation- Segmentation of the Fetal Head in ultrasound images- Pytorch
## About the Project
This project focuses on developing a deep-learning model using PyTorch to outline the boundary of one target object in an image by a binary mask. Thus, task is single-object segmentation and goal is to predict a binary mask.
In this project, automatically segmenting a fetal head in ultrasound image is performed.

![recipe](https://user-images.githubusercontent.com/75105778/153649787-46a34ba4-83b7-4a1f-9e9f-87babf9a3d95.jpg)


## About Database

Find dataset in https:/ / zenodo. org/ record/ 1322001#. XcX1jk9KhhE

## Built with
* Pytorch
* Model is an encoder-decoder
* Combined Loss function (The binary cross-entropy (BCE) and dice metric).
* Adam optimizer.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch

## Examples

![s1](https://user-images.githubusercontent.com/75105778/153672646-b2861baf-a99a-4d53-bb3e-d95dff02ca34.png)

![s2](https://user-images.githubusercontent.com/75105778/153673413-4829a662-f856-4da6-b547-3a08197ca764.png)

![s3](https://user-images.githubusercontent.com/75105778/153673432-0fc435e8-5518-41b6-afaa-67ebec8fa129.png)

